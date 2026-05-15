from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from config import AppConfig, Config
from pipeline.orchestrator import PiiMaskingPipeline
from utils.logger import configure_logging, get_logger

app = typer.Typer(name="pii-masker", help="PII masking pipeline — pattern + trained GLiNER.")
console = Console()
log = get_logger(__name__)


def _build_app_config(config_path: Path) -> AppConfig:
    settings = Config(entities_config_path=config_path)
    configure_logging(settings.log_level)
    return AppConfig(settings=settings)


@app.command("mask-text")
def mask_text(
    text: str = typer.Option(..., "--text", help="Input text"),
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
    output: str = typer.Option("table", "--output", help="table | json"),
    show_stats: bool = typer.Option(False, "--show-stats"),
) -> None:
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)
    result = pipeline.process_sync(text)

    if output == "json":
        console.print_json(result.model_dump_json())
        return

    table = Table(title="PII Detection Results")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("Masked", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Source", style="dim")

    for span in result.detected_spans:
        masked = next(
            (m.masked for m in result.masked_spans
             if m.entity_id == span.entity_id and m.original == span.text),
            "N/A",
        )
        table.add_row(
            span.display_name,
            span.text,
            masked,
            f"{span.confidence:.2f}",
            span.source,
        )

    console.print(table)
    s = result.stats
    console.print(
        f"[bold]{s.spans_total} entities detected[/bold], "
        f"{s.total_ms:.0f}ms total "
        f"(pattern {s.pattern_ms:.0f}ms | ner {s.ner_ms:.0f}ms)"
    )

    if show_stats:
        console.print(result.stats.model_dump_json(indent=2))

    console.rule()
    console.print("[bold]Masked text:[/bold]")
    console.print(result.masked_text)


@app.command("mask-file")
def mask_file(
    input: Path = typer.Option(..., "--input"),
    output: Path = typer.Option(..., "--output"),
    format: str = typer.Option("text", "--format", help="json | text"),
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)
    text = input.read_text(encoding="utf-8")
    result = pipeline.process_sync(text)

    if format == "json":
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        output.write_text(result.masked_text, encoding="utf-8")

    console.print(
        f"[green]✓[/green] Masked {result.stats.spans_total} entities "
        f"in {result.stats.total_ms:.0f}ms → {output}"
    )


@app.command("mask-batch")
def mask_batch(
    input_dir: Path = typer.Option(..., "--input-dir"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workers: int = typer.Option(4, "--workers"),
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)

    files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.json"))
    if not files:
        console.print("[yellow]No .txt or .json files found.[/yellow]")
        return

    async def _run() -> None:
        results = await pipeline.process_batch(
            [f.read_text(encoding="utf-8") for f in files],
            max_concurrency=workers,
        )
        for file, result in zip(files, results):
            (output_dir / file.name).write_text(result.masked_text, encoding="utf-8")

    asyncio.run(_run())
    console.print(f"[green]✓[/green] Processed {len(files)} files → {output_dir}")


@app.command("list-entities")
def list_entities(
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    import yaml
    app_config = _build_app_config(config)

    table = Table(title="Configured PII Entities")
    table.add_column("ID", style="cyan")
    table.add_column("Display Name")
    table.add_column("Enabled", justify="center")
    table.add_column("Strategy")
    table.add_column("Format")

    with open(config, encoding="utf-8") as fh:
        raw_entities = yaml.safe_load(fh).get("entities", [])

    for entity in raw_entities:
        masking = entity.get("masking", {})
        table.add_row(
            entity.get("id", ""),
            entity.get("display_name", ""),
            "[green]✓[/green]" if entity.get("enabled", True) else "[red]✗[/red]",
            masking.get("strategy", ""),
            masking.get("format", ""),
        )

    console.print(table)


if __name__ == "__main__":
    app()
