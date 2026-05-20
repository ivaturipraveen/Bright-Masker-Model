from __future__ import annotations

import asyncio
import json
import re
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*byte fallback.*", category=UserWarning)

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig, Config
from pipeline.orchestrator import PiiMaskingPipeline
from utils.logger import configure_logging, get_logger

log = get_logger(__name__)

_settings: Config | None = None
_app_config: AppConfig | None = None
_pipeline: PiiMaskingPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _app_config, _pipeline
    _settings = Config()
    configure_logging(_settings.log_level)
    _app_config = AppConfig(settings=_settings)
    _pipeline = PiiMaskingPipeline(_app_config)
    await _pipeline._ensure_initialized()

    ner_model = _settings.fine_tuned_model_path or _settings.gliner_model_name
    log.info(
        "server_ready",
        ner_model=ner_model,
        spacy=_settings.spacy_model_name,
        entities=len(_app_config.entities),
        bright_shield=bool(_settings.bright_shield_base_url),
    )

    if _settings.warmup_text.strip():
        try:
            await _pipeline.process(_settings.warmup_text)
            log.info("warmup_complete")
        except Exception as _e:
            log.warning("warmup_failed", error=str(_e)[:200])

    yield
    log.info("server_shutdown")


app = FastAPI(title="PII Masker API", version="3.1.0", lifespan=lifespan)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MaskRequest(BaseModel):
    text: str


class EntityCreateRequest(BaseModel):
    id: str
    display_name: str
    description: str = ""
    gliner_label: str = ""
    policy: list[str] = ["general"]
    priority: int = 5
    confidence_threshold: float = 0.55
    masking_format: str = "[ENTITY {n}]"
    pattern: Optional[str] = None


class SpanInfo(BaseModel):
    entity_id: str
    display_name: str
    original: str
    masked: str
    confidence: float
    source: str
    strategy: str


class MaskResponse(BaseModel):
    masked_text: str
    original_text: str
    spans: list[SpanInfo]
    stats: dict
    response_time_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mask_response(result, app_config: AppConfig, response_time_ms: float) -> MaskResponse:
    masked_lookup = {(ms.entity_id, ms.original): ms.masked for ms in result.masked_spans}
    spans: list[SpanInfo] = []
    for span in result.detected_spans:
        entity_cfg = app_config.entities_by_id.get(span.entity_id)
        strategy = entity_cfg.masking.strategy.value if entity_cfg else "unknown"
        masked_val = masked_lookup.get((span.entity_id, span.text), span.text)
        spans.append(SpanInfo(
            entity_id=span.entity_id,
            display_name=span.display_name,
            original=span.text,
            masked=masked_val,
            confidence=round(span.confidence, 4),
            source=span.source,
            strategy=strategy,
        ))
    return MaskResponse(
        masked_text=result.masked_text,
        original_text=result.original_text,
        spans=spans,
        stats=result.stats.model_dump(),
        response_time_ms=round(response_time_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _branding_file(filename: str) -> FileResponse:
    path = _static_dir / filename
    if not path.is_file():
        raise HTTPException(404, f"Missing static asset: {filename}")
    return FileResponse(str(path), media_type="image/webp")


@app.get("/branding/logo.webp", include_in_schema=False)
async def branding_logo():
    return _branding_file("brightcone-logo.webp")


@app.get("/branding/wordmark.webp", include_in_schema=False)
async def branding_wordmark():
    return _branding_file("brightcone-wordmark.webp")


@app.get("/", include_in_schema=False)
async def index():
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "PII Masker API — see /docs"})


@app.get("/health")
async def health():
    if _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    ner_model = _app_config.fine_tuned_model_path or _app_config.gliner_model_name
    return {
        "status": "ok",
        "ner_model": ner_model,
        "spacy_model": _app_config.spacy_model_name,
        "entities_loaded": len(_app_config.entities),
        "ner_threshold": _app_config.gliner_threshold,
    }


_DEFAULT_REGISTRY = {
    "policy_priority": ["hipaa", "pci_dss", "law_enforcement", "transportation", "general"],
    "categories": [
        {"id": "hipaa", "label": "HIPAA Safe Harbor", "description": "18 PHI identifiers", "icon": "🛡"},
        {"id": "pci_dss", "label": "PCI-DSS v4.0", "description": "Cardholder & authentication data", "icon": "💳"},
        {"id": "general", "label": "General PII", "description": "Credentials & sensitive attributes", "icon": "🔐"},
        {"id": "law_enforcement", "label": "Law Enforcement", "description": "CJIS & criminal justice records", "icon": "⚖"},
        {"id": "transportation", "label": "Transportation", "description": "Vehicle & transit identifiers", "icon": "🚗"},
    ],
}


def _primary_category(policies: list[str], policy_priority: list[str]) -> str:
    for policy in policy_priority:
        if policy in policies:
            return policy
    return policies[0] if policies else "general"


def _load_entity_registry() -> dict:
    if _app_config is None or _settings is None:
        raise HTTPException(503, "Pipeline not initialized")

    path = Path(_settings.entities_config_path)
    if not path.is_file():
        raise HTTPException(500, f"entities config not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    registry = {**_DEFAULT_REGISTRY, **(data.get("registry") or {})}
    policy_priority = registry.get("policy_priority") or _DEFAULT_REGISTRY["policy_priority"]
    categories = registry.get("categories") or _DEFAULT_REGISTRY["categories"]

    global_cfg = data.get("global", {})
    active_policies: set[str] = set(global_cfg.get("enabled_policies") or [])
    default_threshold = float(global_cfg.get("default_confidence_threshold", 0.85))

    entities_out: list[dict] = []
    for raw in data.get("entities") or []:
        policies = raw.get("policy") or ["general"]
        masking = raw.get("masking") or {}
        enabled = bool(raw.get("enabled", True))
        policy_ok = not active_policies or bool(set(policies) & active_policies)
        threshold = raw.get("confidence_threshold")
        if threshold is None:
            threshold = default_threshold

        entities_out.append({
            "id": raw.get("id", ""),
            "display_name": raw.get("display_name", ""),
            "description": raw.get("description", ""),
            "policy": policies,
            "category": _primary_category(policies, policy_priority),
            "priority": raw.get("priority", 5),
            "enabled": enabled,
            "is_active": enabled and policy_ok,
            "confidence_threshold": threshold,
            "gliner_label": raw.get("gliner_label"),
            "has_pattern": bool(raw.get("patterns")),
            "has_presidio": bool(raw.get("presidio_type")),
            "has_gliner": bool(raw.get("gliner_label")),
            "strategy": masking.get("strategy", "redact"),
            "format": masking.get("format", ""),
            "pattern_count": len(raw.get("patterns") or []),
        })

    active_count = sum(1 for e in entities_out if e["is_active"])
    by_category: dict[str, int] = {}
    for e in entities_out:
        by_category[e["category"]] = by_category.get(e["category"], 0) + 1

    return {
        "entities": entities_out,
        "categories": categories,
        "policy_priority": policy_priority,
        "total": len(entities_out),
        "active_count": active_count,
        "enabled_count": sum(1 for e in entities_out if e["enabled"]),
        "by_category": by_category,
        "pipeline_loaded": len(_app_config.entities),
    }


@app.get("/entities")
async def list_entities():
    return _load_entity_registry()


def _append_entity_to_yaml(req: EntityCreateRequest) -> None:
    """Append a new entity block to entities_config.yaml."""
    config_path = Path(_settings.entities_config_path) if _settings else Path("entities_config.yaml")
    label = req.gliner_label or req.display_name

    lines = [
        f"\n  - id: {req.id}",
        f"    display_name: {req.display_name}",
    ]
    if req.description:
        safe_desc = req.description.replace('"', '\\"')
        lines.append(f'    description: "{safe_desc}"')
    lines.append(f"    policy: [{', '.join(req.policy)}]")
    lines.append(f"    enabled: true")
    lines.append(f"    priority: {req.priority}")
    lines.append(f"    confidence_threshold: {req.confidence_threshold}")
    lines.append(f'    gliner_label: "{label}"')
    if req.pattern:
        safe_pat = req.pattern.replace("'", "\\'")
        lines.append(f"    patterns:")
        lines.append(f"      - '{safe_pat}'")
    lines.append(f"    masking:")
    lines.append(f"      strategy: redact")
    lines.append(f'      format: "{req.masking_format}"')

    block = "\n".join(lines) + "\n"
    with open(config_path, "a", encoding="utf-8") as f:
        f.write(block)


async def _reload_entity_config() -> None:
    """Reload entities from disk and rebuild NER label groups."""
    if _app_config is None or _pipeline is None:
        return
    await asyncio.to_thread(_app_config.load)
    if _pipeline.ner_layer is not None:
        await asyncio.to_thread(_pipeline.ner_layer._build_label_groups)


@app.post("/api/entities")
async def create_entity(req: EntityCreateRequest):
    if _app_config is None or _settings is None:
        raise HTTPException(503, "Pipeline not initialized")

    if not re.match(r'^[a-z][a-z0-9_]{1,63}$', req.id):
        raise HTTPException(400, "id must be lowercase letters/digits/underscores, start with a letter, 2-64 chars")
    if req.id in _app_config.entities_by_id:
        raise HTTPException(409, f"Entity '{req.id}' already exists")
    if not req.display_name.strip():
        raise HTTPException(400, "display_name is required")

    try:
        await asyncio.to_thread(_append_entity_to_yaml, req)
        await _reload_entity_config()
    except Exception as exc:
        raise HTTPException(500, f"Failed to save entity: {exc}") from exc

    return {
        "status": "created",
        "id": req.id,
        "total_entities": len(_app_config.entities),
    }


def _max_text_chars() -> int:
    if _settings is None:
        return 500_000
    return _settings.max_text_chars


@app.get("/api/config")
async def api_config():
    """Public UI settings (labels, proxy path, comparison flags) — all from .env."""
    settings = _settings or Config()
    entity_count = len(_app_config.entities) if _app_config else 0
    local_desc = settings.local_model_desc or (
        f"{entity_count} entity types · local inference · no LLM" if entity_count else "Local inference"
    )

    return {
        "app_title": settings.app_title,
        "page_title": settings.page_title,
        "comparison_subtitle": settings.comparison_subtitle,
        "local_model_name": settings.local_model_name,
        "local_model_badge": settings.local_model_badge,
        "local_model_desc": local_desc,
        "remote_model_name": settings.remote_model_name,
        "remote_model_badge": settings.remote_model_badge,
        "remote_model_desc": settings.remote_model_desc,
        "remote_model_offline_label": settings.remote_model_offline_label,
        "entity_count": entity_count,
        "bright_shield_proxy_path": settings.bright_shield_proxy_path,
        "comparison_enabled": bool(
            settings.bright_shield_base_url and settings.bright_shield_proxy_enabled
        ),
        "ui_health_timeout_ms": settings.ui_health_timeout_ms,
    }


@app.post("/mask", response_model=MaskResponse)
async def mask_text(request: MaskRequest):
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")
    max_chars = _max_text_chars()
    if len(request.text) > max_chars:
        raise HTTPException(413, f"text exceeds {max_chars:,} characters")

    t0 = time.perf_counter()
    result = await _pipeline.process(request.text)
    return _build_mask_response(result, _app_config, (time.perf_counter() - t0) * 1000)


def _register_bright_shield_proxy_routes(application: FastAPI, settings: Config) -> None:
    if not settings.bright_shield_base_url or not settings.bright_shield_proxy_enabled:
        return

    base_url = settings.bright_shield_base_url
    proxy_path = settings.bright_shield_proxy_path
    health_timeout = settings.bright_shield_health_timeout_sec
    mask_timeout = settings.bright_shield_mask_timeout_sec

    @application.get(f"{proxy_path}/health", include_in_schema=False)
    async def proxy_bright_shield_health():
        """Proxy the Bright Shield health check to avoid CORS issues in the browser."""
        try:
            async with httpx.AsyncClient(timeout=health_timeout) as client:
                r = await client.get(f"{base_url}/health")
                return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            raise HTTPException(502, f"Bright Shield unreachable: {e}") from e

    @application.post(f"{proxy_path}/mask", include_in_schema=False)
    async def proxy_bright_shield_mask(request: Request):
        """Proxy mask requests to Bright Shield and normalise to the local /mask response shape."""
        body = await request.body()
        try:
            payload = json.loads(body)
            original_text: str = payload.get("text", "")
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=mask_timeout) as client:
                r = await client.post(
                    f"{base_url}/rpc/text-detect",
                    json={"text": original_text, "aggregate_entities": True},
                    headers={"Content-Type": "application/json"},
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if r.status_code != 200:
                return JSONResponse(content=r.json(), status_code=r.status_code)

            results = r.json().get("results", [])

            # Build spans in the same shape renderPanel/buildDiff expect
            spans = [
                {
                    "entity_id": res.get("entity_type", "Unknown").lower().replace(" ", "_"),
                    "display_name": res.get("entity_type", "Unknown"),
                    "original": res.get("entity_text", ""),
                    "masked": f"[{res.get('entity_type', 'UNKNOWN').upper()}]",
                    "confidence": round(res.get("score", 0.0), 4),
                    "source": "bright_shield",
                    "strategy": "redact",
                }
                for res in results
            ]

            # Apply masks right-to-left to preserve character offsets
            masked_text = original_text
            for res in sorted(results, key=lambda x: x.get("start", 0), reverse=True):
                start, end = res.get("start", 0), res.get("end", 0)
                label = f"[{res.get('entity_type', 'UNKNOWN').upper()}]"
                masked_text = masked_text[:start] + label + masked_text[end:]

            return JSONResponse(content={
                "masked_text": masked_text,
                "original_text": original_text,
                "spans": spans,
                "stats": {"spans_total": len(spans), "language": "en"},
                "response_time_ms": round(elapsed_ms, 2),
            })
        except httpx.TimeoutException as e:
            raise HTTPException(504, "Bright Shield request timed out") from e
        except Exception as e:
            raise HTTPException(502, f"Bright Shield unreachable: {e}") from e


_bootstrap_settings = Config()
_register_bright_shield_proxy_routes(app, _bootstrap_settings)


@app.post("/mask/stream")
async def mask_text_stream(request: MaskRequest):
    """SSE — streams per-step progress then final result."""
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")
    max_chars = _max_text_chars()
    if len(request.text) > max_chars:
        raise HTTPException(413, f"text exceeds {max_chars:,} characters")

    queue: asyncio.Queue = asyncio.Queue()
    result_holder: list = []
    error_holder: list = []

    async def _run() -> None:
        try:
            res = await _pipeline.process(request.text, progress_queue=queue)
            result_holder.append(res)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            await queue.put(None)

    async def generate():
        t0 = time.perf_counter()
        asyncio.create_task(_run())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

        if error_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': str(error_holder[0])})}\n\n"
            return

        if result_holder:
            response_time_ms = (time.perf_counter() - t0) * 1000
            resp = _build_mask_response(result_holder[0], _app_config, response_time_ms)
            yield f"data: {json.dumps({'type': 'complete', 'result': resp.model_dump()})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
