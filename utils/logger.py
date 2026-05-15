import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import structlog

_BAR_WIDE = "=" * 60
_BAR_THIN = "-" * 60


def configure_logging(level: str = "INFO") -> None:
    fmt = os.getenv("LOG_FORMAT", "json").lower()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f" if fmt == "pretty" else "iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "pretty":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]
    else:
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


def log_line(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def log_request_start(req_id: str, chars: int, preview: str) -> float:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line()
    log_line(_BAR_WIDE)
    log_line("  PII MASKING REQUEST")
    log_line(_BAR_WIDE)
    log_line(f"  req_id  : {req_id}")
    log_line(f"  chars   : {chars:,}")
    log_line(f'  preview : "{preview}"')
    log_line(f"  started : {now}")
    log_line(_BAR_WIDE)
    return time.perf_counter()


def log_step_header(n: int, total: int, name: str, subtitle: str = "") -> None:
    sub = f"  ·  {subtitle}" if subtitle else ""
    log_line()
    log_line(_BAR_THIN)
    log_line(f"  [STEP {n}/{total}]  {name}{sub}")
    log_line(_BAR_THIN)


def log_step_timing(elapsed_s: float, ok: bool) -> None:
    status = "✓" if ok else "✗  FAILED"
    log_line(f"  timing   : {elapsed_s:.3f} s  {status}")
    log_line()


def log_pipeline_summary(
    total_s: float,
    pattern_ner_s: float,
    entities: int,
    language: str,
    ner_model: str,
) -> None:
    log_line()
    log_line(_BAR_WIDE)
    log_line("  PIPELINE COMPLETE")
    log_line(_BAR_WIDE)
    log_line(f"  total       : {total_s:.2f} s")
    log_line(f"  pattern+ner : {pattern_ner_s:.2f} s")
    log_line(f"  entities    : {entities}")
    log_line(f"  language    : {language}")
    log_line(f"  ner model   : {ner_model}")
    log_line(_BAR_WIDE)
    log_line()
