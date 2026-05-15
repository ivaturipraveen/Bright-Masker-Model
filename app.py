from __future__ import annotations

import asyncio
import json
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*byte fallback.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig, Config
from pipeline.orchestrator import PiiMaskingPipeline
from utils.logger import configure_logging, get_logger

log = get_logger(__name__)

_app_config: AppConfig | None = None
_pipeline: PiiMaskingPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_config, _pipeline
    settings = Config()
    configure_logging(settings.log_level)
    _app_config = AppConfig(settings=settings)
    _pipeline = PiiMaskingPipeline(_app_config)
    await _pipeline._ensure_initialized()

    ner_model = settings.fine_tuned_model_path or settings.gliner_model_name
    log.info("server_ready", ner_model=ner_model, spacy=settings.spacy_model_name,
             entities=len(_app_config.entities))

    # Warmup — loads spaCy/Presidio cold-start and primes GLiNER
    try:
        await _pipeline.process("Warm-up: John Smith, john@example.com, (555) 010-0100.")
        log.info("warmup_complete")
    except Exception as _e:
        log.warning("warmup_failed", error=str(_e)[:200])

    yield
    log.info("server_shutdown")


app = FastAPI(title="PII Masker API", version="3.0.0", lifespan=lifespan)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MaskRequest(BaseModel):
    text: str


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


@app.get("/entities")
async def list_entities():
    if _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    return {
        "entities": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "strategy": e.masking.strategy.value,
                "format": e.masking.format,
                "priority": e.priority,
            }
            for e in _app_config.entities
        ]
    }


_MAX_TEXT_CHARS = 500_000


@app.post("/mask", response_model=MaskResponse)
async def mask_text(request: MaskRequest):
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")
    if len(request.text) > _MAX_TEXT_CHARS:
        raise HTTPException(413, f"text exceeds {_MAX_TEXT_CHARS:,} characters")

    t0 = time.perf_counter()
    result = await _pipeline.process(request.text)
    return _build_mask_response(result, _app_config, (time.perf_counter() - t0) * 1000)


@app.post("/mask/stream")
async def mask_text_stream(request: MaskRequest):
    """SSE — streams per-step progress then final result."""
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")
    if len(request.text) > _MAX_TEXT_CHARS:
        raise HTTPException(413, f"text exceeds {_MAX_TEXT_CHARS:,} characters")

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
