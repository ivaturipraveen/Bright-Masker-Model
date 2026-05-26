from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import structlog

from config import AppConfig
from models.schemas import DetectedSpan, PipelineOutput, ProcessingStats
from pipeline.masking_engine import MaskingEngine
from pipeline.ner_layer import NerLayer
from pipeline.pattern_layer import PatternLayer
from pipeline.preprocessor import Preprocessor
from pipeline.span_merger import SpanMerger
from utils.logger import (
    get_logger,
    log_line,
    log_pipeline_summary,
    log_request_start,
    log_step_header,
    log_step_timing,
)

log = get_logger(__name__)

_ENT_INDENT = "               "


def _entity_count_lines(spans: list[DetectedSpan]) -> list[str]:
    counts: dict[str, int] = {}
    for s in spans:
        counts[s.entity_id] = counts.get(s.entity_id, 0) + 1
    if not counts:
        return [_ENT_INDENT + "—"]
    items = [f"{k}:{v}" for k, v in sorted(counts.items())]
    sep = "  |  "
    lines: list[str] = []
    row: list[str] = []
    row_len = 0
    limit = 65
    for item in items:
        added = len(item) + (len(sep) if row else 0)
        if row_len + added > limit and row:
            lines.append(_ENT_INDENT + sep.join(row))
            row = [item]
            row_len = len(item)
        else:
            row.append(item)
            row_len += added
    if row:
        lines.append(_ENT_INDENT + sep.join(row))
    return lines


def _trim_multiline_spans(
    spans: list[DetectedSpan],
    no_multiline_ids: frozenset[str],
) -> list[DetectedSpan]:
    result: list[DetectedSpan] = []
    for span in spans:
        if span.entity_id in no_multiline_ids and "\n" in span.text:
            nl = span.text.index("\n")
            trimmed = span.text[:nl].rstrip()
            if len(trimmed) < 2:
                continue
            result.append(DetectedSpan(
                text=trimmed,
                start=span.start,
                end=span.start + len(trimmed),
                entity_id=span.entity_id,
                display_name=span.display_name,
                confidence=span.confidence,
                source=span.source,
            ))
        else:
            result.append(span)
    return result


class PiiMaskingPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._initialized = False
        self.preprocessor: Optional[Preprocessor] = None
        self.pattern_layer: Optional[PatternLayer] = None
        self.ner_layer: Optional[NerLayer] = None
        self.span_merger: Optional[SpanMerger] = None
        self.masking_engine: Optional[MaskingEngine] = None
        self._no_multiline_ids: frozenset[str] = frozenset()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        gc = self._config.global_config
        self._no_multiline_ids = frozenset(gc.no_multiline_entity_ids)

        self.preprocessor = Preprocessor(self._config)
        self.pattern_layer = PatternLayer(self._config)
        self.ner_layer = NerLayer(self._config)
        await self.ner_layer.initialize()
        self.span_merger = SpanMerger()
        self.masking_engine = MaskingEngine(self._config)
        self._initialized = True
        log.info("pipeline_initialized", entities=len(self._config.entities))

    async def process(
        self,
        text: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> PipelineOutput:
        await self._ensure_initialized()

        doc_id = uuid.uuid4().hex[:8]
        structlog.contextvars.bind_contextvars(doc_id=doc_id)

        try:
            return await self._process(text, doc_id, progress_queue)
        finally:
            structlog.contextvars.clear_contextvars()

    async def _process(
        self,
        text: str,
        doc_id: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> PipelineOutput:

        async def emit(event: dict) -> None:
            if progress_queue is not None:
                await progress_queue.put(event)

        t0 = log_request_start(doc_id, len(text), text[:60].replace("\n", " "))
        log.info("pipeline_start", chars=len(text))

        # ── [1/4] Preprocessor ────────────────────────────────────────────────
        log_step_header(1, 4, "PREPROCESSOR")
        t1 = time.perf_counter()

        preprocessed = await self.preprocessor.process(text)

        pre_s = time.perf_counter() - t1
        log_line(f"  language : {preprocessed.language}")
        log_line(f"  format   : {preprocessed.format}")
        log_line(f"  chars    : {len(preprocessed.text):,}")
        log_step_timing(pre_s, True)

        await emit({"type": "progress", "step": 1, "name": "preprocessor",
                    "ms": round(pre_s * 1000, 1),
                    "language": preprocessed.language,
                    "format": preprocessed.format,
                    "chars": len(preprocessed.text)})

        # ── [2/4] Pattern + NER (parallel) ───────────────────────────────────
        log_step_header(2, 4, "PATTERN + NER", "parallel")
        t2 = time.perf_counter()

        async def _timed_pattern():
            t = time.perf_counter()
            spans = await self.pattern_layer.analyze(preprocessed.text, preprocessed.language)
            return spans, (time.perf_counter() - t) * 1000

        async def _timed_ner():
            t = time.perf_counter()
            spans = await self.ner_layer.analyze(preprocessed.text)
            return spans, (time.perf_counter() - t) * 1000

        _pat_result, _ner_result = await asyncio.gather(
            _timed_pattern(), _timed_ner(), return_exceptions=True
        )

        if isinstance(_pat_result, BaseException):
            log.warning("pattern_layer_failed", error=str(_pat_result)[:200])
            pattern_spans, pattern_ms = [], 0.0
        else:
            pattern_spans, pattern_ms = _pat_result

        if isinstance(_ner_result, BaseException):
            log.warning("ner_layer_failed", error=str(_ner_result)[:200])
            ner_spans, ner_ms = [], 0.0
        else:
            ner_spans, ner_ms = _ner_result

        local_s = time.perf_counter() - t2

        log_line(f"  pattern  : {len(pattern_spans):>3} spans  ({pattern_ms:.0f} ms)")
        for _l in _entity_count_lines(pattern_spans):
            log_line(_l)
        log_line(f"  ner      : {len(ner_spans):>3} spans  ({ner_ms:.0f} ms)")
        for _l in _entity_count_lines(ner_spans):
            log_line(_l)
        log_step_timing(local_s, True)

        await emit({"type": "progress", "step": 2, "name": "pattern_ner",
                    "ms": round(local_s * 1000, 1),
                    "pattern_spans": len(pattern_spans),
                    "ner_spans": len(ner_spans)})

        # ── [3/4] Merge + Filter ──────────────────────────────────────────────
        log_step_header(3, 4, "MERGE + FILTER")
        t3 = time.perf_counter()

        merged = self.span_merger.merge_all(
            pattern_spans + ner_spans, self._config.entities_by_id, preprocessed.text
        )
        merged = _trim_multiline_spans(merged, self._no_multiline_ids)

        t3_s = time.perf_counter() - t3
        log_line(f"  input    : {len(pattern_spans) + len(ner_spans)} spans")
        log_line(f"  output   : {len(merged)} spans")
        for _l in _entity_count_lines(merged):
            log_line(_l)
        log_step_timing(t3_s, True)

        await emit({"type": "progress", "step": 3, "name": "merge",
                    "ms": round(t3_s * 1000, 1),
                    "spans_final": len(merged)})

        # ── [4/4] Masking ─────────────────────────────────────────────────────
        log_step_header(4, 4, "MASKING")
        t4 = time.perf_counter()

        masked_text, masked_spans = self.masking_engine.mask(preprocessed.text, merged)

        t4_s = time.perf_counter() - t4
        log_line(f"  masked   : {len(masked_spans)} entities")
        log_step_timing(t4_s, True)

        await emit({"type": "progress", "step": 4, "name": "masking",
                    "ms": round(t4_s * 1000, 1),
                    "entities_masked": len(masked_spans)})

        # ── Summary ───────────────────────────────────────────────────────────
        total_s = time.perf_counter() - t0
        ner_model = getattr(self._config, "fine_tuned_model_path", "") or self._config.gliner_model_name

        log_pipeline_summary(
            total_s=total_s,
            pattern_ner_s=local_s,
            entities=len(merged),
            language=preprocessed.language,
            ner_model=ner_model,
        )

        return PipelineOutput(
            original_text=text,
            masked_text=masked_text,
            detected_spans=merged,
            masked_spans=masked_spans,
            stats=ProcessingStats(
                total_ms=total_s * 1000,
                pattern_ms=pattern_ms,
                ner_ms=ner_ms,
                local_ms=local_s * 1000,
                spans_pattern=len(pattern_spans),
                spans_ner=len(ner_spans),
                spans_total=len(merged),
                language=preprocessed.language,
                ner_model=ner_model,
            ),
        )

    async def process_batch(
        self, texts: list[str], max_concurrency: int | None = None
    ) -> list[PipelineOutput]:
        await self._ensure_initialized()
        semaphore = asyncio.Semaphore(max_concurrency or self._config.batch_max_concurrency)

        async def _bounded(text: str) -> PipelineOutput:
            async with semaphore:
                return await self.process(text)

        return await asyncio.gather(*[_bounded(t) for t in texts])

    def process_sync(self, text: str) -> PipelineOutput:
        return asyncio.run(self.process(text))
