from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from config import AppConfig
from exceptions import LayerInitError
from models.schemas import DetectedSpan
from utils.logger import get_logger
from utils.text_utils import chunk_text

log = get_logger(__name__)

_MAX_PARALLEL_NER_CHUNKS = 4


def _best_device() -> str:
    forced = os.getenv("GLINER_DEVICE", "").lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        pass
    return "cpu"


class NerLayer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Any = None
        self._model_name: str = ""

    def _load_model(self) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise LayerInitError(
                "gliner package not installed — run: pip install gliner"
            ) from exc

        device = _best_device()

        # Prefer fine-tuned model if path is configured and exists
        fine_tuned = self._config.fine_tuned_model_path
        if fine_tuned:
            model_path = Path(fine_tuned)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / fine_tuned
            if model_path.exists():
                self._model_name = str(model_path)
                log.info("gliner_loading_finetuned", path=self._model_name, device=device)
            else:
                log.warning("fine_tuned_model_not_found",
                            path=str(model_path),
                            fallback=self._config.gliner_model_name)
                self._model_name = self._config.gliner_model_name
        else:
            self._model_name = self._config.gliner_model_name

        log.info("gliner_loading", model=self._model_name, device=device)
        self._model = GLiNER.from_pretrained(self._model_name)
        try:
            self._model.to(device)
        except Exception:
            pass
        log.info("gliner_loaded", model=self._model_name, device=device)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load_model)

    def _predict_chunk(self, chunk: str, offset: int) -> list[DetectedSpan]:
        label_map = self._config.gliner_label_to_entity_id
        labels = list(label_map.keys())
        if not labels or self._model is None:
            return []

        try:
            entities = self._model.predict_entities(
                chunk, labels, threshold=self._config.gliner_threshold
            )
        except Exception as exc:
            log.warning("gliner_predict_failed", error=str(exc))
            return []

        spans: list[DetectedSpan] = []
        for ent in entities:
            entity_id = label_map.get(ent["label"])
            if not entity_id:
                continue
            entity_cfg = self._config.entities_by_id.get(entity_id)
            if not entity_cfg:
                continue
            spans.append(DetectedSpan(
                text=ent["text"],
                start=offset + ent["start"],
                end=offset + ent["end"],
                entity_id=entity_id,
                display_name=entity_cfg.display_name,
                confidence=float(ent["score"]),
                source="ner",
            ))

        return spans

    async def analyze(self, text: str) -> list[DetectedSpan]:
        if self._model is None:
            return []

        chunks = chunk_text(
            text,
            max_chars=self._config.gliner_max_chunk_chars,
            overlap_chars=self._config.gliner_chunk_overlap_chars,
        )

        semaphore = asyncio.Semaphore(_MAX_PARALLEL_NER_CHUNKS)

        async def _run_chunk(idx: int, chunk_text_str: str, offset: int) -> list[DetectedSpan]:
            async with semaphore:
                try:
                    return await asyncio.to_thread(
                        self._predict_chunk, chunk_text_str, offset
                    )
                except Exception as exc:
                    log.warning("ner_chunk_failed", chunk=idx + 1, error=str(exc))
                    return []

        chunk_results = await asyncio.gather(*[
            _run_chunk(i, ct, off) for i, (ct, off) in enumerate(chunks)
        ])
        all_spans: list[DetectedSpan] = [s for result in chunk_results for s in result]

        # Deduplicate: same (start, end, entity_id) → keep highest confidence
        seen: dict[tuple[int, int, str], DetectedSpan] = {}
        for span in all_spans:
            key = (span.start, span.end, span.entity_id)
            if key not in seen or seen[key].confidence < span.confidence:
                seen[key] = span

        return list(seen.values())
