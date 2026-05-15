from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class MaskingStrategy(str, Enum):
    REDACT = "redact"
    SUBSTITUTE = "substitute"
    HASH = "hash"
    ENCRYPT = "encrypt"
    PARTIAL_REDACT = "partial_redact"


class DetectedSpan(BaseModel):
    text: str
    start: int
    end: int
    entity_id: str
    display_name: str
    confidence: float
    source: Literal["pattern", "ner"]
    context: str = ""


class MaskedSpan(BaseModel):
    original: str
    masked: str
    entity_id: str
    strategy: MaskingStrategy
    start: int
    end: int


class ProcessingStats(BaseModel):
    total_ms: float
    pattern_ms: float
    ner_ms: float
    local_ms: float = 0.0   # wall-clock of parallel pattern+NER phase
    spans_pattern: int
    spans_ner: int
    spans_total: int
    language: str
    ner_model: str = ""


class PipelineOutput(BaseModel):
    original_text: str
    masked_text: str
    detected_spans: list[DetectedSpan]
    masked_spans: list[MaskedSpan]
    stats: ProcessingStats


class PreprocessedText(BaseModel):
    text: str
    language: str
    format: Literal["plain", "json", "csv", "xml"]
    original_length: int
