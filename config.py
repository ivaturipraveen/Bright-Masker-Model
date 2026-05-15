from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from exceptions import ConfigValidationError
from models.schemas import MaskingStrategy

load_dotenv()


class MaskingConfig(BaseModel):
    strategy: MaskingStrategy
    format: str


class EntityConfig(BaseModel):
    id: str
    display_name: str
    description: str = ""
    enabled: bool = True
    policy: list[str] = ["general"]
    priority: int = 5
    confidence_threshold: Optional[float] = None
    presidio_type: Optional[str] = None
    gliner_label: Optional[str] = None
    patterns: list[str] = []
    masking: MaskingConfig
    notes: str = ""


class GlobalConfig(BaseModel):
    default_confidence_threshold: float = 0.85
    default_masking_strategy: str = "redact"
    language: str = "auto"
    enabled_policies: list[str] = []
    label_blocklist: list[str] = []
    no_multiline_entity_ids: list[str] = []


class Config:
    def __init__(self, **overrides: Any) -> None:
        def _s(key: str, env: str, default: str) -> str:
            return str(overrides[key]) if key in overrides else os.getenv(env, default)

        def _i(key: str, env: str, default: int) -> int:
            return int(overrides[key]) if key in overrides else int(os.getenv(env, str(default)))

        def _f(key: str, env: str, default: float) -> float:
            return float(overrides[key]) if key in overrides else float(os.getenv(env, str(default)))

        def _b(key: str, env: str, default: bool) -> bool:
            if key in overrides:
                return bool(overrides[key])
            return os.getenv(env, str(default)).lower() in ("true", "1", "yes")

        # ── GLiNER model ────────────────────────────────────────────────────
        # Set FINE_TUNED_MODEL_PATH after running train/finetune.py to use
        # the trained model. Empty string = base GLiNER model.
        self.gliner_model_name = _s("gliner_model_name", "GLINER_MODEL_NAME", "urchade/gliner_large-v2.1")
        self.fine_tuned_model_path = _s("fine_tuned_model_path", "FINE_TUNED_MODEL_PATH", "")

        # NER threshold — 0.55 is a good default without an LLM to filter FPs.
        # Lower = higher recall, more false positives.
        # Higher = fewer false positives, may miss borderline entities.
        self.gliner_threshold = _f("gliner_threshold", "GLINER_THRESHOLD", 0.55)
        self.gliner_max_chunk_chars = _i("gliner_max_chunk_chars", "GLINER_MAX_CHUNK_CHARS", 1200)
        self.gliner_chunk_overlap_chars = _i("gliner_chunk_overlap_chars", "GLINER_CHUNK_OVERLAP_CHARS", 100)

        # ── spaCy / Presidio (pattern layer) ────────────────────────────────
        self.spacy_model_name = _s("spacy_model_name", "SPACY_MODEL_NAME", "en_core_web_sm")
        self.presidio_min_score = _f("presidio_min_score", "PRESIDIO_MIN_SCORE", 0.6)
        self.presidio_nlp_engine = _s("presidio_nlp_engine", "PRESIDIO_NLP_ENGINE", "spacy")
        self.presidio_language = _s("presidio_language", "PRESIDIO_LANGUAGE", "en")

        # ── Entity config ───────────────────────────────────────────────────
        self.entities_config_path = Path(_s("entities_config_path", "ENTITIES_CONFIG_PATH", "entities_config.yaml"))

        # ── Masking engine ──────────────────────────────────────────────────
        self.faker_seed = _i("faker_seed", "FAKER_SEED", 42)
        self.encryption_key = _s("encryption_key", "ENCRYPTION_KEY", "change-this-key-in-production")

        # ── Pipeline ────────────────────────────────────────────────────────
        self.enable_async_layers = _b("enable_async_layers", "ENABLE_ASYNC_LAYERS", True)
        self.batch_max_concurrency = _i("batch_max_concurrency", "BATCH_MAX_CONCURRENCY", 4)

        # ── Logging ─────────────────────────────────────────────────────────
        self.log_level = _s("log_level", "LOG_LEVEL", "INFO")


class AppConfig:
    def __init__(
        self,
        settings: Optional[Config] = None,
        entities_config_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or Config()
        if entities_config_path is not None:
            self._settings.entities_config_path = Path(entities_config_path)
        self._entities: list[EntityConfig] = []
        self._global_config: GlobalConfig = GlobalConfig()
        self.load()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._settings, name)

    def load(self) -> "AppConfig":
        self._entities = self.load_entities()
        return self

    def load_entities(self) -> list[EntityConfig]:
        path = self._settings.entities_config_path
        if not path.exists():
            raise ConfigValidationError(f"entities_config.yaml not found at {path}")

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        self._global_config = GlobalConfig(**data.get("global", {}))

        active_policies: set[str] = set(self._global_config.enabled_policies)

        entities: list[EntityConfig] = []
        for raw in data.get("entities", []):
            entity_id = raw.get("id", "unknown")
            try:
                entity = EntityConfig(**raw)
            except Exception as exc:
                raise ConfigValidationError(
                    f"Invalid entity config for '{entity_id}': {exc}"
                ) from exc
            if not entity.enabled:
                continue
            if active_policies and not set(entity.policy).intersection(active_policies):
                continue
            entities.append(entity)
        return entities

    @property
    def global_config(self) -> GlobalConfig:
        return self._global_config

    @property
    def entities(self) -> list[EntityConfig]:
        return self._entities

    @property
    def entities_by_id(self) -> dict[str, EntityConfig]:
        return {e.id: e for e in self._entities}

    @property
    def presidio_entity_map(self) -> dict[str, str]:
        return {e.presidio_type: e.id for e in self._entities if e.presidio_type}

    @property
    def gliner_label_to_entity_id(self) -> dict[str, str]:
        return {(e.gliner_label or e.display_name): e.id for e in self._entities}

    def get_confidence_threshold(self, entity_id: str) -> float:
        entity = self.entities_by_id.get(entity_id)
        if entity and entity.confidence_threshold is not None:
            return entity.confidence_threshold
        return self._global_config.default_confidence_threshold
