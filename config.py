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
    counter_group: Optional[str] = None


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

        # ── Model ───────────────────────────────────────────────────────────
        self.gliner_model_name       = "urchade/gliner_large-v2.1"
        self.fine_tuned_model_path   = _s("fine_tuned_model_path", "FINE_TUNED_MODEL_PATH", "")
        self.gliner_threshold        = _f("gliner_threshold", "GLINER_THRESHOLD", 0.35)
        self.gliner_max_chunk_chars  = 1200
        self.gliner_chunk_overlap_chars = 100

        # ── Pattern layer ───────────────────────────────────────────────────
        self.spacy_model_name   = "en_core_web_lg"
        self.presidio_min_score = 0.6
        self.presidio_nlp_engine = "spacy"
        self.presidio_language  = "en"

        # ── Entity config ───────────────────────────────────────────────────
        self.entities_config_path = Path(
            _s("entities_config_path", "ENTITIES_CONFIG_PATH", "./entities_config.yaml")
        )

        # ── Masking ─────────────────────────────────────────────────────────
        self.faker_seed      = 42
        self.encryption_key  = _s("encryption_key", "ENCRYPTION_KEY",
                                  "change-this-to-a-random-secret-key-in-production")

        # ── Server ──────────────────────────────────────────────────────────
        self.log_level            = _s("log_level", "LOG_LEVEL", "INFO")
        self.log_format           = "json"
        self.max_text_chars       = 500_000
        self.batch_max_concurrency = 4
        self.warmup_text          = "Warm-up: John Smith, john@example.com, (555) 010-0100."

        # ── Bright Shield remote (optional — leave BRIGHT_SHIELD_BASE_URL empty for local-only) ──
        self.bright_shield_base_url         = _s("bright_shield_base_url", "BRIGHT_SHIELD_BASE_URL", "").rstrip("/")
        self.bright_shield_proxy_path       = "/proxy/bright-shield"
        self.bright_shield_proxy_enabled    = True
        self.bright_shield_health_timeout_sec = 10.0
        self.bright_shield_mask_timeout_sec   = 60.0
        self.ui_health_timeout_ms             = 8000

        # ── UI labels (hardcoded — not env-configurable) ────────────────────
        self.app_title          = "Bright Masker"
        self.page_title         = "Bright Masker — PII Detection"
        self.comparison_subtitle = (
            "Side-by-side: GLiNER Fine-tuned (local) vs. Bright Shield API — same input, both in parallel."
        )
        self.local_model_name         = "GLiNER Fine-tuned"
        self.local_model_badge        = "GLiNER Fine-tuned · local"
        self.local_model_desc         = "105 entity types · fine-tuned · no LLM"
        self.remote_model_name        = "Bright Shield"
        self.remote_model_badge       = "Bright Shield · API"
        self.remote_model_desc        = "Bright Shield PII Detection API"
        self.remote_model_offline_label = "Bright Shield · offline"


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
