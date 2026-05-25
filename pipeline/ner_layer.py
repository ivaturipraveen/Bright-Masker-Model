from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from config import AppConfig
from exceptions import LayerInitError
from models.schemas import DetectedSpan
from utils.logger import get_logger
from utils.text_utils import chunk_text

log = get_logger(__name__)

_MAX_PARALLEL_NER_CHUNKS = 4

# Semantic label groups used on CPU/MPS — parallel thread pool, one group per core.
# Kept ≤25 labels per group to prevent GLiNER accuracy degradation on CPU.
# On CUDA these are ignored; all labels go in a single flat call instead.
_SEMANTIC_GROUPS: list[list[str]] = [
    # Group 1 — People + contact (21)
    [
        "person_name", "physician_name", "date_of_birth", "physical_characteristics",
        "signature", "username", "race", "racial_ethnic_origin", "religion",
        "biometric_facial_recognition", "biometric_iris_scan",
        "biometric_voiceprint", "biometric_dna", "fingerprint",
        "phone_number", "fax_number", "email_address", "url_with_pii",
        "ip_address", "device_identifier", "cookie_session_token",
    ],
    # Group 2 — Address (5 — small so city_name gets full attention)
    [
        "street_address", "city_name", "us_state", "zipcode", "precise_geolocation",
    ],
    # Group 3 — Government IDs + medical + employment (24)
    [
        "ssn", "passport_number", "drivers_license", "state_id_number",
        "tax_id_number", "medical_license_number", "dea_number", "npi_number",
        "license_plate", "vehicle_vin", "vehicle_registration", "application_id",
        "medical_record_number", "health_plan_beneficiary_number",
        "insurance_policy_number", "medication_name", "clinical_date",
        "insurance_company_name", "hospital_name",
        "employee_id", "organization_name", "employment_history",
        "performance_evaluation", "bart_employee_id",
    ],
    # Group 4 — Financial (19)
    [
        "bank_account_number", "bank_routing_number", "credit_card_number",
        "card_expiration_date", "card_last4", "card_type", "card_track_data",
        "card_pin", "card_cryptogram", "card_iin_bin", "card_holder_name",
        "card_service_code", "iban", "swift_bic_code", "financial_amount",
        "transaction_id", "merchant_id", "terminal_id", "billing_number",
    ],
    # Group 5 — Legal + criminal records (21)
    [
        "case_number", "claim_number", "court_name", "law_firm_name",
        "incident_report", "driver_history",
        "arrest_record", "chri", "fbi_number", "incarceration_info",
        "parole_record", "probation_record", "supervised_release",
        "warrant_data", "wanted_person_report", "foreign_fugitives",
        "gang_terrorist_member", "sex_offender_report",
        "missing_person_report", "protection_orders", "identity_theft_victims",
    ],
    # Group 6 — Education + travel + misc (15)
    [
        "student_id", "university_name", "student_records_ferpa",
        "flight_number", "booking_reference", "hotel_name", "bank_name",
        "stolen_articles", "stolen_boats", "stolen_guns", "stolen_license_plate",
        "stolen_securities", "stolen_vehicle", "password", "confidential",
    ],
]


def _best_device() -> str:
    forced = os.getenv("GLINER_DEVICE", "").lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
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
        self._device: str = "cpu"
        self._label_groups: list[list[str]] = []  # CPU/MPS: parallel groups
        self._flat_labels: list[str] = []          # CUDA: single flat call
        self._pool: ThreadPoolExecutor | None = None

    def _build_labels(self) -> None:
        label_map = self._config.gliner_label_to_entity_id
        id_to_label: dict[str, str] = {v: k for k, v in label_map.items()}

        # Build semantic groups (CPU/MPS path)
        groups: list[list[str]] = []
        assigned_ids: set[str] = set()
        for template_group in _SEMANTIC_GROUPS:
            group_labels = [id_to_label[eid] for eid in template_group if eid in id_to_label]
            if group_labels:
                groups.append(group_labels)
                assigned_ids.update(eid for eid in template_group if eid in id_to_label)

        # Remaining entities not in any semantic group → overflow batches of 25
        remaining = [id_to_label[eid] for eid in id_to_label if eid not in assigned_ids]
        for i in range(0, len(remaining), 25):
            groups.append(remaining[i : i + 25])

        self._label_groups = groups
        self._flat_labels = [label for group in groups for label in group]

        log.info(
            "ner_labels_built",
            total=len(self._flat_labels),
            groups=len(groups),
            group_sizes=[len(g) for g in groups],
            device=self._device,
            strategy="flat_single_call" if self._device == "cuda" else "parallel_groups",
        )

    def _load_model(self) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise LayerInitError("gliner not installed — run: pip install gliner") from exc

        self._device = _best_device()

        fine_tuned = self._config.fine_tuned_model_path
        if fine_tuned:
            model_path = Path(fine_tuned)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / fine_tuned
            if model_path.exists():
                self._model_name = str(model_path)
                log.info("gliner_loading_finetuned", path=self._model_name, device=self._device)
            else:
                log.warning("fine_tuned_not_found", path=str(model_path),
                            fallback=self._config.gliner_model_name)
                self._model_name = self._config.gliner_model_name
        else:
            self._model_name = self._config.gliner_model_name

        log.info("gliner_loading", model=self._model_name, device=self._device)
        self._model = GLiNER.from_pretrained(self._model_name)
        try:
            self._model.to(self._device)
            self._model.eval()  # disable dropout — required for deterministic inference
            if self._device == "cuda":
                import torch
                # fp16: halves VRAM (~850 MB) and speeds up tensor ops on RTX/A100
                self._model.half()
                # TF32: uses tensor cores for matrix multiplies — free speedup on Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # cuDNN benchmark: finds fastest conv algorithm for this input shape
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                log.info("gliner_cuda_optimized",
                         fp16=True, tf32=True, cudnn_benchmark=True)
        except Exception:
            pass
        log.info("gliner_loaded", model=self._model_name, device=self._device)

        self._build_labels()

        # CPU + MPS: thread pool for parallel group inference.
        # CUDA: no pool needed — single flat call is faster than concurrent CUDA threads.
        if self._device in ("cpu", "mps"):
            n_workers = min(len(self._label_groups), (os.cpu_count() or 4))
            self._pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ner_grp")
            log.info("ner_pool_created", device=self._device, workers=n_workers)

    def _warmup(self) -> None:
        dummy = "John Smith, 078-05-1120, john@example.com, 123 Main St."
        if self._device == "cuda":
            # Single flat warmup — primes CUDA kernels for the full label set (fp16)
            self._run_flat_cuda(dummy)
        elif self._pool is not None:
            # Parallel warmup for CPU/MPS — one future per group
            futures = [
                self._pool.submit(self._model.predict_entities, dummy, group,
                                  self._config.gliner_threshold)
                for group in self._label_groups
            ]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass
        log.info("ner_warmup_done", strategy="flat" if self._device == "cuda" else "grouped")

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load_model)
        await asyncio.to_thread(self._warmup)

    def _run_group(self, chunk: str, labels: list[str]) -> list[dict]:
        try:
            return self._model.predict_entities(
                chunk, labels, threshold=self._config.gliner_threshold
            )
        except Exception as exc:
            log.warning("ner_group_failed", n_labels=len(labels), error=str(exc))
            return []

    def _run_flat_cuda(self, chunk: str) -> list[dict]:
        """CUDA path: single forward pass covering all 105 labels with fp16 + no_grad."""
        try:
            import torch
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            with torch.no_grad(), ctx:
                return self._model.predict_entities(
                    chunk, self._flat_labels, threshold=self._config.gliner_threshold
                )
        except Exception as exc:
            log.warning("ner_cuda_flat_failed", error=str(exc))
            return []

    def _predict_chunk(self, chunk: str, offset: int) -> list[DetectedSpan]:
        if not self._model:
            return []

        label_map = self._config.gliner_label_to_entity_id

        if self._device == "cuda":
            # Single call — GPU scores all 105 labels in one forward pass (fp16).
            # Text encoded once; all label embeddings computed simultaneously.
            raw_results = [self._run_flat_cuda(chunk)]
        elif self._pool is not None:
            # CPU/MPS: parallel groups via thread pool
            futures = [self._pool.submit(self._run_group, chunk, group)
                       for group in self._label_groups]
            raw_results = [f.result() for f in futures]
        else:
            # Fallback: sequential (should not normally be hit)
            raw_results = [self._run_group(chunk, group) for group in self._label_groups]

        # Merge — keep highest-confidence span per (start, end, label)
        best: dict[tuple[int, int, str], dict] = {}
        for group_result in raw_results:
            for ent in group_result:
                key = (ent["start"], ent["end"], ent["label"])
                if key not in best or ent["score"] > best[key]["score"]:
                    best[key] = ent

        spans: list[DetectedSpan] = []
        for ent in best.values():
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
                # NER detections have no keyword anchor — match_length
                # equals the value width. Keyword-anchored pattern
                # matches will outrank these via the merger tiebreaker.
                match_length=ent["end"] - ent["start"],
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

        async def _run_chunk(idx: int, chunk_str: str, off: int) -> list[DetectedSpan]:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self._predict_chunk, chunk_str, off)
                except Exception as exc:
                    log.warning("ner_chunk_failed", chunk=idx + 1, error=str(exc))
                    return []

        chunk_results = await asyncio.gather(*[
            _run_chunk(i, ct, off) for i, (ct, off) in enumerate(chunks)
        ])
        all_spans = [s for result in chunk_results for s in result]

        # Deduplicate — keep highest confidence per (start, end, entity_id)
        seen: dict[tuple[int, int, str], DetectedSpan] = {}
        for span in all_spans:
            key = (span.start, span.end, span.entity_id)
            if key not in seen or seen[key].confidence < span.confidence:
                seen[key] = span

        return list(seen.values())
