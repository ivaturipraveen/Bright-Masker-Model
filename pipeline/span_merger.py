from __future__ import annotations

import re

from config import AppConfig, EntityConfig
from models.schemas import DetectedSpan
from utils.logger import get_logger

log = get_logger(__name__)

_SOURCE_ORDER = {"pattern": 0, "ner": 1, "llm": 2}

_DATE_RE = re.compile(r'^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$')
_TRAILING_LOWERCASE_RE = re.compile(r'(\s+[a-z]\w*)+$')

# Known medical/lab terms that Presidio/spaCy NER incorrectly tags as PERSON
_PERSON_NAME_BLOCKLIST = frozenset({
    "lipid profile", "complete blood count", "comprehensive metabolic panel",
    "cbc", "cmp", "urinalysis", "hba1c", "blood pressure", "heart rate",
    "blood glucose", "oxygen saturation", "metabolic panel",
})


class SpanMerger:
    def merge(
        self,
        spans: list[DetectedSpan],
        threshold: float,
        entity_configs: dict[str, EntityConfig],
        app_config: AppConfig | None = None,
    ) -> tuple[list[DetectedSpan], list[DetectedSpan]]:
        spans = self._validate_and_clean(spans)
        deduped = self._remove_exact_duplicates(spans)
        log.debug("merger_dedup",
                 before=len(spans),
                 after=len(deduped),
                 removed=len(spans) - len(deduped))

        merged = self._resolve_overlaps(deduped, entity_configs)
        log.debug("merger_overlap",
                 before=len(deduped),
                 after=len(merged),
                 removed=len(deduped) - len(merged))

        high_conf, low_conf = self._split_by_threshold(merged, app_config, threshold)
        log.debug("merger_threshold_split",
                 threshold=threshold,
                 high_conf=len(high_conf),
                 low_conf=len(low_conf),
                 low_conf_spans=[(s.entity_id, round(s.confidence, 2)) for s in low_conf])

        return high_conf, low_conf

    def _validate_and_clean(self, spans: list[DetectedSpan]) -> list[DetectedSpan]:
        result: list[DetectedSpan] = []
        for span in spans:
            # Physician name from GLiNER must contain Dr./Doctor prefix — rejects hospital names
            if span.entity_id == "physician_name" and span.source == "ner":
                if not any(p in span.text for p in ("Dr.", "Doctor", "Dr ")):
                    log.debug("span_rejected_physician_no_prefix", text=span.text)
                    continue
            # Known lab/medical terms misclassified as person names by Presidio/spaCy
            if span.entity_id == "person_name" and span.text.lower() in _PERSON_NAME_BLOCKLIST:
                log.debug("span_rejected_lab_term_as_person", text=span.text)
                continue
            # Email must contain @ — rejects person names misclassified as email
            if span.entity_id == "email_address" and "@" not in span.text:
                log.debug("span_rejected_no_at", text=span.text, entity_id=span.entity_id)
                continue
            # License plate must not be a bare date (MM/DD/YYYY misclassified by GLiNER)
            if span.entity_id == "license_plate" and _DATE_RE.match(span.text.strip()):
                log.debug("span_rejected_date_as_plate", text=span.text)
                continue
            # Physician name: strip trailing lowercase words e.g. "supervised", "and"
            if span.entity_id == "physician_name":
                cleaned = _TRAILING_LOWERCASE_RE.sub("", span.text)
                if cleaned != span.text:
                    trim = len(span.text) - len(cleaned)
                    span = span.model_copy(update={"text": cleaned, "end": span.end - trim})
                    log.debug("physician_name_trimmed", original=span.text, cleaned=cleaned)
            result.append(span)
        return result

    def _remove_exact_duplicates(
        self, spans: list[DetectedSpan]
    ) -> list[DetectedSpan]:
        best: dict[tuple[int, int, str], DetectedSpan] = {}
        for span in spans:
            key = (span.start, span.end, span.entity_id)
            if key not in best or span.confidence > best[key].confidence:
                best[key] = span
        return list(best.values())

    def _resolve_overlaps(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        sorted_spans = sorted(spans, key=lambda s: s.start)
        result: list[DetectedSpan] = []

        for span in sorted_spans:
            if not result:
                result.append(span)
                continue

            last = result[-1]
            if span.start < last.end:
                winner = self._pick_winner(last, span, entity_configs)
                loser = span if winner is last else last
                log.debug("merger_overlap_resolved",
                          winner_entity=winner.entity_id,
                          winner_text=winner.text,
                          loser_entity=loser.entity_id,
                          loser_text=loser.text)
                result[-1] = winner
            else:
                result.append(span)

        return result

    # Entity IDs that only ever match numeric/structured codes — never plain text names.
    _NUMERIC_ENTITY_IDS = frozenset({
        "bank_account_number", "bank_routing_number", "credit_card_number",
        "iban", "swift_bic_code", "ssn", "tax_id_number", "passport_number",
        "state_id_number", "ip_address", "device_identifier",
        "card_cryptogram", "card_iin_bin", "card_pin", "billing_number",
        "terminal_id", "transaction_id", "merchant_id",
    })

    def _pick_winner(
        self,
        a: DetectedSpan,
        b: DetectedSpan,
        entity_configs: dict[str, EntityConfig],
    ) -> DetectedSpan:
        # If one span is all letters/spaces (a name) but tagged as a numeric-only entity,
        # that is a false positive — prefer the other span.
        def is_alpha_only(text: str) -> bool:
            return all(c.isalpha() or c in " .-'" for c in text)

        if a.entity_id in self._NUMERIC_ENTITY_IDS and is_alpha_only(a.text):
            return b
        if b.entity_id in self._NUMERIC_ENTITY_IDS and is_alpha_only(b.text):
            return a

        if a.confidence != b.confidence:
            return a if a.confidence > b.confidence else b

        priority_a = entity_configs.get(a.entity_id, None)
        priority_b = entity_configs.get(b.entity_id, None)
        pa = priority_a.priority if priority_a else 5
        pb = priority_b.priority if priority_b else 5
        if pa != pb:
            return a if pa < pb else b

        sa = _SOURCE_ORDER.get(a.source, 99)
        sb = _SOURCE_ORDER.get(b.source, 99)
        if sa != sb:
            return a if sa <= sb else b

        # Final tiebreaker: prefer the longer span so containing spans (e.g., a full URL)
        # are not discarded in favour of a short substring (e.g., an extracted token).
        return a if (a.end - a.start) >= (b.end - b.start) else b

    def merge_all(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        """Merge all spans from all sources without threshold split."""
        spans = self._validate_and_clean(spans)
        deduped = self._remove_exact_duplicates(spans)
        log.debug("merger_all_dedup",
                 before=len(spans),
                 after=len(deduped),
                 removed=len(spans) - len(deduped))

        merged = self._resolve_overlaps(deduped, entity_configs)
        log.debug("merger_all_overlap",
                 before=len(deduped),
                 after=len(merged),
                 removed=len(deduped) - len(merged),
                 entities=sorted({s.entity_id for s in merged}))

        return merged

    def resolve_overlaps(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        """Public entry point for post-merge overlap resolution (e.g. final dedup)."""
        deduped = self._remove_exact_duplicates(spans)
        return self._resolve_overlaps(deduped, entity_configs)

    def _split_by_threshold(
        self,
        spans: list[DetectedSpan],
        app_config: AppConfig | None,
        global_threshold: float,
    ) -> tuple[list[DetectedSpan], list[DetectedSpan]]:
        high: list[DetectedSpan] = []
        low: list[DetectedSpan] = []

        for span in spans:
            threshold = global_threshold
            if app_config is not None:
                threshold = app_config.get_confidence_threshold(span.entity_id)

            if span.confidence >= threshold:
                high.append(span)
            else:
                low.append(span)

        return high, low
