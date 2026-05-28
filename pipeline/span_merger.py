from __future__ import annotations

import re

from config import AppConfig, EntityConfig
from models.schemas import DetectedSpan
from utils.logger import get_logger

log = get_logger(__name__)

_SOURCE_ORDER = {"pattern": 0, "ner": 1, "llm": 2}

_DATE_RE = re.compile(r'^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$')
_TRAILING_LOWERCASE_RE = re.compile(r'(\s+[a-z]\w*)+$')

# Record/ID entity types that should never be purely alphabetic text.
# GLiNER tags person names as these types when surrounding words provide context
# (e.g. "suspects" → wanted_person_report, "patients" → medical_record_number).
_ALPHA_REJECT_ENTITY_IDS = frozenset({
    "wanted_person_report", "missing_person_report", "gang_terrorist_member",
    "foreign_fugitives", "identity_theft_victims", "sex_offender_report",
    "supervised_release", "probation_record", "parole_record",
    "medical_record_number", "billing_number", "health_plan_beneficiary_number",
    "insurance_policy_number",
})

# Known medical/lab terms that Presidio/spaCy NER incorrectly tags as PERSON
_PERSON_NAME_BLOCKLIST = frozenset({
    "lipid profile", "complete blood count", "comprehensive metabolic panel",
    "cbc", "cmp", "urinalysis", "hba1c", "blood pressure", "heart rate",
    "blood glucose", "oxygen saturation", "metabolic panel",
})

# Common English words that GLiNER over-detects as entities when they appear
# in field-label positions ("Bank Identifier Code", "State ID Number", etc.).
# Any NER span whose lowercased trimmed value is in this set is rejected
# regardless of which entity type was assigned — these are NEVER PII by
# themselves. Sentence-level adversarial-negative training cuts most of
# these out, but this acts as a defence-in-depth safety net.
_NER_COMMON_WORD_BLOCKLIST = frozenset({
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Common label heads that got mistagged in production
    "bank", "treasury", "transaction", "transfer", "domestic",
    "international", "charter", "web", "secure", "corporate",
    "session", "card", "order", "policy", "policies", "number",
    "identifier", "identification", "statement", "reference",
    "service", "services", "investment", "refund", "purchase",
    "merchant", "payment", "gateway", "issuer", "issued",
    "delivery", "billing", "mailing", "shipping",
    # Region / direction words
    "state", "city", "country", "region", "regional", "global",
    "national", "federal", "central", "north", "south", "east",
    "west", "northern", "southern", "eastern", "western",
    # Card / payment scheme words (NOT the brand names — see card_type entity)
    "credit", "debit", "prepaid", "loyalty",
    # Verbs commonly mistagged
    "send", "sent", "receive", "received", "process", "processed",
    "verify", "verified", "validate", "validated", "approve",
    "approved", "issue", "issues", "deliver", "delivered",
    "confirm", "confirmed", "complete", "completed",
    # Connectors
    "and", "or", "of", "for", "to", "from", "with", "without",
    # Other common label heads
    "field", "form", "record", "records", "report", "document",
    "file", "entry", "data", "info", "information", "details",
    "name", "type", "category", "code", "key", "token",
})


def _is_blocklisted_ner_span(span: DetectedSpan) -> bool:
    """True if the span was tagged by NER and is a common English word.

    Catches the failure mode where 'Bank' → [COMPANY], 'state' → [STATE],
    'Transaction' → [LOCATION], etc.
    """
    if span.source != "ner":
        return False
    text = span.text.strip().lower()
    return text in _NER_COMMON_WORD_BLOCKLIST

# NER shape gates — entities whose value MUST match a structured prefix.
# When GLiNER mislabels a generic alphanumeric value as one of these types
# (e.g. "F12345" tagged merchant_id), the gate filters it out so a more
# appropriate detection (or no detection) wins.
_NER_PREFIX_REQUIRED: dict[str, re.Pattern] = {
    "merchant_id": re.compile(r"\bMID[\s\-_]?", re.IGNORECASE),
    "terminal_id": re.compile(r"\bTID[\s\-_]?", re.IGNORECASE),
}

# Person-name spans tagged by NER must NOT look like a structured ID
# (e.g. "FRT-99281", "ML-CA-1234567"). When NER drifts onto a code,
# rejecting it here lets the pattern layer or another label take over.
_PERSON_NAME_STRUCTURED_ID_RE = re.compile(
    r"^[A-Z]{1,6}[\s\-_/]?\d{2,}(?:[\s\-_/][A-Z0-9]+)*$"
)

# Matches a label word ("ID", "number", "No.") right after a span, capturing the
# token that follows it. Used to detect GLiNER tagging the word before "ID" as a
# place ("Transaction ID TXN-…", "employee id EMP-…").
_LABEL_AFTER_ID_RE = re.compile(r"\s+(?:id|number|no\.?)\b[:#]?\s*(\S+)?", re.IGNORECASE)


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

    def _validate_and_clean(
        self, spans: list[DetectedSpan], text: str = ""
    ) -> list[DetectedSpan]:
        result: list[DetectedSpan] = []
        for span in spans:
            # Common-English-word kill switch — any NER span whose value is a
            # generic English word ("Bank", "state", "Transaction", etc.) is
            # rejected regardless of entity_id. Defence-in-depth alongside
            # the adversarial-negative training rows.
            if _is_blocklisted_ner_span(span):
                log.debug("span_rejected_common_english_word",
                          entity_id=span.entity_id, text=span.text)
                continue
            # ── Context gates (need the surrounding document text) ───────────
            if text and span.source == "ner":
                s, e = span.start, span.end
                # (A) Repair NER spans that begin mid-word — a GLiNER sub-token
                #     artifact: "Identifier"->"entifier", "reference"->"erence",
                #     "Statement"->"ment". Advance start past the partial leading
                #     word: this keeps a real value ("erence BK-12" -> "BK-12")
                #     and drops pure fragments ("entifier" -> "" -> rejected).
                if 0 < s < len(text) and text[s - 1].isalnum() and text[s].isalnum():
                    i = s
                    while i < e and text[i].isalnum():
                        i += 1
                    while i < e and not text[i].isalnum():
                        i += 1
                    if len(text[i:e].strip()) < 2:
                        log.debug("span_rejected_midword_fragment", text=span.text)
                        continue
                    span = span.model_copy(update={"text": text[i:e], "start": i})
                # (B) Suppress city/state spans that are really the label word
                #     before "ID"/"number" ("Transaction ID", "employee id",
                #     "Tax ID number"). Guard: only when the token after the
                #     label is an identifier (dash, letter+digit mix, or the word
                #     "number") so a real "Boise ID 83701" (city + state + zip)
                #     is preserved.
                if span.entity_id in ("city_name", "us_state"):
                    m = _LABEL_AFTER_ID_RE.match(text[e:])
                    if m:
                        nxt = m.group(1) or ""
                        mixed = any(c.isalpha() for c in nxt) and any(c.isdigit() for c in nxt)
                        if "-" in nxt or mixed or nxt.lower() in ("number", "no", "no."):
                            log.debug("span_rejected_label_before_id",
                                      entity_id=span.entity_id, text=span.text)
                            continue
            # Physician name from GLiNER must contain Dr./Doctor prefix — rejects hospital names
            if span.entity_id == "physician_name" and span.source == "ner":
                if not any(p in span.text for p in ("Dr.", "Doctor", "Dr ")):
                    log.debug("span_rejected_physician_no_prefix", text=span.text)
                    continue
            # Record/ID entities must not be pure alphabetic text (person names).
            # GLiNER tags names in context sentences as record types.
            if span.source == "ner" and span.entity_id in _ALPHA_REJECT_ENTITY_IDS:
                if all(c.isalpha() or c in " .'-" for c in span.text):
                    log.debug("span_rejected_alpha_only_record_id",
                              entity_id=span.entity_id, text=span.text)
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
            # NER shape gates — merchant_id/terminal_id from GLiNER must contain
            # the canonical prefix (MID-, TID-). Free-form values are rejected
            # so they don't eat structured codes that belong to other entities.
            if span.source == "ner":
                gate = _NER_PREFIX_REQUIRED.get(span.entity_id)
                if gate is not None and not gate.search(span.text):
                    log.debug("span_rejected_missing_required_prefix",
                              entity_id=span.entity_id, text=span.text)
                    continue
            # Person-name NER spans must not look like a structured ID
            # (e.g. NER drifting onto "FRT-99281" or "ML-CA-1234567").
            if (span.entity_id == "person_name" and span.source == "ner"
                    and _PERSON_NAME_STRUCTURED_ID_RE.match(span.text.strip())):
                log.debug("span_rejected_structured_id_as_person", text=span.text)
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

        # Pattern layer always beats NER/LLM — patterns are precision-crafted
        # for structured codes and more trustworthy than the neural model.
        sa = _SOURCE_ORDER.get(a.source, 99)
        sb = _SOURCE_ORDER.get(b.source, 99)
        if sa != sb:
            return a if sa < sb else b

        if a.confidence != b.confidence:
            return a if a.confidence > b.confidence else b

        # Specificity tiebreaker (Group C): when two spans share source and
        # confidence, prefer the one whose regex match was *wider* than the
        # captured value. A keyword-anchored pattern like
        # "Insurance Policy Number: A12345678" has match_length ~30 while
        # a bare-format passport pattern matching the same A12345678
        # has match_length 9 — so insurance correctly wins.
        ml_a = a.match_length if a.match_length else (a.end - a.start)
        ml_b = b.match_length if b.match_length else (b.end - b.start)
        if ml_a != ml_b:
            return a if ml_a > ml_b else b

        priority_a = entity_configs.get(a.entity_id, None)
        priority_b = entity_configs.get(b.entity_id, None)
        pa = priority_a.priority if priority_a else 5
        pb = priority_b.priority if priority_b else 5
        if pa != pb:
            return a if pa < pb else b

        # Final tiebreaker: prefer the longer span so containing spans (e.g., a full URL)
        # are not discarded in favour of a short substring (e.g., an extracted token).
        return a if (a.end - a.start) >= (b.end - b.start) else b

    def merge_all(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
        text: str = "",
    ) -> list[DetectedSpan]:
        """Merge all spans from all sources without threshold split."""
        spans = self._validate_and_clean(spans, text)
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
