"""
Validate masking: call /mask API, then OpenAI Pass/Fail on every row.

Input : validation_dataset.csv
Output: validation_results_<DD-MM-YYYY>__<hh-mmam/pm>.csv  (new file each run)
        validation_failures_<same stamp>.csv  (failed rows only)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

TESTING_DIR = Path(__file__).resolve().parent
DATASET_FILE = TESTING_DIR / "validation_dataset.csv"

API_BASE_URL = os.getenv(
    "MASK_API_BASE_URL",
    "https://36owxpb34jb9et-8000.proxy.runpod.net",
).rstrip("/")
MASK_URL = f"{API_BASE_URL}/mask"
API_TIMEOUT_SEC = int(os.getenv("MASK_API_TIMEOUT_SEC", "120"))
OPENAI_MODEL = os.getenv("OPENAI_VALIDATION_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ── Column names (input + output) ───────────────────────────────────────────
COL_EXAMPLE = "Example ID"
COL_ORIGINAL = "Original Query"
COL_EXPECTED = "Expected Masked Response"
COL_MODEL = "Model Masked Response"
COL_API_SEC = "API Response Time (seconds)"
COL_RESULT = "OpenAI Validation Result"
COL_MISSED = "Missed Entities"
COL_REMARKS = "OpenAI Validation Remarks"

INPUT_COLUMNS = [COL_EXAMPLE, COL_ORIGINAL, COL_EXPECTED]
OUTPUT_COLUMNS = INPUT_COLUMNS + [
    COL_MODEL,
    COL_API_SEC,
    COL_RESULT,
    COL_MISSED,
    COL_REMARKS,
]

ENTITY_EXTRACTION_PROMPT = """You are a PII/PHI entity extractor.

Given a text, extract every sensitive value that must be masked for privacy.
Return ONLY a JSON array of strings — the exact values as they appear in the text.

Include: person names, dates of birth, dates, addresses, cities, states, ZIP codes,
phone numbers, emails, card numbers, IBANs, routing numbers, account numbers,
insurance IDs, MRNs, employee IDs, usernames, passwords, IP addresses, URLs,
tax IDs, driver licence numbers, case/docket numbers, medication names,
doctor names, hospital names, organisation names used as identifiers,
terminal IDs, merchant IDs, transaction amounts, and any other value that
could identify or expose a person or account.

Return format (JSON array only, no other text):
["value1", "value2", "value3"]"""

# Legacy header aliases when reading old files
_READ_ALIASES = {
    COL_EXAMPLE: ["example_id", "example_name", "Example", "example no", "Example ID"],
    COL_ORIGINAL: ["original_query", "input_query", "original query", "Original Query"],
    COL_EXPECTED: [
        "expected_masked_response",
        "expected_masked_output",
        "original expected Response",
        "Expected Masked Response",
    ],
}


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(TESTING_DIR.parent / ".env")


def _pick(row: dict[str, str], col: str) -> str:
    for key in _READ_ALIASES.get(col, [col]):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def format_run_stamp(when: datetime | None = None) -> str:
    """Display/file stamp: 19/05/2025//03:04pm"""
    when = when or datetime.now()
    date_s = when.strftime("%d/%m/%Y")
    hour12 = when.hour % 12 or 12
    time_s = f"{hour12:02d}:{when.strftime('%M')}{'am' if when.hour < 12 else 'pm'}"
    return f"{date_s}//{time_s}"


def format_run_filename(when: datetime | None = None) -> str:
    """Filesystem-safe: 19-05-2025__03-04pm"""
    when = when or datetime.now()
    date_s = when.strftime("%d-%m-%Y")
    hour12 = when.hour % 12 or 12
    time_s = f"{hour12:02d}-{when.strftime('%M')}{'am' if when.hour < 12 else 'pm'}"
    return f"{date_s}__{time_s}"


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows: list[dict[str, str]] = []
        for i, row in enumerate(csv.DictReader(f), start=1):
            original = _pick(row, COL_ORIGINAL)
            if not original:
                continue
            example = _pick(row, COL_EXAMPLE) or f"Example {i}"
            rows.append({
                COL_EXAMPLE: example,
                COL_ORIGINAL: original,
                COL_EXPECTED: _pick(row, COL_EXPECTED),
            })
        return rows


def call_mask_api(text: str) -> tuple[str, str]:
    try:
        resp = requests.post(
            MASK_URL,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=API_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        return f"[API ERROR] {exc}", ""

    if resp.status_code != 200:
        body = resp.text[:500]
        return f"[API ERROR] HTTP {resp.status_code}: {body}", ""

    data = resp.json()
    masked = str(data.get("masked_text", "")).strip() or "[API ERROR] Empty masked_text"
    api_ms = data.get("response_time_ms")
    if api_ms is None:
        return masked, ""
    return masked, f"{float(api_ms) / 1000:.3f}"


def _openai_post(messages: list, api_key: str, timeout: int = 60) -> dict | None:
    """POST to OpenAI chat completions; returns parsed JSON or None on failure."""
    for _ in range(2):
        try:
            resp = requests.post(
                OPENAI_URL,
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0,
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return None


def extract_sensitive_values(original_query: str, api_key: str) -> list[str]:
    """Step 1 — ask OpenAI to list all sensitive raw values in the original text."""
    result = _openai_post(
        [
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": original_query[:8000]},
        ],
        api_key,
        timeout=60,
    )
    if not result:
        return []
    try:
        content = result["choices"][0]["message"]["content"].strip()
        # parse JSON array from the response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            values = json.loads(content[start:end])
            if isinstance(values, list):
                return [str(v).strip() for v in values if str(v).strip()]
    except Exception:
        pass
    return []


def openai_validate(
    original_query: str,
    expected_masked_response: str,
    model_masked_response: str,
    api_key: str,
) -> tuple[str, str, str]:
    """
    Two-step validation — no hallucination possible:
      Step 1: OpenAI extracts the list of raw sensitive values from Original Query.
      Step 2: Python checks whether any of those values appear in Model Masked Response.
    Returns (PASS|FAIL, missed_entities_text, remarks).
    """
    sensitive_values = extract_sensitive_values(original_query, api_key)

    if not sensitive_values:
        return "FAIL", "", "Could not extract sensitive entities from original query (OpenAI error)."

    # Step 2 — pure string check, no AI involved
    model_lower = model_masked_response.lower()
    leaked: list[str] = []
    for val in sensitive_values:
        if len(val) >= 3 and val.lower() in model_lower:
            leaked.append(val)

    if not leaked:
        return (
            "PASS",
            "",
            "No sensitive data leakage detected. All original values are replaced by placeholders.",
        )

    missed_str = " | ".join(leaked)
    remarks = f"{len(leaked)} sensitive value(s) still visible in model output: {', '.join(leaked[:5])}{'...' if len(leaked) > 5 else ''}."
    return "FAIL", missed_str, remarks


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_validation(limit: int | None = None) -> Path:
    _load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAI_API_KEY is required in .env")

    if not DATASET_FILE.is_file():
        sys.exit(f"Dataset not found: {DATASET_FILE}")

    rows = load_dataset(DATASET_FILE)
    if limit:
        rows = rows[:limit]

    run_at = datetime.now()
    run_stamp = format_run_stamp(run_at)
    file_stamp = format_run_filename(run_at)
    output_path = TESTING_DIR / f"validation_results_{file_stamp}.csv"
    failures_path = TESTING_DIR / f"validation_failures_{file_stamp}.csv"

    passed = failed = 0
    api_times_sec: list[float] = []
    failures: list[dict[str, str]] = []

    print(f"Mask API  : {MASK_URL}")
    print(f"Dataset   : {DATASET_FILE.name} ({len(rows)} rows)")
    print(f"Run time  : {run_stamp}")
    print(f"Results   : {output_path.name}\n")

    results: list[dict[str, str]] = []

    for row in rows:
        example = row[COL_EXAMPLE]
        original = row[COL_ORIGINAL]
        expected = row[COL_EXPECTED]

        model_out, api_sec = call_mask_api(original)
        if api_sec:
            api_times_sec.append(float(api_sec))

        result, missed, remarks = openai_validate(original, expected, model_out, api_key)

        if result == "PASS":
            passed += 1
            mark = "PASS"
        else:
            failed += 1
            mark = "FAIL"

        print(f"  {example[:30]:30} {mark}  api={api_sec or '—'}s")

        out_row = {
            **row,
            COL_MODEL: model_out,
            COL_API_SEC: api_sec,
            COL_RESULT: result,
            COL_MISSED: missed,
            COL_REMARKS: remarks,
        }
        results.append(out_row)
        if result == "FAIL":
            failures.append(out_row)

        _write_csv(output_path, results)

    _write_csv(failures_path, failures)

    total = len(rows)
    avg_api = sum(api_times_sec) / len(api_times_sec) if api_times_sec else 0.0

    print("\n" + "=" * 70)
    print(f"  PASS: {passed}   FAIL: {failed}   TOTAL: {total}")
    if total:
        print(f"  Pass rate: {100 * passed / total:.1f}%")
    if api_times_sec:
        print(f"  Avg API time: {avg_api:.3f} s")
    print(f"  Full results : {output_path}")
    print(f"  Failures only: {failures_path} ({len(failures)} rows)")
    print("=" * 70)

    if failures:
        print("\n--- FAILURES ---\n")
        for f in failures:
            print(f"{f[COL_EXAMPLE]} — {f[COL_RESULT]}")
            if f[COL_MISSED]:
                print(f"  Missed: {f[COL_MISSED]}")
            print(f"  Remarks: {f[COL_REMARKS]}\n")

    return output_path


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_validation(limit=limit)
