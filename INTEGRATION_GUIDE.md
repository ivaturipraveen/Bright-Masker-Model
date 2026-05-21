# Bright Masker — Integration Guide

**URL:** `https://36owxpb34jb9et-8000.proxy.runpod.net`  
**Auth:** None required  

---

## 1. Health Check

Verify the server is up before sending requests.

**cURL**
```bash
curl https://36owxpb34jb9et-8000.proxy.runpod.net/health
```

**Python**
```python
import httpx

r = httpx.get("https://36owxpb34jb9et-8000.proxy.runpod.net/health", timeout=10)
print(r.json())
```

**Expected output**
```json
{
  "status": "ok",
  "ner_model": "models/pii_gliner",
  "spacy_model": "en_core_web_lg",
  "entities_loaded": 105,
  "ner_threshold": 0.35
}
```

---

## 2. Mask Text — `POST /mask`

### Request parameter

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | string | Yes | Max 500,000 characters |

**cURL**
```bash
curl -X POST https://36owxpb34jb9et-8000.proxy.runpod.net/mask \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient Jane Doe, SSN 123-45-6789, email jane.doe@hospital.org, DOB 1985-03-22"}'
```

**Python**
```python
import httpx

result = httpx.post(
    "https://36owxpb34jb9et-8000.proxy.runpod.net/mask",
    json={"text": "Patient Jane Doe, SSN 123-45-6789, email jane.doe@hospital.org, DOB 1985-03-22"},
    timeout=60,
).json()

print(result["masked_text"])

for span in result["spans"]:
    print(f"  {span['entity_id']}  |  '{span['original']}'  →  '{span['masked']}'  |  {span['confidence']:.1%}")
```

### Expected output

```json
{
  "masked_text": "Patient [PERSON NAME 1], SSN [SSN 1], email [EMAIL ADDRESS 1], DOB [DATE OF BIRTH 1]",
  "original_text": "Patient Jane Doe, SSN 123-45-6789, email jane.doe@hospital.org, DOB 1985-03-22",
  "spans": [
    {
      "entity_id": "person_name",
      "display_name": "Person / Full Name / Alias",
      "original": "Jane Doe",
      "masked": "[PERSON NAME 1]",
      "confidence": 0.9994,
      "source": "ner",
      "strategy": "redact"
    },
    {
      "entity_id": "ssn",
      "display_name": "Social Security Number",
      "original": "123-45-6789",
      "masked": "[SSN 1]",
      "confidence": 0.9997,
      "source": "pattern",
      "strategy": "redact"
    },
    {
      "entity_id": "email_address",
      "display_name": "Email Address",
      "original": "jane.doe@hospital.org",
      "masked": "[EMAIL ADDRESS 1]",
      "confidence": 0.9999,
      "source": "pattern",
      "strategy": "redact"
    },
    {
      "entity_id": "date_of_birth",
      "display_name": "Date of Birth",
      "original": "1985-03-22",
      "masked": "[DATE OF BIRTH 1]",
      "confidence": 0.9981,
      "source": "pattern",
      "strategy": "redact"
    }
  ],
  "stats": {
    "total_ms": 148.5,
    "pattern_ms": 12.3,
    "ner_ms": 136.2,
    "spans_pattern": 3,
    "spans_ner": 1,
    "spans_total": 4,
    "language": "en",
    "ner_model": "models/pii_gliner"
  },
  "response_time_ms": 149.02
}
```

### Response field reference

**Top level**

| Field | Type | Description |
|-------|------|-------------|
| `masked_text` | string | Input text with all PII replaced |
| `original_text` | string | Your original input, unchanged |
| `spans` | array | One entry per detected entity |
| `stats` | object | Timing and counts |
| `response_time_ms` | float | Total time in milliseconds |

**Each `spans[]` item**

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | string | Entity type, e.g. `person_name`, `ssn`, `credit_card_number` |
| `display_name` | string | Human label, e.g. `Social Security Number` |
| `original` | string | The exact PII text found in your input |
| `masked` | string | The token written into `masked_text` |
| `confidence` | float | Score 0.0 – 1.0 |
| `source` | string | `"pattern"` (regex) or `"ner"` (GLiNER model) |
| `strategy` | string | Always `"redact"` for current entities |

**`stats` object**

| Field | Type | Description |
|-------|------|-------------|
| `total_ms` | float | Full pipeline time (ms) |
| `pattern_ms` | float | Regex layer time (ms) |
| `ner_ms` | float | GLiNER model time (ms) |
| `spans_pattern` | int | Entities found by regex |
| `spans_ner` | int | Entities found by GLiNER |
| `spans_total` | int | Total after merge and dedup |
| `language` | string | Detected language, e.g. `"en"` |
| `ner_model` | string | Model path used |

---

## 3. Error Responses

All errors return:
```json
{ "detail": "error message here" }
```

| Status | Reason |
|--------|--------|
| `400` | Empty or missing `text` field |
| `413` | Text exceeds 500,000 characters |
| `503` | Server still starting up — retry in a few seconds |
| `504` | Request timed out — split text into smaller chunks |

---

## 4. Notes

- **Same value → same token number.** `Jane Doe` appears twice → both become `[PERSON NAME 1]`, not 1 and 2. The `{n}` counter increments only for distinct values.
- **`source: "pattern"`** = regex match (structured PII like SSN, card numbers, dates, emails). Always high confidence.
- **`source: "ner"`** = GLiNER model detection (names, medications, locations, job titles). Score reflects model certainty.
- **Typical response time:** 100–250ms for average documents after warmup.
