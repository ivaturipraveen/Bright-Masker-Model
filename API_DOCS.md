# Bright Masker — Fine-Tuned PII Model API

**Base URL (RunPod / deployed):** `https://<your-runpod-url>`  
**Local dev:** `http://localhost:8000`  
**API version:** 3.1.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [POST /mask](#post-mask) ← primary endpoint
4. [POST /mask/stream](#post-maskstream) ← streaming / SSE
5. [GET /health](#get-health)
6. [GET /entities](#get-entities)
7. [POST /api/entities](#post-apientities)
8. [Error Codes](#error-codes)
9. [Masking Strategies](#masking-strategies)
10. [Supported Entity Types](#supported-entity-types)
11. [Code Examples](#code-examples)

---

## Authentication

The API currently runs without token-based authentication. Access is controlled at the infrastructure level (RunPod network / firewall). No `Authorization` header is required.

---

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mask` | Detect and mask PII — returns full JSON result |
| `POST` | `/mask/stream` | Same as `/mask` but streams progress via SSE |
| `GET` | `/health` | Server + model health check |
| `GET` | `/entities` | List all configured entity types |
| `POST` | `/api/entities` | Add a new entity type at runtime |
| `GET` | `/api/config` | UI / comparison panel configuration |

---

## POST /mask

The main endpoint. Sends text, receives masked text + all detected spans.

### Request

```
POST /mask
Content-Type: application/json
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Plain text to scan. Max 500,000 characters. |

```json
{
  "text": "Patient Jane Doe, DOB 1985-03-22, SSN 123-45-6789, email jane.doe@email.com. Prescribed Lisinopril 10mg."
}
```

### Response `200 OK`

```json
{
  "masked_text": "Patient [PERSON NAME 1], DOB [DATE OF BIRTH 1], SSN [SSN 1], email [EMAIL ADDRESS 1]. Prescribed [MEDICATION 1].",
  "original_text": "Patient Jane Doe, DOB 1985-03-22, SSN 123-45-6789, email jane.doe@email.com. Prescribed Lisinopril 10mg.",
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
      "entity_id": "date_of_birth",
      "display_name": "Date of Birth",
      "original": "1985-03-22",
      "masked": "[DATE OF BIRTH 1]",
      "confidence": 0.9981,
      "source": "pattern",
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
      "original": "jane.doe@email.com",
      "masked": "[EMAIL ADDRESS 1]",
      "confidence": 0.9999,
      "source": "pattern",
      "strategy": "redact"
    },
    {
      "entity_id": "medication_name",
      "display_name": "Prescription Medication / Drug Name with Dosage",
      "original": "Lisinopril 10mg",
      "masked": "[MEDICATION 1]",
      "confidence": 0.9873,
      "source": "ner",
      "strategy": "redact"
    }
  ],
  "stats": {
    "total_ms": 148.5,
    "pattern_ms": 12.3,
    "ner_ms": 136.2,
    "local_ms": 136.2,
    "spans_pattern": 3,
    "spans_ner": 2,
    "spans_total": 5,
    "language": "en",
    "ner_model": "models/pii_gliner"
  },
  "response_time_ms": 149.02
}
```

### Response Fields

#### Top level

| Field | Type | Description |
|-------|------|-------------|
| `masked_text` | `string` | Input text with all PII replaced by masked tokens |
| `original_text` | `string` | Original input text, unchanged |
| `spans` | `array` | Each detected PII entity (see below) |
| `stats` | `object` | Timing and pipeline statistics (see below) |
| `response_time_ms` | `float` | Total wall-clock time in milliseconds |

#### `spans[]` fields

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | `string` | Snake-case entity identifier, e.g. `person_name` |
| `display_name` | `string` | Human-readable label, e.g. `Person / Full Name / Alias` |
| `original` | `string` | The exact PII text found in the input |
| `masked` | `string` | The replacement token written into `masked_text` |
| `confidence` | `float` | Detection confidence score `0.0–1.0` |
| `source` | `string` | `"pattern"` (regex) or `"ner"` (GLiNER model) |
| `strategy` | `string` | Masking strategy applied — see [Masking Strategies](#masking-strategies) |

#### `stats` fields

| Field | Type | Description |
|-------|------|-------------|
| `total_ms` | `float` | Full pipeline time (ms) |
| `pattern_ms` | `float` | Regex/Presidio layer time (ms) |
| `ner_ms` | `float` | GLiNER NER layer time (ms) |
| `local_ms` | `float` | Wall-clock of parallel pattern+NER phase |
| `spans_pattern` | `int` | Entities found by pattern layer |
| `spans_ner` | `int` | Entities found by NER layer |
| `spans_total` | `int` | Total entities after merge and dedup |
| `language` | `string` | Detected language code, e.g. `"en"` |
| `ner_model` | `string` | Model path/name used for NER |

---

## POST /mask/stream

Same semantics as `/mask` but streams pipeline progress as **Server-Sent Events (SSE)**. Useful for long documents or showing a live progress indicator.

### Request

Same as `/mask`:

```
POST /mask/stream
Content-Type: application/json

{ "text": "..." }
```

### SSE Event Stream

The server sends multiple `data:` events until the stream closes.

**Progress events** (emitted after each pipeline step):

```
data: {"type": "step", "step": "pattern", "ms": 12.3, "spans": 3}

data: {"type": "step", "step": "ner", "ms": 136.2, "spans": 9}

data: {"type": "step", "step": "merge", "ms": 0.4, "spans": 10}
```

**Final complete event** (last event — contains the full result):

```
data: {"type": "complete", "result": { ...same shape as /mask response... }}
```

**Error event** (only on failure):

```
data: {"type": "error", "message": "Pipeline not initialized"}
```

### JavaScript example

```javascript
const response = await fetch('/mask/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: inputText }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6));

    if (event.type === 'step') {
      console.log(`Step ${event.step}: ${event.spans} spans in ${event.ms}ms`);
    } else if (event.type === 'complete') {
      console.log('Masked text:', event.result.masked_text);
      console.log('Entities:', event.result.spans);
    } else if (event.type === 'error') {
      console.error('Error:', event.message);
    }
  }
}
```

---

## GET /health

Returns the current server and model status.

### Response `200 OK`

```json
{
  "status": "ok",
  "ner_model": "models/pii_gliner",
  "spacy_model": "en_core_web_lg",
  "entities_loaded": 105,
  "ner_threshold": 0.35
}
```

| Field | Description |
|-------|-------------|
| `status` | `"ok"` when ready |
| `ner_model` | Path or name of the loaded GLiNER model |
| `spacy_model` | spaCy model used for the pattern layer |
| `entities_loaded` | Number of active entity configs loaded |
| `ner_threshold` | Current GLiNER confidence threshold |

---

## GET /entities

Returns the full entity registry with category metadata — useful for building UI dropdowns or understanding what the model can detect.

### Response `200 OK` (abbreviated)

```json
{
  "entities": [
    {
      "id": "person_name",
      "display_name": "Person / Full Name / Alias",
      "description": "...",
      "policy": ["hipaa"],
      "category": "hipaa",
      "priority": 3,
      "enabled": true,
      "is_active": true,
      "confidence_threshold": 0.85,
      "gliner_label": "person_name",
      "has_pattern": false,
      "has_presidio": true,
      "has_gliner": true,
      "strategy": "redact",
      "format": "[PERSON NAME {n}]",
      "pattern_count": 0
    }
  ],
  "categories": [
    { "id": "hipaa", "label": "HIPAA Safe Harbor", "description": "18 PHI identifiers", "icon": "🛡" },
    { "id": "pci_dss", "label": "PCI-DSS v4.0", "description": "Cardholder & authentication data", "icon": "💳" },
    { "id": "general", "label": "General PII", "description": "Credentials & sensitive attributes", "icon": "🔐" },
    { "id": "law_enforcement", "label": "Law Enforcement", "description": "CJIS & criminal justice records", "icon": "⚖" },
    { "id": "transportation", "label": "Transportation", "description": "Vehicle & transit identifiers", "icon": "🚗" }
  ],
  "total": 105,
  "active_count": 105,
  "enabled_count": 105,
  "by_category": {
    "hipaa": 34,
    "pci_dss": 15,
    "general": 32,
    "law_enforcement": 20,
    "transportation": 4
  },
  "pipeline_loaded": 105
}
```

---

## POST /api/entities

Add a new custom entity type at runtime — no server restart required.

### Request

```json
{
  "id": "employee_badge",
  "display_name": "Employee Badge Number",
  "description": "Internal badge ID in format EMP-XXXXX",
  "gliner_label": "employee badge number",
  "policy": ["general"],
  "priority": 5,
  "confidence_threshold": 0.70,
  "masking_format": "[BADGE {n}]",
  "pattern": "EMP-\\d{5}"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | ✅ | Lowercase, letters/digits/underscores, 2–64 chars |
| `display_name` | `string` | ✅ | Human-readable label |
| `description` | `string` | | Detection hint for the model |
| `gliner_label` | `string` | | Label string passed to GLiNER (defaults to `display_name`) |
| `policy` | `string[]` | | One or more of: `hipaa`, `pci_dss`, `general`, `law_enforcement`, `transportation` |
| `priority` | `int` | | 1 (highest) to 10. Resolves overlapping spans. Default `5` |
| `confidence_threshold` | `float` | | Override global threshold (0.0–1.0). Default `0.55` |
| `masking_format` | `string` | | Token template. `{n}` = sequential counter. Default `[ENTITY {n}]` |
| `pattern` | `string` | | Optional regex. Capturing group = the PII value. |

### Response `200 OK`

```json
{
  "status": "created",
  "id": "employee_badge",
  "total_entities": 106
}
```

### Error responses

| Status | Reason |
|--------|--------|
| `400` | `id` format invalid or `display_name` empty |
| `409` | Entity with that `id` already exists |
| `503` | Pipeline not yet initialized |

---

## Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| `400` | Bad request — empty text or invalid field |
| `404` | Path not found |
| `409` | Conflict — duplicate entity `id` |
| `413` | Text exceeds 500,000 character limit |
| `500` | Internal server error |
| `503` | Pipeline not yet initialized (server still loading) |

All error responses follow:

```json
{ "detail": "Human-readable error message" }
```

---

## Masking Strategies

Configured per entity in `entities_config.yaml`. All current entities use `redact`.

| Strategy | Behaviour | Example output |
|----------|-----------|----------------|
| `redact` | Replace with a labelled token | `[SSN 1]` |
| `substitute` | Replace with a realistic fake value | `John Smith` → `Michael Brown` |
| `hash` | Replace with a deterministic hash | `[HASH:a3f9c2]` |
| `partial_redact` | Show only part of the value | `4111 **** **** 1111` |
| `encrypt` | AES-encrypt the value (reversible) | `[ENC:dGVzdA==]` |

The `{n}` counter in the format string is per entity type per document — the same original value always gets the same number so co-references stay traceable.

---

## Supported Entity Types

105 entity types across 5 regulatory categories.

### HIPAA Safe Harbor (34 entities)

| `entity_id` | Display Name |
|-------------|--------------|
| `person_name` | Person / Full Name / Alias |
| `physician_name` | Physician / Doctor Name |
| `street_address` | Street Address Line |
| `city_name` | City Name / Town |
| `us_state` | US State or Territory Abbreviation |
| `zipcode` | ZIP Code / Postal Code |
| `precise_geolocation` | Precise Geolocation |
| `date_of_birth` | Date of Birth |
| `clinical_date` | Clinical / Procedure / Event Date |
| `phone_number` | Phone Number |
| `fax_number` | Fax Number |
| `email_address` | Email Address |
| `ssn` | Social Security Number |
| `medical_record_number` | Medical Record Number (MRN) |
| `health_plan_beneficiary_number` | Health Plan Beneficiary / Member Number |
| `insurance_policy_number` | Insurance Policy Number |
| `billing_number` | Billing Number |
| `medical_license_number` | Medical License / Certificate Number |
| `dea_number` | DEA Registration Number |
| `npi_number` | National Provider Identifier (NPI) |
| `passport_number` | Passport Number |
| `drivers_license` | Driver's License Number |
| `device_identifier` | Device Identifier / MAC Address / IMEI |
| `url_with_pii` | URL with PII |
| `ip_address` | IP Address |
| `biometric_facial_recognition` | Biometric — Facial Recognition |
| `biometric_voiceprint` | Biometric — Voiceprint |
| `biometric_iris_scan` | Biometric — Iris Scan |
| `biometric_dna` | Biometric — DNA Information |
| `fingerprint` | Fingerprint Data |
| `organization_name` | Company / Organization Name |
| `signature` | Signature |
| `medication_name` | Prescription Medication / Drug Name with Dosage |
| `hospital_name` | Hospital / Medical Center / Clinic Name |

### PCI-DSS v4.0 (15 entities)

| `entity_id` | Display Name |
|-------------|--------------|
| `credit_card_number` | Credit / Debit Card Number (PAN) |
| `card_holder_name` | Card Holder Name |
| `card_expiration_date` | Card Expiry MM/YY |
| `card_service_code` | Card Service Code / CVV / CVC |
| `card_track_data` | Card Track Data (Magnetic Stripe) |
| `card_pin` | Card PIN / PIN Block |
| `card_cryptogram` | Card Cryptogram / Dynamic Auth Value |
| `card_iin_bin` | Card IIN / BIN |
| `bank_account_number` | Bank Account Number |
| `bank_routing_number` | Bank Routing Number |
| `iban` | IBAN |
| `swift_bic_code` | SWIFT / BIC Code |
| `merchant_id` | Merchant ID (MID) |
| `terminal_id` | Terminal ID (TID) |
| `transaction_id` | Transaction ID |

### General PII (32 entities)

| `entity_id` | Display Name |
|-------------|--------------|
| `password` | Password |
| `username` | Username / User ID |
| `confidential` | Confidential Data |
| `cookie_session_token` | Session Token / API Key / JWT |
| `racial_ethnic_origin` | Racial or Ethnic Origin |
| `physical_characteristics` | Physical Characteristics |
| `race` | Race |
| `religion` | Religion |
| `employment_history` | Employment History |
| `performance_evaluation` | Performance Evaluation |
| `student_records_ferpa` | Student Records (FERPA) |
| `state_id_number` | State ID Number |
| `employee_id` | Employee ID / Staff Number |
| `vehicle_registration` | Vehicle Registration Number |
| `student_id` | Student ID Number |
| `tax_id_number` | Tax Identification Number / TIN / EIN |
| `flight_number` | Flight Number |
| `booking_reference` | Booking / Reservation Reference |
| `claim_number` | Insurance Claim Number |
| `application_id` | Government Application / Reference Number |
| `insurance_company_name` | Insurance Company Name |
| `university_name` | University / College Name |
| `law_firm_name` | Law Firm / Legal Practice Name |
| `court_name` | Court / Judicial Body Name |
| `hotel_name` | Hotel / Accommodation Name |
| `financial_amount` | Transaction / Financial Amount |
| `card_type` | Payment Card Brand |
| `card_last4` | Card Last 4 Digits |
| `bank_name` | Bank / Financial Institution Name |

### Law Enforcement / CJIS (20 entities)

| `entity_id` | Display Name |
|-------------|--------------|
| `fbi_number` | FBI Number |
| `chri` | Criminal History Record Information (CHRI) |
| `arrest_record` | Arrest Record |
| `case_number` | Case Number |
| `warrant_data` | Warrant Data |
| `incident_report` | Incident Report |
| `incarceration_info` | Incarceration Information |
| `missing_person_report` | Missing Person Report |
| `wanted_person_report` | Wanted Person Report |
| `sex_offender_report` | Sex Offender Report |
| `protection_orders` | Protection Orders |
| `foreign_fugitives` | Foreign Fugitives Record |
| `identity_theft_victims` | Identity Theft Victim Record |
| `gang_terrorist_member` | Gang / Terrorist Member Record |
| `supervised_release` | Supervised Release Record |
| `probation_record` | Probation Record |
| `parole_record` | Parole Record |
| `stolen_vehicle` | Stolen Vehicle Record |
| `stolen_guns` | Stolen Guns Record |
| `stolen_license_plate` | Stolen License Plate |

### Transportation (4 entities)

| `entity_id` | Display Name |
|-------------|--------------|
| `vehicle_vin` | Vehicle Identification Number (VIN) |
| `license_plate` | License Plate Number |
| `driver_history` | Driver History Record |
| `bart_employee_id` | BART Employee ID |

---

## Code Examples

### Python (httpx)

```python
import httpx

BASE = "https://<your-runpod-url>"   # or http://localhost:8000

# ── Health check ────────────────────────────────────────
r = httpx.get(f"{BASE}/health", timeout=10)
print(r.json())
# {"status": "ok", "ner_model": "models/pii_gliner", ...}

# ── Mask text ───────────────────────────────────────────
r = httpx.post(
    f"{BASE}/mask",
    json={"text": "My name is John Smith and my SSN is 123-45-6789."},
    timeout=60,
)
result = r.json()

print("Masked:", result["masked_text"])
# Masked: My name is [PERSON NAME 1] and my SSN is [SSN 1].

for span in result["spans"]:
    print(f"  [{span['entity_id']}]  {span['original']!r}  →  {span['masked']!r}  ({span['confidence']:.2%})")
```

### Python (requests)

```python
import requests

BASE = "https://<your-runpod-url>"

response = requests.post(
    f"{BASE}/mask",
    json={"text": "Card 4111111111111111 exp 12/26, CVV 123"},
    timeout=60,
)
data = response.json()
print(data["masked_text"])
print(f"Detected {data['stats']['spans_total']} entities in {data['response_time_ms']:.0f}ms")
```

### cURL

```bash
# Health check
curl https://<your-runpod-url>/health

# Mask text
curl -X POST https://<your-runpod-url>/mask \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Doe, DOB 1980-01-01, MRN 78345, email j.doe@hospital.org"}' \
  | python3 -m json.tool
```

### JavaScript / Node.js (fetch)

```javascript
const BASE = 'https://<your-runpod-url>';

const response = await fetch(`${BASE}/mask`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Driver license D1234567, plate ABC-1234, VIN 1HGCM82633A123456',
  }),
});

const { masked_text, spans, stats } = await response.json();

console.log('Masked:', masked_text);
console.log(`Found ${stats.spans_total} entities in ${stats.total_ms.toFixed(0)}ms`);
spans.forEach(s =>
  console.log(`  ${s.display_name}: "${s.original}" → "${s.masked}" (${(s.confidence * 100).toFixed(1)}%)`)
);
```

### Streaming (JavaScript SSE)

```javascript
const BASE = 'https://<your-runpod-url>';

async function maskWithProgress(text, onStep, onComplete) {
  const response = await fetch(`${BASE}/mask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    for (const line of buffer.split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === 'step')     onStep(event);
      if (event.type === 'complete') onComplete(event.result);
      if (event.type === 'error')    throw new Error(event.message);
    }
    buffer = buffer.includes('\n') ? buffer.split('\n').at(-1) : buffer;
  }
}

// Usage:
await maskWithProgress(
  'SSN 123-45-6789, credit card 4111111111111111',
  step    => console.log(`Step: ${step.step} — ${step.spans} spans`),
  result  => console.log('Done:', result.masked_text),
);
```

---

## Detection Sources

Each detected span has a `source` field indicating which pipeline layer found it:

| Source | Layer | How it works |
|--------|-------|-------------|
| `"pattern"` | Regex + Presidio | Fast, rule-based. Catches structured PII like SSNs, credit cards, phone numbers, dates. Zero false negatives on well-formatted values. |
| `"ner"` | GLiNER fine-tuned | Neural entity recognition. Catches contextual PII like names, job titles, hospital names, medications, locations. Confidence-scored. |

When both layers detect the same span, the pipeline merges them and keeps the higher-confidence detection with the higher-priority entity type.

---

*Model: GLiNER fine-tuned · 105 entity types · HIPAA / PCI-DSS / CJIS / Transportation · no LLM required*
