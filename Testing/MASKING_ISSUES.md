# Masking Issues Tracker

Track all false positives, misclassifications, and missed detections.
Review daily — mark RESOLVED after fix is confirmed working on RunPod.
When Retrain Queue hits 5+ issues, trigger `./start.sh --samples 2000` on A100.

**Status:**
- `OPEN` — found, not yet fixed
- `FIXED` — config/code fix applied, pending RunPod verification
- `NEEDS RETRAIN` — only the model can fix this, queued for next training run
- `RESOLVED` — confirmed working on RunPod after deploy or retrain

---

## Issue Log

| # | Date | Input Text | Detected As | Should Be | Status | Fix |
|---|---|---|---|---|---|---|
| 001 | 2026-05-19 | `CA-8821901` | PHONE NUMBER | INSURANCE ID | RESOLVED | Added `[A-Z]{2}-\d{5,10}` pattern to `health_plan_beneficiary_number` — confirmed working in latest run |
| 002 | 2026-05-19 | `Daniel Wilson` | EMAIL | PERSON | FIXED | `@` guard in `span_merger._validate_and_clean()` + email threshold → 0.92 |
| 003 | 2026-05-19 | `04/03/2025` | LICENSE PLATE | DATE | FIXED | Date regex guard in `span_merger._validate_and_clean()` + license_plate threshold → 0.92 |
| 004 | 2026-05-19 | `07/15/2025` | LICENSE PLATE | DATE | FIXED | Same as #003 |
| 005 | 2026-05-19 | `Riverside Community Health Center` | DOCTOR | HOSPITAL | FIXED | `hospital_name` priority → 1, `physician_name` threshold → 0.75 |
| 006 | 2026-05-19 | `the patient` | LOCATION | Not PII | FIXED | `city_name` threshold 0.40 → 0.72 |
| 007 | 2026-05-19 | `diabetic neuropathy` | LOCATION | Not PII | FIXED | Same as #006 |
| 008 | 2026-05-19 | `current medication therapy` | LOCATION | Not PII | FIXED | Same as #006 |
| 009 | 2026-05-19 | `Lipid Profile` | PERSON | Not PII | FIXED | Added to `_PERSON_NAME_BLOCKLIST` in `span_merger._validate_and_clean()` |
| 010 | 2026-05-19 | `Dr. Rebecca Moore supervised` | DOCTOR (extra word) | `Dr. Rebecca Moore` | FIXED | Trailing lowercase word trim in `span_merger._validate_and_clean()` |
| 011 | 2026-05-19 | `Dr. Samuel Peterson and` | DOCTOR (extra word) | `Dr. Samuel Peterson` | FIXED | Same as #010 |
| 012 | 2026-05-19 | `BL-9910274` | BILLING NO (GLiNER only) | BILLING NO | FIXED | Added `billing account [A-Z]-\d` pattern — now pattern layer covers it too |
| 013 | 2026-05-19 | `Complete Blood Count` | Potential PERSON | Not PII | FIXED | Added to `_PERSON_NAME_BLOCKLIST` in `span_merger._validate_and_clean()` |
| 014 | 2026-05-19 | `Comprehensive Metabolic Panel` | Potential PERSON | Not PII | FIXED | Same as #013 |
| 015 | 2026-05-19 | `confidential` (in "remains confidential") | CONFIDENTIAL | Not PII in context | FIXED | Pattern changed to `(?-i:...)` — now only fires on uppercase CONFIDENTIAL |
| 016 | 2026-05-19 | `The healthcare` (from "The healthcare team") | HOSPITAL | Not PII | FIXED | `hospital_name` pattern prefix changed from `{1,5}` → `{2,5}` — requires 2+ capitalized words before suffix |

---

## Retrain Queue

Issues that require new model weights to fix properly.

| # | Issue | Why Retrain Needed |
|---|---|---|
| — | _(none yet)_ | — |

> Trigger retraining when 5+ issues are queued here, or validation drops below 95%.

---

## Resolved Log

| # | Date Resolved | Confirmed On | How Resolved |
|---|---|---|---|
| 001 | 2026-05-19 | Local run | `CA-8821901` correctly shown as `[INSURANCE ID 1]` 100% Pattern in latest output |

---

## All Config Changes

| Entity | Field | Before | After | Fixes |
|---|---|---|---|---|
| `person_name` | confidence_threshold | 0.45 | 0.62 | #009 |
| `city_name` | confidence_threshold | 0.40 | 0.72 | #006 #007 #008 |
| `physician_name` | confidence_threshold | 0.45 | 0.75 | #005 |
| `license_plate` | confidence_threshold | _(global 0.85)_ | 0.92 | #003 #004 |
| `hospital_name` | priority | 2 | 1 | #005 |
| `hospital_name` | pattern prefix count | `{1,5}` | `{2,5}` | #016 |
| `email_address` | confidence_threshold | _(global 0.85)_ | 0.92 | #002 |
| `email_address` | patterns | none | added `@` format regex | #002 |
| `health_plan_beneficiary_number` | patterns | no state-prefix | added `[A-Z]{2}-\d{5,10}` | #001 |
| `billing_number` | patterns | digits only | added `billing account [A-Z]-\d` | #012 |
| `confidential` | pattern | case-insensitive | `(?-i:...)` uppercase only | #015 |

## All Code Changes — `pipeline/span_merger.py`

`_validate_and_clean()` runs before dedup on every request:

| Check | Fixes |
|---|---|
| Reject `email_address` spans with no `@` | #002 |
| Reject `license_plate` spans matching date format `\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}` | #003 #004 |
| Trim trailing lowercase words from `physician_name` spans | #010 #011 |
| Reject `person_name` spans in `_PERSON_NAME_BLOCKLIST` | #009 #013 #014 |

`_PERSON_NAME_BLOCKLIST`:
`lipid profile, complete blood count, comprehensive metabolic panel, cbc, cmp, urinalysis, hba1c, blood pressure, heart rate, blood glucose, oxygen saturation, metabolic panel`

---

## How to Add a New Issue

1. Run text through `/mask` endpoint
2. Find wrong detection in the results panel
3. Add row to Issue Log with status `OPEN`
4. Investigate: config threshold / pattern gap / model confusion
5. Apply fix → change status to `FIXED` or `NEEDS RETRAIN`
6. After deploy + RunPod verify → move to Resolved Log with date
