# Masking Issues Tracker

Track all false positives, misclassifications, and missed detections.
Review daily — mark resolved after fix is confirmed on RunPod.
When enough "Needs Retrain" issues accumulate, run `./start.sh --samples 2000` on A100.

**Status legend:**
- `OPEN` — found, not yet fixed
- `FIXED` — fixed via config/code, no retraining needed, pending deploy verification
- `NEEDS RETRAIN` — fix requires new model weights
- `RESOLVED` — confirmed fixed on RunPod after deploy or retrain

---

## Issue Log

| # | Date Found | Input Text | Detected As | Should Be | Status | Fix Applied |
|---|---|---|---|---|---|---|
| 001 | 2026-05-19 | `CA-8821901` | PHONE NUMBER | INSURANCE ID | FIXED | Added pattern `[A-Z]{2}-\d{5,10}` to `health_plan_beneficiary_number` |
| 002 | 2026-05-19 | `Daniel Wilson` | EMAIL | PERSON | FIXED | Added `@` guard in `span_merger._validate_and_clean()` + email threshold → 0.92 |
| 003 | 2026-05-19 | `04/03/2025` | LICENSE PLATE | DATE | FIXED | Date regex guard in `span_merger._validate_and_clean()` + license_plate threshold → 0.92 |
| 004 | 2026-05-19 | `07/15/2025` | LICENSE PLATE | DATE | FIXED | Same as #003 |
| 005 | 2026-05-19 | `Riverside Community Health Center` | DOCTOR | HOSPITAL | FIXED | `hospital_name` priority → 1, `physician_name` threshold → 0.75 |
| 006 | 2026-05-19 | `the patient` | LOCATION | Not PII | FIXED | `city_name` threshold 0.40 → 0.72 |
| 007 | 2026-05-19 | `diabetic neuropathy` | LOCATION | Not PII | FIXED | Same as #006 |
| 008 | 2026-05-19 | `current medication therapy` | LOCATION | Not PII | FIXED | Same as #006 |
| 009 | 2026-05-19 | `Lipid Profile` | PERSON | Not PII | OPEN | `person_name` threshold → 0.62 applied but Presidio scores 0.85 — may still appear |
| 010 | 2026-05-19 | `Dr. Rebecca Moore supervised` | DOCTOR (with trailing word) | `Dr. Rebecca Moore` only | FIXED | Trim trailing lowercase words in `span_merger._validate_and_clean()` |
| 011 | 2026-05-19 | `Dr. Samuel Peterson and` | DOCTOR (with trailing word) | `Dr. Samuel Peterson` only | FIXED | Same as #010 |

---

## Retrain Queue

Issues that need model-level fixes — collect here and retrain in one batch.

| # | Issue | Why Retrain Needed |
|---|---|---|
| — | _(none yet)_ | — |

> **Trigger retraining** when 5+ issues are in this queue, or when precision drops below 95% on validation run.

---

## Resolved Log

Issues confirmed fixed on RunPod after deploy or retrain.

| # | Date Resolved | How Resolved |
|---|---|---|
| — | _(pending first deploy verification)_ | — |

---

## How to Add a New Issue

1. Run the test document through `/mask`
2. Find any wrong detection in the results panel
3. Add a row to the Issue Log with status `OPEN`
4. Investigate root cause (config threshold / pattern / model confusion)
5. Apply fix to `entities_config.yaml` or `pipeline/span_merger.py`
6. Change status to `FIXED` or `NEEDS RETRAIN`
7. After deploying and verifying on RunPod, move to Resolved Log
