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


| #   | Date       | Input Text                              | Detected As              | Should Be             | Status   | Fix                                                                                               |
| --- | ---------- | --------------------------------------- | ------------------------ | --------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| 001 | 2026-05-19 | `CA-8821901`                            | PHONE NUMBER             | INSURANCE ID          | RESOLVED | Added `[A-Z]{2}-\d{5,10}` pattern to `health_plan_beneficiary_number`                             |
| 002 | 2026-05-19 | `Daniel Wilson`                         | EMAIL                    | PERSON                | RESOLVED | `@` guard in `span_merger._validate_and_clean()` + email threshold → 0.92                         |
| 003 | 2026-05-19 | `04/03/2025`                            | LICENSE PLATE            | DATE                  | RESOLVED | Date regex guard in `span_merger._validate_and_clean()` + `recorded as` pattern added             |
| 004 | 2026-05-19 | `07/15/2025`                            | MISSED                   | DATE                  | FIXED    | Added `scheduled|planned|set|confirmed for` pattern to `clinical_date`                            |
| 005 | 2026-05-19 | `Riverside Community Health Center`     | DOCTOR                   | HOSPITAL              | FIXED    | GLiNER physician_name spans without "Dr."/"Doctor" prefix now rejected in `_validate_and_clean()` |
| 006 | 2026-05-19 | `the patient`                           | LOCATION                 | Not PII               | RESOLVED | city_name pattern made case-sensitive `(?-i:[A-Z])` — lowercase phrases no longer match           |
| 007 | 2026-05-19 | `diabetic neuropathy`                   | LOCATION                 | Not PII               | RESOLVED | Same as #006                                                                                      |
| 008 | 2026-05-19 | `current medication therapy`            | LOCATION                 | Not PII               | RESOLVED | Same as #006                                                                                      |
| 009 | 2026-05-19 | `Lipid Profile`                         | PERSON                   | Not PII               | RESOLVED | Added to `_PERSON_NAME_BLOCKLIST` in `span_merger._validate_and_clean()`                          |
| 010 | 2026-05-19 | `Dr. Rebecca Moore supervised`          | DOCTOR (extra word)      | `Dr. Rebecca Moore`   | RESOLVED | physician_name pattern made case-sensitive — lowercase trailing words no longer captured          |
| 011 | 2026-05-19 | `Dr. Samuel Peterson and`               | DOCTOR (extra word)      | `Dr. Samuel Peterson` | RESOLVED | Same as #010                                                                                      |
| 012 | 2026-05-19 | `BL-9910274`                            | BILLING NO (GLiNER only) | BILLING NO (Pattern)  | RESOLVED | Added `billing account [A-Z]-\d` pattern — pattern layer now covers it                            |
| 013 | 2026-05-19 | `Complete Blood Count`                  | Potential PERSON         | Not PII               | RESOLVED | Added to `_PERSON_NAME_BLOCKLIST`                                                                 |
| 014 | 2026-05-19 | `Comprehensive Metabolic Panel`         | Potential PERSON         | Not PII               | RESOLVED | Same as #013                                                                                      |
| 015 | 2026-05-19 | `confidential` (lowercase, in sentence) | CONFIDENTIAL             | Not PII               | RESOLVED | Pattern changed to `(?-i:...)` — uppercase only                                                   |
| 016 | 2026-05-19 | `The healthcare team`                   | HOSPITAL                 | Not PII               | RESOLVED | hospital_name pattern prefix `{1,5}` → `{2,5}` — requires 2+ capitalized words                    |
| 017 | 2026-05-19 | `HIPAA compliance guidelines`           | CONFIDENTIAL             | Not PII               | RESOLVED | `confidential` confidence_threshold → 0.95                                                        |


---

## Retrain Queue


| #   | Issue        | Why Retrain Needed |
| --- | ------------ | ------------------ |
| —   | *(none yet)* | —                  |


> Trigger retraining when 5+ issues are queued here, or validation drops below 95%.

---

## Resolved Log


| #   | Date Resolved | Confirmed On    | How Resolved                                   |
| --- | ------------- | --------------- | ---------------------------------------------- |
| 001 | 2026-05-19    | RunPod RTX 4090 | Pattern confirmed working                      |
| 002 | 2026-05-19    | RunPod RTX 4090 | Daniel Wilson correctly → PERSON 3             |
| 003 | 2026-05-19    | RunPod RTX 4090 | 04/03/2025 correctly → DATE 4 (Pattern)        |
| 006 | 2026-05-19    | RunPod RTX 4090 | "the patient" no longer flagged                |
| 007 | 2026-05-19    | RunPod RTX 4090 | "diabetic neuropathy" no longer flagged        |
| 008 | 2026-05-19    | RunPod RTX 4090 | "current medication therapy" no longer flagged |
| 009 | 2026-05-19    | RunPod RTX 4090 | Lipid Profile no longer flagged as PERSON      |
| 010 | 2026-05-19    | RunPod RTX 4090 | Dr. Rebecca Moore — clean, no trailing word    |
| 011 | 2026-05-19    | RunPod RTX 4090 | Dr. Samuel Peterson — clean, no trailing word  |
| 012 | 2026-05-19    | RunPod RTX 4090 | BL-9910274 → BILLING NO via Pattern            |
| 013 | 2026-05-19    | RunPod RTX 4090 | Complete Blood Count not flagged               |
| 014 | 2026-05-19    | RunPod RTX 4090 | Comprehensive Metabolic Panel not flagged      |
| 015 | 2026-05-19    | RunPod RTX 4090 | "confidential" lowercase not flagged           |
| 016 | 2026-05-19    | RunPod RTX 4090 | "The healthcare" not flagged as HOSPITAL       |
| 017 | 2026-05-19    | RunPod RTX 4090 | "HIPAA compliance guidelines" not flagged      |


---

## All Config Changes Applied


| Entity                           | Field                | Before                   | After                                       | Fixes     |
| -------------------------------- | -------------------- | ------------------------ | ------------------------------------------- | --------- |
| `person_name`                    | confidence_threshold | 0.45                     | 0.62                                        | #009      |
| `person_name`                    | description          | —                        | added "lab test names" to DO NOT flag list  | #009      |
| `city_name`                      | confidence_threshold | 0.40                     | 0.72                                        | #006–008  |
| `city_name`                      | pattern              | case-insensitive `[A-Z]` | `(?-i:[A-Z])` case-sensitive                | #006–008  |
| `physician_name`                 | confidence_threshold | 0.45                     | 0.75                                        | #005      |
| `physician_name`                 | pattern              | case-insensitive `[A-Z]` | `(?-i:[A-Z])` case-sensitive                | #010 #011 |
| `license_plate`                  | confidence_threshold | *(global 0.85)*          | 0.92                                        | #003      |
| `hospital_name`                  | priority             | 2                        | 1                                           | #005      |
| `hospital_name`                  | pattern prefix count | `{1,5}`                  | `{2,5}`                                     | #016      |
| `email_address`                  | confidence_threshold | *(global 0.85)*          | 0.92                                        | #002      |
| `email_address`                  | patterns             | none                     | added `@` format regex                      | #002      |
| `health_plan_beneficiary_number` | patterns             | no state-prefix          | added `[A-Z]{2}-\d{5,10}`                   | #001      |
| `billing_number`                 | patterns             | digits only              | added `billing account [A-Z]-\d`            | #012      |
| `clinical_date`                  | patterns             | no "recorded as"         | added `recorded|noted|documented as`        | #003      |
| `clinical_date`                  | patterns             | no "scheduled for"       | added `scheduled|planned|set|confirmed for` | #004      |
| `confidential`                   | confidence_threshold | *(global 0.85)*          | 0.95                                        | #017      |
| `confidential`                   | pattern              | case-insensitive         | `(?-i:...)` uppercase only                  | #015      |


## All Code Changes — `pipeline/span_merger.py`

`_validate_and_clean()` now runs in both `merge()` and `merge_all()`:


| Check                                                           | Fixes          |
| --------------------------------------------------------------- | -------------- |
| Reject `physician_name` NER spans without "Dr."/"Doctor" prefix | #005           |
| Reject `email_address` spans with no `@`                        | #002           |
| Reject `license_plate` spans matching date regex                | #003           |
| Trim trailing lowercase words from `physician_name` spans       | #010 #011      |
| Reject `person_name` spans in `_PERSON_NAME_BLOCKLIST`          | #009 #013 #014 |


`_PERSON_NAME_BLOCKLIST`:
`lipid profile, complete blood count, comprehensive metabolic panel, cbc, cmp, urinalysis, hba1c, blood pressure, heart rate, blood glucose, oxygen saturation, metabolic panel`

---

## How to Add a New Issue

1. Run text through `/mask` endpoint
2. Find wrong detection in the results panel
3. Add row with status `OPEN`
4. Investigate: config threshold / pattern gap / model confusion
5. Apply fix → `FIXED` or `NEEDS RETRAIN`
6. After RunPod verify → Resolved Log with date

