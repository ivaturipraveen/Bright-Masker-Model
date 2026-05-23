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


| #   | Date       | Input Text                                        | Detected As              | Should Be             | Status   | Fix                                                                                               |
| --- | ---------- | ------------------------------------------------- | ------------------------ | --------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| 001 | 2026-05-19 | `CA-8821901`                                      | PHONE NUMBER             | INSURANCE ID          | RESOLVED | Added `[A-Z]{2}-\d{5,10}` pattern to `health_plan_beneficiary_number`                             |
| 002 | 2026-05-19 | `Daniel Wilson`                                   | EMAIL                    | PERSON                | RESOLVED | `@` guard in `span_merger._validate_and_clean()` + email threshold → 0.92                         |
| 003 | 2026-05-19 | `04/03/2025`                                      | LICENSE PLATE            | DATE                  | RESOLVED | Date regex guard in `span_merger._validate_and_clean()` + `recorded as` pattern added             |
| 004 | 2026-05-19 | `07/15/2025`                                      | MISSED                   | DATE                  | RESOLVED | Added `scheduled|planned|set|confirmed for` pattern to `clinical_date`                            |
| 005 | 2026-05-19 | `Riverside Community Health Center`               | DOCTOR                   | HOSPITAL              | RESOLVED | GLiNER physician_name spans without "Dr."/"Doctor" prefix now rejected in `_validate_and_clean()` |
| 006 | 2026-05-19 | `the patient`                                     | LOCATION                 | Not PII               | RESOLVED | city_name pattern made case-sensitive `(?-i:[A-Z])` — lowercase phrases no longer match           |
| 007 | 2026-05-19 | `diabetic neuropathy`                             | LOCATION                 | Not PII               | RESOLVED | Same as #006                                                                                      |
| 008 | 2026-05-19 | `current medication therapy`                      | LOCATION                 | Not PII               | RESOLVED | Same as #006                                                                                      |
| 009 | 2026-05-19 | `Lipid Profile`                                   | PERSON                   | Not PII               | RESOLVED | Added to `_PERSON_NAME_BLOCKLIST` in `span_merger._validate_and_clean()`                          |
| 010 | 2026-05-19 | `Dr. Rebecca Moore supervised`                    | DOCTOR (extra word)      | `Dr. Rebecca Moore`   | RESOLVED | physician_name pattern made case-sensitive — lowercase trailing words no longer captured          |
| 011 | 2026-05-19 | `Dr. Samuel Peterson and`                         | DOCTOR (extra word)      | `Dr. Samuel Peterson` | RESOLVED | Same as #010                                                                                      |
| 012 | 2026-05-19 | `BL-9910274`                                      | BILLING NO (GLiNER only) | BILLING NO (Pattern)  | RESOLVED | Added `billing account [A-Z]-\d` pattern — pattern layer now covers it                            |
| 013 | 2026-05-19 | `Complete Blood Count`                            | Potential PERSON         | Not PII               | RESOLVED | Added to `_PERSON_NAME_BLOCKLIST`                                                                 |
| 014 | 2026-05-19 | `Comprehensive Metabolic Panel`                   | Potential PERSON         | Not PII               | RESOLVED | Same as #013                                                                                      |
| 015 | 2026-05-19 | `confidential` (lowercase, in sentence)           | CONFIDENTIAL             | Not PII               | RESOLVED | Pattern changed to `(?-i:...)` — uppercase only                                                   |
| 016 | 2026-05-19 | `The healthcare team`                             | HOSPITAL                 | Not PII               | RESOLVED | hospital_name pattern prefix `{1,5}` → `{2,5}` — requires 2+ capitalized words                    |
| 017 | 2026-05-19 | `HIPAA compliance guidelines`                     | CONFIDENTIAL             | Not PII               | RESOLVED | `confidential` confidence_threshold → 0.95                                                        |
| 018 | 2026-05-23 | `Provider NPI 1234567890`                         | PHONE NUMBER             | NPI                   | RESOLVED | NPI pattern captures full `NPI XXXXXXXXXX` token — longer span beats bare phone digits            |
| 019 | 2026-05-23 | `SPL-7392`                                        | MED LICENSE              | STOLEN PLATE          | RESOLVED | Removed `SPL` prefix from `medical_license_number` prefix list                                    |
| 020 | 2026-05-23 | `Recorded 2023-07-14 09:32:00 UTC`                | DATE OF BIRTH            | CLINICAL DATE         | RESOLVED | Removed timestamp pattern from `date_of_birth` — timestamps are clinical, not DOB                 |
| 021 | 2026-05-23 | `Date 2026-05-21`                                 | DATE OF BIRTH            | CLINICAL DATE         | RESOLVED | Removed generic `Date:` keyword patterns from `date_of_birth` — only DOB/born keywords trigger it |
| 022 | 2026-05-23 | `PROCEDURE DATE 04/15/2025`                       | DATE OF BIRTH            | CLINICAL DATE         | RESOLVED | Same as #021                                                                                      |
| 023 | 2026-05-23 | `John Smith and Mary Johnson are suspects`        | WANTED PERSON            | PERSON                | RESOLVED | Alpha-only NER rejection added for law-enf record entities in `_validate_and_clean()`             |
| 024 | 2026-05-23 | `John Smith and Mary Johnson are patients`        | MRN                      | PERSON                | RESOLVED | Extended alpha-only rejection to `medical_record_number`, `billing_number`, `insurance_policy_number` |


---

## Retrain Queue

> These issues cannot be fixed by patterns/code alone — the model learns wrong associations at training time.
> **Trigger retraining** when 5+ issues are queued here, or validation drops below 95%.

| #   | Issue                                              | Count  | Why Retrain Needed                                                                                        |
| --- | -------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| R01 | LICENSE PLATE label on CASE NUMBER / CARD NUMBER   | ~335   | Model learned license plate format overlaps with case numbers and card formats — needs disambiguation examples |
| R02 | PHONE NUMBER label on NPI / ROUTING NUMBER digits  | ~198   | Bare 10-digit NPI and 9-digit routing indistinguishable from phone without keyword — pattern layer helps but model-level confusion remains at scale |
| R03 | INSURANCE COMPANY label on HOSPITAL names          | ~88    | Model conflates org-type entities — needs 200+ co-occurrence examples with both types in same document    |
| R04 | Second-occurrence entities missed                  | ~220   | Training data had single-occurrence entities per doc — model never learned to scan exhaustively           |
| R05 | City + State missed (no address context)           | ~283   | City/state underrepresented as PII in training data — needs city/state in every address training example  |
| R06 | Untrained entity types (Sheet 2 catalog)           | ~300+  | Model has zero coverage for: Visit ID, Lab Sample ID, Radiology Study ID, Cardiac implant serials, Face/Voice biometrics, Medicare/Medicaid IDs, Telemedicine Session IDs, Employment History, Student Records, Performance Evaluations |

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
| 018 | 2026-05-23    | Local (pattern) | "Provider NPI 1234567890" → NPI, not PHONE     |
| 019 | 2026-05-23    | Local (pattern) | "SPL-7392" → STOLEN PLATE, not MED LICENSE     |
| 020 | 2026-05-23    | Local (pattern) | "Recorded 2023-07-14 09:32:00 UTC" → CLINICAL DATE |
| 021 | 2026-05-23    | Local (pattern) | "Date 2026-05-21" → CLINICAL DATE, not DOB     |
| 022 | 2026-05-23    | Local (pattern) | "PROCEDURE DATE 04/15/2025" → CLINICAL DATE    |
| 023 | 2026-05-23    | Local (code)    | Person names in "suspects" context → PERSON    |
| 024 | 2026-05-23    | Local (code)    | Person names in "patients" context → PERSON    |


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


| Check                                                                        | Fixes              |
| ---------------------------------------------------------------------------- | ------------------ |
| Reject `physician_name` NER spans without "Dr."/"Doctor" prefix              | #005               |
| Reject `email_address` spans with no `@`                                     | #002               |
| Reject `license_plate` spans matching date regex                             | #003               |
| Trim trailing lowercase words from `physician_name` spans                    | #010 #011          |
| Reject `person_name` spans in `_PERSON_NAME_BLOCKLIST`                       | #009 #013 #014     |
| Reject alpha-only NER spans for `_ALPHA_REJECT_ENTITY_IDS` (record/ID types) | #023 #024          |

`_ALPHA_REJECT_ENTITY_IDS` — NER detections of purely alphabetic text rejected for:
`wanted_person_report, missing_person_report, gang_terrorist_member, foreign_fugitives, identity_theft_victims, sex_offender_report, supervised_release, probation_record, parole_record, medical_record_number, billing_number, health_plan_beneficiary_number, insurance_policy_number`

`_PERSON_NAME_BLOCKLIST`:
`lipid profile, complete blood count, comprehensive metabolic panel, cbc, cmp, urinalysis, hba1c, blood pressure, heart rate, blood glucose, oxygen saturation, metabolic panel`

`_pick_winner()` source order fix (2026-05-23):
Pattern layer (`source=pattern`) now always beats NER (`source=ner`) before confidence comparison — prevents high-confidence NER labels overriding precision-crafted patterns.

---

## How to Add a New Issue

1. Run text through `/mask` endpoint
2. Find wrong detection in the results panel
3. Add row with status `OPEN`
4. Investigate: config threshold / pattern gap / model confusion
5. Apply fix → `FIXED` or `NEEDS RETRAIN`
6. After RunPod verify → Resolved Log with date

