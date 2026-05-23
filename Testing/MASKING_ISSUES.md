# Masking Issues Tracker

Track all false positives, misclassifications, and missed detections.
Review daily — mark RESOLVED after fix is confirmed working on RunPod.
When Retrain Queue hits 5+ issues, trigger `./start.sh --samples 3000` on H100.

**Status:**

- `OPEN` — found, not yet fixed
- `FIXED` — config/code fix applied, pending RunPod verification
- `NEEDS RETRAIN` — only the model can fix this, queued for next training run
- `RESOLVED` — confirmed working on RunPod after deploy or retrain

---

## Issue Log — OPEN Only

> All previously tracked issues (#001–#030) are RESOLVED. See Resolved Log below.
> Add new issues here as they are discovered.

| #   | Date | Input Text | Detected As | Should Be | Status | Fix |
| --- | ---- | ---------- | ----------- | --------- | ------ | --- |
| —   | —    | *(no open issues)* | — | — | — | — |

---

## Retrain Queue

> These issues cannot be fixed by patterns/code alone — the model learns wrong associations at training time.
> **Trigger retraining when 6+ issues are queued here**, or validation drops below 95%.
> Source: Excel sheet analysis (Sheet1 / Sheet4 — 579 + 923 failures before 2026-05-23 fixes).

| #   | Issue                                             | Count  | Why Retrain Needed                                                                                                                      |
| --- | ------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| R01 | LICENSE PLATE label on CASE NUMBER / CARD NUMBER  | ~335   | Model conflates structured ID formats — needs 500+ disambiguation examples (case number near license plate, card number in same doc)    |
| R02 | PHONE NUMBER label on NPI / ROUTING digits        | ~198   | Bare 10/9-digit numbers look identical without keyword — pattern layer helps specific cases but model-level confusion remains at scale   |
| R03 | INSURANCE COMPANY label on HOSPITAL names         | ~88    | Model conflates org-type entities — needs 200+ co-occurrence examples with both in same document                                        |
| R04 | Second-occurrence entities missed                 | ~220   | Training data had single entity per doc — model never learned to scan exhaustively; needs 2–3 instances of same entity per example      |
| R05 | Bare city names missed (no state abbreviation)    | ~159   | Cities like Garland, Honolulu, Reno, Seattle without "ST" suffix not caught — needs city-only NER training examples                     |
| R06 | Standalone years (2024, 2018) without keywords    | ~160   | Bare 4-digit years in employment/education context missed — needs training examples with year-only dates                                |
| R07 | Company / org names missed (NER coverage)         | ~32    | Nexus Technologies Corp., Pacific Rim Enterprises, Goldman Sachs etc. — needs org-name NER training examples                           |
| R08 | Untrained entity types (Sheet 2 catalog)          | ~300+  | Model has zero NER coverage for: Cardiac implant serials, Face/Voice biometrics, Employment History, Student Records, Performance Evals |

---

## Resolved Log

| #   | Date Resolved | Confirmed On         | Input / Issue                                          | How Resolved                                                                                   |
| --- | ------------- | -------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| 001 | 2026-05-19    | RunPod RTX 4090      | `CA-8821901` → PHONE, should be INSURANCE ID           | Added `[A-Z]{2}-\d{5,10}` pattern to `health_plan_beneficiary_number`                         |
| 002 | 2026-05-19    | RunPod RTX 4090      | `Daniel Wilson` → EMAIL, should be PERSON              | `@` guard in `_validate_and_clean()` + email threshold → 0.92                                 |
| 003 | 2026-05-19    | RunPod RTX 4090      | `04/03/2025` → LICENSE PLATE, should be DATE           | Date regex guard in `_validate_and_clean()` + `recorded as` pattern added                     |
| 004 | 2026-05-19    | RunPod RTX 4090      | `07/15/2025` MISSED, should be DATE                    | Added `scheduled|planned|set|confirmed for` pattern to `clinical_date`                        |
| 005 | 2026-05-19    | RunPod RTX 4090      | `Riverside Community Health Center` → DOCTOR           | GLiNER `physician_name` spans without Dr./Doctor prefix rejected in `_validate_and_clean()`   |
| 006 | 2026-05-19    | RunPod RTX 4090      | `the patient` → LOCATION, not PII                      | city_name pattern made case-sensitive `(?-i:[A-Z])` — lowercase no longer matches             |
| 007 | 2026-05-19    | RunPod RTX 4090      | `diabetic neuropathy` → LOCATION, not PII              | Same as #006                                                                                   |
| 008 | 2026-05-19    | RunPod RTX 4090      | `current medication therapy` → LOCATION, not PII       | Same as #006                                                                                   |
| 009 | 2026-05-19    | RunPod RTX 4090      | `Lipid Profile` → PERSON, not PII                      | Added to `_PERSON_NAME_BLOCKLIST`                                                              |
| 010 | 2026-05-19    | RunPod RTX 4090      | `Dr. Rebecca Moore supervised` → DOCTOR (extra word)   | physician_name pattern case-sensitive — lowercase trailing words no longer captured           |
| 011 | 2026-05-19    | RunPod RTX 4090      | `Dr. Samuel Peterson and` → DOCTOR (extra word)        | Same as #010                                                                                   |
| 012 | 2026-05-19    | RunPod RTX 4090      | `BL-9910274` → BILLING NO (GLiNER only, no pattern)    | Added `billing account [A-Z]-\d` pattern — pattern layer now covers it                        |
| 013 | 2026-05-19    | RunPod RTX 4090      | `Complete Blood Count` → PERSON, not PII               | Added to `_PERSON_NAME_BLOCKLIST`                                                              |
| 014 | 2026-05-19    | RunPod RTX 4090      | `Comprehensive Metabolic Panel` → PERSON, not PII      | Same as #013                                                                                   |
| 015 | 2026-05-19    | RunPod RTX 4090      | `confidential` (lowercase) → CONFIDENTIAL, not PII     | Pattern changed to `(?-i:...)` — uppercase only                                               |
| 016 | 2026-05-19    | RunPod RTX 4090      | `The healthcare team` → HOSPITAL, not PII              | hospital_name pattern prefix `{1,5}` → `{2,5}` — requires 2+ capitalized words               |
| 017 | 2026-05-19    | RunPod RTX 4090      | `HIPAA compliance guidelines` → CONFIDENTIAL, not PII  | `confidential` confidence_threshold → 0.95                                                    |
| 018 | 2026-05-23    | RunPod RTX 4090      | `Provider NPI 1234567890` → PHONE, should be NPI       | NPI pattern captures full `NPI XXXXXXXXXX` token — longer span beats bare phone digits        |
| 019 | 2026-05-23    | RunPod RTX 4090      | `SPL-7392` → MED LICENSE, should be STOLEN PLATE       | Removed `SPL` prefix from `medical_license_number` prefix list                                |
| 020 | 2026-05-23    | RunPod RTX 4090      | `Recorded 2023-07-14 09:32:00 UTC` → DOB, should be CLINICAL DATE | Removed timestamp pattern from `date_of_birth`                                    |
| 021 | 2026-05-23    | RunPod RTX 4090      | `Date 2026-05-21` → DOB, should be CLINICAL DATE       | Removed generic `Date:` keyword patterns from `date_of_birth`                                 |
| 022 | 2026-05-23    | RunPod RTX 4090      | `PROCEDURE DATE 04/15/2025` → DOB, should be CLINICAL DATE | Same as #021                                                                              |
| 023 | 2026-05-23    | RunPod RTX 4090      | `John Smith and Mary Johnson are suspects` → WANTED PERSON | Alpha-only NER rejection added for law-enf record entities in `_validate_and_clean()`    |
| 024 | 2026-05-23    | RunPod RTX 4090      | `John Smith and Mary Johnson are patients` → MRN       | Extended alpha-only rejection to MRN, billing, insurance_policy entities                      |
| 025 | 2026-05-23    | RunPod RTX 4090      | `Electronic signature dated 03/13/2024` — date MISSED  | Added `dated/signed MM/DD/YYYY` pattern to `clinical_date`                                    |
| 026 | 2026-05-23    | RunPod RTX 4090      | `St. Louis IN` — city MISSED                           | All city_name patterns updated to allow `St.` prefix                                          |
| 027 | 2026-05-23    | RunPod RTX 4090      | `St. Mary's Hospital` — hospital MISSED                | hospital_name pattern allows `St.` prefix and apostrophes in names                            |
| 028 | 2026-05-23    | RunPod RTX 4090      | `Westlake Oncology Center` — hospital MISSED           | Added Oncology Center / Cancer Center / Surgery Center as recognized hospital suffixes        |
| 029 | 2026-05-23    | RunPod RTX 4090      | `Admission Date: March 2, 2025` → DOB, should be CLINICAL DATE | clinical_date pattern now captures full keyword phrase — longer span wins tiebreaker |
| 030 | 2026-05-23    | RunPod RTX 4090      | `Discharge Date / Surgery Date: Month DD, YYYY` → DOB | Same as #029                                                                                   |

---

## All Config Changes — `entities_config.yaml`

| Entity                           | Field                | Before                             | After                                                        | Fixes         |
| -------------------------------- | -------------------- | ---------------------------------- | ------------------------------------------------------------ | ------------- |
| `person_name`                    | confidence_threshold | 0.45                               | 0.62                                                         | #009          |
| `city_name`                      | confidence_threshold | 0.40                               | 0.72                                                         | #006–008      |
| `city_name`                      | pattern              | `[A-Z][a-zA-Z]+`                   | `(?-i:[A-Z])` case-sensitive + `(?:St\.\s+)?` prefix        | #006–008 #026 |
| `physician_name`                 | confidence_threshold | 0.45                               | 0.75                                                         | #005          |
| `physician_name`                 | pattern              | case-insensitive                   | `(?-i:[A-Z])` case-sensitive                                 | #010 #011     |
| `license_plate`                  | confidence_threshold | *(global 0.85)*                    | 0.97                                                         | #003          |
| `hospital_name`                  | priority             | 2                                  | 1                                                            | #005          |
| `hospital_name`                  | pattern              | `{1,5}` words, no St./apostrophe   | `{1,5}` + `St.` prefix + apostrophes + Oncology/Cancer/Surgery suffixes | #016 #027 #028 |
| `email_address`                  | confidence_threshold | *(global 0.85)*                    | 0.92                                                         | #002          |
| `health_plan_beneficiary_number` | patterns             | no state-prefix                    | added `[A-Z]{2}-\d{5,10}`                                   | #001          |
| `billing_number`                 | patterns             | digits only                        | added `billing account [A-Z]-\d`                             | #012          |
| `npi_number`                     | pattern              | captures digits only               | captures full `NPI XXXXXXXXXX` token for longer span         | #018          |
| `medical_license_number`         | pattern prefix list  | included `SPL`                     | removed `SPL` (Stolen Plate, not medical)                    | #019          |
| `date_of_birth`                  | patterns             | included timestamp + generic Date: | removed — only DOB/born-on keywords trigger date_of_birth    | #020 #021 #022|
| `clinical_date`                  | patterns             | no `dated/signed` keyword          | added `dated/signed MM/DD/YYYY` for signature dates          | #025          |
| `clinical_date`                  | patterns             | Admission/Discharge captured date only | captures full keyword phrase — longer span beats DOB     | #029 #030     |
| `clinical_date`                  | patterns             | no "recorded as"                   | added `recorded|noted|documented as`                         | #003          |
| `clinical_date`                  | patterns             | no "scheduled for"                 | added `scheduled|planned|set|confirmed for`                  | #004          |
| `confidential`                   | confidence_threshold | *(global 0.85)*                    | 0.95                                                         | #017          |
| `confidential`                   | pattern              | case-insensitive                   | `(?-i:...)` uppercase only                                   | #015          |

---

## All Code Changes — `pipeline/span_merger.py`

### `_validate_and_clean()` guards

| Check                                                                         | Fixes              |
| ----------------------------------------------------------------------------- | ------------------ |
| Reject `physician_name` NER spans without "Dr."/"Doctor" prefix               | #005               |
| Reject `email_address` spans with no `@`                                      | #002               |
| Reject `license_plate` spans matching bare date regex                         | #003               |
| Trim trailing lowercase words from `physician_name` spans                     | #010 #011          |
| Reject `person_name` spans in `_PERSON_NAME_BLOCKLIST`                        | #009 #013 #014     |
| Reject alpha-only NER spans for `_ALPHA_REJECT_ENTITY_IDS`                    | #023 #024          |

### `_pick_winner()` priority fix (2026-05-23)

Pattern layer (`source=pattern`, order=0) now checked **before** confidence — prevents high-confidence NER labels overriding precision-crafted patterns.

### Constants

**`_ALPHA_REJECT_ENTITY_IDS`** — alpha-only NER detections rejected for these entity types:
```
wanted_person_report, missing_person_report, gang_terrorist_member, foreign_fugitives,
identity_theft_victims, sex_offender_report, supervised_release, probation_record,
parole_record, medical_record_number, billing_number, health_plan_beneficiary_number,
insurance_policy_number
```

**`_PERSON_NAME_BLOCKLIST`**:
```
lipid profile, complete blood count, comprehensive metabolic panel, cbc, cmp,
urinalysis, hba1c, blood pressure, heart rate, blood glucose, oxygen saturation,
metabolic panel
```

---

## How to Add a New Issue

1. Run text through `/mask` endpoint
2. Find wrong detection in the results panel
3. Add row to Issue Log with status `OPEN`
4. Investigate: config threshold / pattern gap / model confusion
5. Apply fix → status `FIXED` or `NEEDS RETRAIN`
6. After RunPod verification → move to Resolved Log with date
