
"""
Synthetic PII training data generator for fine-tuning GLiNER.
Production-quality: 20-25 templates per entity, hard negatives, realistic
multi-entity documents across 8 document types.

Reads entities_config.yaml — single source of truth for entity labels.
Output: train/data/pii_train.json

Run:
    python train/generate_data.py [--samples N] [--out PATH] [--seed S]

GLiNER training format:
    {"tokenized_text": ["tok1", ...], "ner": [[start, end, "label"], ...]}
    Indices are 0-based, inclusive, token-level (whitespace split).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

import yaml

try:
    from faker import Faker
    fake = Faker("en_US")
except ImportError:
    print("ERROR: pip install faker", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent.parent
YAML_PATH = ROOT / "entities_config.yaml"
DEFAULT_OUT = ROOT / "train" / "data" / "pii_train.json"


# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

def _phone() -> str:
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"

def _phone2() -> str:
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def _fax() -> str:
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"

def _ssn() -> str:
    return f"{random.randint(100,799):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"

def _credit_card() -> str:
    return f"{random.randint(4000,4999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def _cvv() -> str:
    return str(random.randint(100, 999))

def _card_exp() -> str:
    return f"{random.randint(1,12):02d}/{random.randint(25,30):02d}"

def _date_mmddyyyy() -> str:
    d = fake.date_of_birth(minimum_age=1, maximum_age=90)
    return d.strftime(random.choice(["%m/%d/%Y", "%m-%d-%Y"]))

def _date_clinical() -> str:
    d = fake.date_between(start_date="-5y", end_date="today")
    return d.strftime(random.choice(["%m/%d/%Y", "%m-%d-%Y"]))

def _ip() -> str:
    return fake.ipv4_private()

def _mac() -> str:
    return ":".join(f"{random.randint(0,255):02X}" for _ in range(6))

def _url() -> str:
    return f"https://portal.hospital.org/patients/{random.randint(10000,99999)}"

def _mrn_value() -> str:
    return str(random.randint(1000000, 9999999))

def _npi() -> str:
    return str(random.randint(1000000000, 9999999999))

def _dea() -> str:
    letters = "ABCDEFGHJKLMNPRSTUX"
    return f"{random.choice(letters)}{random.choice(letters)}{random.randint(1000000,9999999)}"

def _iban() -> str:
    return f"GB{random.randint(10,99)}NWBK{random.randint(10000000,99999999)}{random.randint(10000000,99999999)}"

def _swift() -> str:
    banks = ["CHAS", "BNPA", "DEUT", "HSBC", "BARC", "CITI", "BOFA"]
    countries = ["US", "FR", "DE", "GB", "JP"]
    locs = ["33", "PP", "2X", "FF"]
    return f"{random.choice(banks)}{random.choice(countries)}{random.choice(locs)}"

def _routing() -> str:
    return str(random.randint(100000000, 999999999))

def _bank_account() -> str:
    return str(random.randint(10000000, 9999999999))

def _vin() -> str:
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(17))

def _license_plate() -> str:
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}-{random.randint(1000,9999)}"

def _api_key() -> str:
    return "sk-" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=32))

def _password() -> str:
    special = "!@#$%^&*"
    return (random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + random.choice("abcdefghijklmnopqrstuvwxyz")
            + str(random.randint(10,99))
            + random.choice(special)
            + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=6)))

def _state_abbr() -> str:
    return random.choice(["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI",
                          "ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI",
                          "MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC",
                          "ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
                          "VT","VA","WA","WV","WI","WY"])

def _zipcode() -> str:
    return str(random.randint(10000,99999))

def _gps() -> str:
    lat = round(random.uniform(24.0, 49.0), 6)
    lon = round(random.uniform(-124.0, -66.0), 6)
    return f"{lat}, {lon}"

def _first_last() -> str:
    return f"{fake.first_name()} {fake.last_name()}"

def _street_address() -> str:
    streets = ["Main St", "Oak Ave", "Elm Blvd", "River Rd", "Park Lane",
               "Maple Drive", "Cedar Court", "Washington Blvd", "Pine Way",
               "Highland Ave", "Sunset Terrace", "Valley Rd", "Green St"]
    return f"{random.randint(1,9999)} {random.choice(streets)}"

def _city() -> str:
    return fake.city()

def _email() -> str:
    return fake.email()

def _company() -> str:
    return fake.company()

def _hospital() -> str:
    adj = ["Regional","Community","General","Central","Pacific","Northern",
           "Valley","Riverside","University","Metropolitan","Lakeside"]
    noun = ["Medical Center","Hospital","Health System","Memorial Hospital","Healthcare"]
    return f"{random.choice(adj)} {random.choice(noun)}"

def _bank_name() -> str:
    return random.choice(["First National Bank","City Savings Bank","Heritage Credit Union",
                           "Chase","Wells Fargo","Bank of America","Citibank","TD Bank",
                           "Union Trust Bank","Capital One","US Bank"])

def _insurance_co() -> str:
    return random.choice(["United Shield Insurance","Blue Cross Blue Shield","Aetna","Cigna",
                           "Humana","Anthem","MetLife Insurance","State Farm","Allstate",
                           "Travelers Insurance","Kaiser Permanente"])

def _case_number() -> str:
    return f"CR-{random.randint(2020,2024)}-{random.randint(1000,99999):05d}"

def _amount() -> str:
    return f"${random.randint(100,99999):,}.{random.randint(0,99):02d}"

def _employee_id() -> str:
    return f"EMP-{random.randint(10000,99999)}"

def _student_id() -> str:
    return f"STU-{random.randint(10000,99999)}"

def _tax_id() -> str:
    return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"

def _flight_number() -> str:
    return f"{random.choice(['AA','UA','DL','SW','BA','LH','EK'])} {random.randint(100,9999)}"

def _booking_ref() -> str:
    return "BK-" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))

def _claim_number() -> str:
    return f"CLM-{random.randint(100000,999999)}"

def _policy_number() -> str:
    return f"POL-{random.randint(10000000,99999999)}"

def _member_id() -> str:
    return f"MBR{random.randint(100000,9999999)}"

def _transaction_id() -> str:
    d = fake.date_between(start_date="-1y", end_date="today").strftime("%Y%m%d")
    return f"TXN-{d}-{random.randint(10000,99999):05d}"

def _merchant_id() -> str:
    return f"MID-{random.randint(10000000,99999999)}"

def _university() -> str:
    return random.choice(["Northern State University","Pacific Coast College",
                           "Riverside Institute of Technology","Midwest University",
                           "Southern Medical School","Eastern State College",
                           "Valley Community College","Lakewood University",
                           "Metropolitan School of Medicine"])

def _law_firm() -> str:
    return random.choice(["Smith & Associates","Johnson & Williams LLP",
                           "Parker & Davis Law Group","Mitchell & Clark Attorneys at Law",
                           "Thompson & Reed LLP","Harrison Law Office","Grant & Associates LLP"])

def _court_name() -> str:
    return random.choice(["California Superior Court","US District Court","Family Court",
                           "Cook County Circuit Court","New York Supreme Court",
                           "Texas District Court","Florida Circuit Court"])

def _hotel_name() -> str:
    return random.choice(["Grand Royal Hotel","Marriott Downtown","Hilton Garden Inn",
                           "Holiday Inn Express","Westin Conference Center","Hyatt Regency",
                           "Sheraton Grand Hotel"])

def _inmate_id() -> str:
    return str(random.randint(10000,999999))

def _warrant_num() -> str:
    return f"W{random.randint(2020,2024)}{random.randint(10000,99999)}"

def _incident_num() -> str:
    return f"IR#{random.randint(2020,2024)}{random.randint(10000,99999)}"

def _bart_emp_id() -> str:
    return str(random.randint(10000,999999))

def _passport_num() -> str:
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10000000,99999999)}"

def _drivers_license() -> str:
    return f"D{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def _state_id_num() -> str:
    return f"ID{random.randint(100000,9999999)}"

def _med_license() -> str:
    return f"ML{random.randint(100000,9999999)}"

def _health_plan_id() -> str:
    prefixes = ["UHC","BCBS","CIGNA","AETNA","HUMANA","KAISER"]
    return f"{random.choice(prefixes)}-{random.randint(1000000,9999999)}"

def _card_track() -> str:
    pan = f"{random.randint(4000,4999)}{random.randint(100000000,999999999)}{random.randint(1000,9999)}"
    name = f"{fake.last_name()}/{fake.first_name()}"
    exp = f"{random.randint(25,30):02d}{random.randint(1,12):02d}"
    return f"%B{pan}^{name}^{exp}101?"

def _pin_block() -> str:
    return "".join(random.choices("0123456789ABCDEF", k=16))

def _cryptogram() -> str:
    return "".join(random.choices("0123456789ABCDEF", k=8))

def _imei() -> str:
    return "".join(str(random.randint(0,9)) for _ in range(15))

def _vehicle_reg() -> str:
    return f"{_state_abbr()}-{random.randint(10,99)}-{_state_abbr()}-{random.randint(100,9999)}"

def _billing_num() -> str:
    return str(random.randint(1000000,99999999))

def _dept_color() -> str:
    return random.choice(["Hispanic","White","Black","Asian","Native American",
                           "Pacific Islander","Middle Eastern","Multiracial"])

def _religion_val() -> str:
    return random.choice(["Christian","Muslim","Jewish","Hindu","Buddhist",
                           "Catholic","Protestant","Sikh","Atheist"])

def _physical_desc() -> str:
    height = f"{random.randint(5,6)}'{random.randint(0,11)}\""
    weight = f"{random.randint(110,250)} lbs"
    hair = random.choice(["brown hair","black hair","blonde hair","red hair","gray hair"])
    return f"{height}, {weight}, {hair}"

def _medication() -> str:
    return random.choice(["Metformin 500mg","Lisinopril 10mg","Atorvastatin 40mg",
                           "Amlodipine 5mg","Omeprazole 20mg","Albuterol 90mcg",
                           "Prednisone 20mg","Sertraline 50mg","Levothyroxine 100mcg",
                           "Gabapentin 300mg","Amoxicillin 500mg","Metoprolol 25mg"])

def _job_title() -> str:
    return fake.job()

def _employer() -> str:
    return fake.company()

def _username() -> str:
    return fake.user_name()


# ---------------------------------------------------------------------------
# Entity template definitions — 20-25 templates per entity
# ---------------------------------------------------------------------------

def _physician_name() -> str:
    titles = ["Dr.", "Doctor"]
    title = random.choice(titles)
    return f"{title} {fake.first_name()} {fake.last_name()}"

def _card_holder() -> str:
    return f"{fake.first_name().upper()} {fake.last_name().upper()}"


ENTITY_DEFS: dict[str, dict] = {

    "physician_name": {
        "generator": _physician_name,
        "templates": [
            "Referring physician: {value}",
            "Primary care provider: {value}",
            "Physician: {value}",
            "Ordered by {value}",
            "Prescribing physician: {value}",
            "Attending physician: {value}",
            "The patient was seen by {value}.",
            "Surgeon: {value}",
            "Treating physician: {value}",
            "Specialty consult with {value}.",
            "Provider: {value}",
            "Signed by {value}, MD",
            "The report was authored by {value}.",
            "Follow-up scheduled with {value}.",
            "Lab ordered by {value}.",
            "Prescription issued by {value}.",
            "Consult note from {value}.",
            "Authorized by {value}",
            "Licensed provider: {value}",
            "Clinician: {value}",
        ],
    },

    "card_holder_name": {
        "generator": _card_holder,
        "templates": [
            "Cardholder: {value}",
            "Card holder name: {value}",
            "Name on card: {value}",
            "Account holder: {value}",
            "The card is issued to {value}.",
            "Printed name: {value}",
            "Card name: {value}",
            "Registered to: {value}",
            "Card owner: {value}",
            "Embossed name: {value}",
        ],
    },

    "person_name": {
        "generator": _first_last,
        "templates": [
            "Patient {value} was admitted to the hospital.",
            "The report was filed on behalf of {value}.",
            "Contact {value} for follow-up appointments.",
            "Name: {value}",
            "Referring physician: {value}",
            "{value} signed the consent form.",
            "Emergency contact listed as {value}.",
            "Prescription written for {value}.",
            "The claimant {value} submitted the form.",
            "Insured: {value}",
            "Primary beneficiary: {value}",
            "{value} was present at the time of incident.",
            "Witness statement provided by {value}.",
            "Account holder: {value}",
            "The patient {value} was discharged on Tuesday.",
            "Dr. appointment for {value} confirmed.",
            "Records released to {value} upon authorization.",
            "The suspect was identified as {value}.",
            "Employee {value} clocked in at 8 AM.",
            "Traveler: {value}",
            "Student: {value}",
            "Defendant: {value}",
            "Claimant name: {value}",
            "Guardian: {value}",
        ],
    },

    "street_address": {
        "generator": _street_address,
        "templates": [
            "The patient resides at {value}.",
            "Mailing address: {value}",
            "Home address: {value}",
            "Address on file: {value}",
            "Please send correspondence to {value}.",
            "Incident occurred at {value}.",
            "Delivery address: {value}",
            "The property located at {value} was inspected.",
            "Service address: {value}",
            "Billing address: {value}",
            "Address: {value}",
            "Residence: {value}",
            "Last known address: {value}",
            "The subject lives at {value}.",
            "Mail was forwarded from {value}.",
            "Search warrant executed at {value}.",
            "Property at {value} was seized.",
            "Report was generated at location {value}.",
            "Parcel delivered to {value}.",
            "Registered address: {value}",
        ],
    },

    "city_name": {
        "generator": _city,
        "templates": [
            "The patient lives in {value}.",
            "Office located in {value}.",
            "Incident reported in {value}.",
            "Transfer to facility in {value} approved.",
            "City: {value}",
            "The suspect was last seen in {value}.",
            "Branch office in {value}.",
            "Relocated to {value} for work.",
            "The claim was processed in {value}.",
            "Court held in {value}.",
            "Arrest made in {value}.",
            "Patient transferred from {value}.",
            "Employer based in {value}.",
            "Address: 123 Oak Ave, {value}, IL 60614",
            "Mailing from {value} region.",
        ],
    },

    "us_state": {
        "generator": _state_abbr,
        "templates": [
            "Address: 123 Main St, Springfield, {value} 60614",
            "License issued in {value}.",
            "State: {value}",
            "Registered in {value} 94301.",
            "Patient relocated to {value} 10001.",
            "The vehicle was registered in {value}.",
            "Claim filed in {value}.",
            "State of residence: {value}",
            "Born in {value}.",
            "Court jurisdiction: {value}",
        ],
    },

    "zipcode": {
        "generator": _zipcode,
        "templates": [
            "ZIP: {value}",
            "Postal code: {value}",
            "Mailing zip code {value}",
            "zip code: {value}",
            "Address: 742 Oak Ave, Chicago, IL {value}",
            "ZIP Code: {value}",
            "The patient's postal code is {value}.",
            "Mail to zip {value}.",
            "Service area: ZIP {value}",
            "Billing ZIP: {value}",
        ],
    },

    "precise_geolocation": {
        "generator": _gps,
        "templates": [
            "GPS coordinates: {value}",
            "Location: {value}",
            "lat/lon: {value}",
            "Recorded position {value}",
            "Device GPS: {value}",
            "Coordinates {value} logged at 14:32 UTC.",
            "Geolocation: {value}",
            "Last known coordinates: {value}",
            "Position fix: {value}",
            "Tracking data: {value}",
            "Phone location: {value}",
        ],
    },

    "date_of_birth": {
        "generator": _date_mmddyyyy,
        "templates": [
            "DOB: {value}",
            "Date of Birth: {value}",
            "The patient was born on {value}.",
            "born on {value}",
            "DOB {value} on file.",
            "Date of birth: {value}",
            "DOB: {value} per government-issued ID.",
            "Birthday: {value}",
            "Birth date: {value}",
            "Patient born {value}.",
            "D.O.B.: {value}",
            "Date of Birth (MM/DD/YYYY): {value}",
            "The suspect's date of birth is {value}.",
            "Employee birth date: {value}",
            "Verified DOB: {value}",
        ],
    },

    "clinical_date": {
        "generator": _date_clinical,
        "templates": [
            "PROCEDURE DATE: {value}",
            "Admission Date: {value}",
            "SERVICE DATE: {value}",
            "Visit date: {value}",
            "Discharge date: {value}",
            "DATE OF SERVICE: {value}",
            "Done on {value}.",
            "Submitted on {value}.",
            "Filed on {value}.",
            "Report date: {value}",
            "Enrolled on {value}.",
            "Completed on {value}.",
            "Date of procedure: {value}",
            "Treatment date: {value}",
            "Lab drawn on {value}.",
            "Authorization effective: {value}",
        ],
    },

    "phone_number": {
        "generator": _phone,
        "templates": [
            "Phone: {value}",
            "Call {value} for appointments.",
            "Contact number: {value}",
            "Mobile: {value}",
            "Patient phone: {value}",
            "Telephone: {value}",
            "Reach us at {value}.",
            "Callback number: {value}",
            "Cell: {value}",
            "Home Phone: {value}",
            "Work Phone: {value}",
            "Primary phone: {value}",
            "The patient's phone number is {value}.",
            "Contact {value} to schedule.",
            "Emergency line: {value}",
            "Pager: {value}",
        ],
    },

    "fax_number": {
        "generator": _fax,
        "templates": [
            "Fax: {value}",
            "Fax Number: {value}",
            "FAX: {value}",
            "Please fax to {value}.",
            "Fax No: {value}",
            "Send via fax to {value}.",
            "Fax referrals to {value}.",
            "Records fax: {value}",
            "Billing fax: {value}",
            "Fax line: {value}",
        ],
    },

    "email_address": {
        "generator": _email,
        "templates": [
            "Email: {value}",
            "Contact via email at {value}.",
            "Send records to {value}.",
            "Confirmation sent to {value}.",
            "Primary email: {value}",
            "Reach the patient at {value}.",
            "Notify {value} of the appointment.",
            "Electronic mail: {value}",
            "Work email: {value}",
            "Login email: {value}",
            "The patient's email is {value}.",
            "Reply to {value}.",
            "Forward reports to {value}.",
        ],
    },

    "ssn": {
        "generator": _ssn,
        "templates": [
            "SSN: {value}",
            "Social Security Number: {value}",
            "SSN on file: {value}",
            "Taxpayer SSN {value}",
            "Social Security: {value}",
            "SSN {value} verified.",
            "Social Security No: {value}",
            "SS#: {value}",
            "The patient's SSN is {value}.",
            "SSN provided: {value}",
            "Federal ID: {value}",
            "Tax SSN: {value}",
            "Identity verified via SSN {value}.",
            "SSN (last 4): see {value}",
            "Taxpayer identification: {value}",
        ],
    },

    "medical_record_number": {
        "generator": _mrn_value,
        "templates": [
            "MRN: {value}",
            "Medical Record Number: {value}",
            "Patient ID: {value}",
            "MRN {value} assigned.",
            "Medical Record No: {value}",
            "Chart Number: {value}",
            "Patient chart: {value}",
            "Record: {value}",
            "The MRN for this patient is {value}.",
            "Medical Record #: {value}",
            "Hospital record {value} created.",
            "EMR ID: {value}",
        ],
    },

    "health_plan_beneficiary_number": {
        "generator": _health_plan_id,
        "templates": [
            "Member ID: {value}",
            "Beneficiary Number: {value}",
            "Member Number: {value}",
            "Subscriber ID: {value}",
            "Group Number: {value}",
            "Health Plan Member ID: {value}",
            "Insurance ID: {value}",
            "Plan member: {value}",
            "The member ID is {value}.",
            "Insured ID: {value}",
            "Medicare ID: {value}",
            "Medicaid ID: {value}",
        ],
    },

    "insurance_policy_number": {
        "generator": _policy_number,
        "templates": [
            "Policy Number: {value}",
            "Insurance Policy: {value}",
            "Policy No: {value}",
            "Policy # {value}",
            "{value} is the active policy number.",
            "Policy Num: {value}",
            "The policy number on file is {value}.",
            "Coverage under policy {value}.",
            "Auto policy: {value}",
            "Health policy: {value}",
            "Life insurance policy: {value}",
        ],
    },

    "billing_number": {
        "generator": lambda: f"BILL-{random.randint(100000,9999999)}",
        "templates": [
            "Billing Number: {value}",
            "Bill ID: {value}",
            "Invoice Number: {value}",
            "Account billing: {value}",
            "Billing Reference: {value}",
            "Invoice ID: {value}",
            "Bill reference: {value}",
            "The billing number is {value}.",
            "Healthcare billing: {value}",
            "Billing account: {value}",
            "Invoice #: {value}",
            "Billing ID: {value}",
        ],
    },

    "medical_license_number": {
        "generator": _med_license,
        "templates": [
            "Medical License Number: {value}",
            "State License No: {value}",
            "License Number: {value}",
            "Certificate Number: {value}",
            "Medical License: {value}",
            "Physician License: {value}",
            "License #: {value}",
            "State medical license: {value}",
            "The provider holds license {value}.",
            "Board certificate: {value}",
        ],
    },

    "dea_number": {
        "generator": _dea,
        "templates": [
            "DEA Number: {value}",
            "DEA Registration: {value}",
            "Prescriber DEA: {value}",
            "DEA Reg No: {value}",
            "DEA license: {value}",
            "Controlled substance license: {value}",
            "The DEA number is {value}.",
            "Physician DEA: {value}",
            "DEA practitioner: {value}",
            "DEA authorized: {value}",
            "DEA registration number: {value}",
            "Rx DEA: {value}",
        ],
    },

    "npi_number": {
        "generator": _npi,
        "templates": [
            "NPI: {value}",
            "NPI Number: {value}",
            "Provider NPI: {value}",
            "Billing NPI: {value}",
            "Rendering NPI: {value}",
            "The NPI is {value}.",
            "National Provider ID: {value}",
            "Practitioner NPI: {value}",
            "NPI on claim: {value}",
            "Group NPI: {value}",
            "Physician NPI: {value}",
            "NPI registered: {value}",
        ],
    },

    "passport_number": {
        "generator": lambda: fake.bothify(text="??#######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "templates": [
            "Passport: {value}",
            "Passport No: {value}",
            "Travel document: {value}",
            "The passport number is {value}.",
            "Passport Number: {value}",
            "US Passport: {value}",
            "Passport document: {value}",
            "International passport: {value}",
            "Passport ID: {value}",
            "Travel passport: {value}",
            "Passport on file: {value}",
            "Traveler passport: {value}",
        ],
    },

    "drivers_license": {
        "generator": _drivers_license,
        "templates": [
            "Driver's License: {value}",
            "DL: {value}",
            "Driver License Number: {value}",
            "License No: {value}",
            "Driver's license number: {value}",
            "State DL: {value}",
            "The DL on file is {value}.",
            "License ID: {value}",
            "CDL: {value}",
            "Operator license: {value}",
        ],
    },

    "vehicle_vin": {
        "generator": _vin,
        "templates": [
            "VIN: {value}",
            "Vehicle Identification Number: {value}",
            "VIN # {value}",
            "The vehicle with VIN {value} was reported stolen.",
            "VIN Number: {value}",
            "Chassis number: {value}",
            "The VIN is {value}.",
            "Vehicle serial: {value}",
            "Auto VIN: {value}",
            "Title VIN: {value}",
        ],
    },

    "license_plate": {
        "generator": _license_plate,
        "templates": [
            "license plate: {value}",
            "Plate Number: {value}",
            "vehicle plate {value}",
            "Tag Number: {value}",
            "The vehicle bearing plate {value} was observed.",
            "License Plate Number: {value}",
            "Plate: {value}",
            "Tag: {value}",
            "Registration plate: {value}",
            "The plate {value} was run through DMV.",
        ],
    },

    "device_identifier": {
        "generator": _mac,
        "templates": [
            "MAC address: {value}",
            "Device MAC: {value}",
            "Network interface {value}",
            "Hardware address: {value}",
            "IMEI: {value}",
            "Device ID: {value}",
            "Device identifier: {value}",
            "MAC Address: {value}",
            "The device MAC is {value}.",
            "Physical address: {value}",
        ],
    },

    "url_with_pii": {
        "generator": _url,
        "templates": [
            "URL: {value}",
            "Portal link: {value}",
            "Access at: {value}",
            "Patient portal: {value}",
            "Link: {value}",
            "Profile URL: {value}",
            "Record URL: {value}",
            "Direct link: {value}",
            "Login URL: {value}",
            "Web address: {value}",
            "Healthcare portal: {value}",
            "Online record: {value}",
        ],
    },

    "ip_address": {
        "generator": _ip,
        "templates": [
            "IP Address: {value}",
            "Device IP: {value}",
            "Session IP: {value}",
            "Connected from {value}",
            "IP: {value}",
            "Source IP: {value}",
            "Login from {value}",
            "Remote address: {value}",
            "Client IP: {value}",
            "The connection originated from {value}.",
        ],
    },

    "biometric_facial_recognition": {
        "generator": lambda: f"facial_scan_{random.randint(1000,9999)}.bin",
        "templates": [
            "Facial recognition data: {value}",
            "Face scan stored as {value}",
            "Biometric face record {value}",
            "Face template: {value}",
            "Facial biometric file: {value}",
            "Facial ID reference: {value}",
            "Subject face hash: {value}",
            "Biometric enrollment record: {value}",
            "FR template: {value}",
            "Patient facial data: {value}",
            "Identity facial scan: {value}",
            "Stored face model: {value}",
        ],
    },

    "biometric_voiceprint": {
        "generator": lambda: f"voice_template_{random.randint(1000,9999)}.dat",
        "templates": [
            "Voiceprint identifier: {value}",
            "Voice recognition template: {value}",
            "Biometric voice data: {value}",
            "Voice file: {value}",
            "Speaker model: {value}",
            "Voice biometric reference: {value}",
            "Speaker ID template: {value}",
            "Enrolled voiceprint: {value}",
            "Audio biometric: {value}",
            "Voice authentication token: {value}",
            "Speaker recognition file: {value}",
            "Voice ID: {value}",
        ],
    },

    "biometric_iris_scan": {
        "generator": lambda: f"iris_scan_{random.randint(1000,9999)}.dat",
        "templates": [
            "Iris scan record: {value}",
            "Retina scan data: {value}",
            "Biometric iris record: {value}",
            "Iris template: {value}",
            "Eye scan file: {value}",
            "Iris recognition: {value}",
            "Iris enrollment: {value}",
            "Ocular biometric: {value}",
            "Iris ID: {value}",
            "Patient iris data: {value}",
            "Iris match reference: {value}",
            "Biometric iris file: {value}",
        ],
    },

    "biometric_dna": {
        "generator": lambda: f"DNA_profile_{random.randint(10000,99999)}",
        "templates": [
            "DNA profile: {value}",
            "Genetic data reference: {value}",
            "DNA record: {value}",
            "Genetic profile: {value}",
            "DNA sample: {value}",
            "CODIS profile: {value}",
            "Genomic reference: {value}",
            "DNA ID: {value}",
            "Genetic marker: {value}",
            "DNA biometric: {value}",
            "Subject DNA: {value}",
            "DNA analysis record: {value}",
        ],
    },

    "fingerprint": {
        "generator": lambda: f"FP_{random.randint(100000,999999)}",
        "templates": [
            "Fingerprint data: {value}",
            "Palm print record: {value}",
            "Biometric fingerprint: {value}",
            "Fingerprint ID: {value}",
            "Print record: {value}",
            "Latent print reference: {value}",
            "AFIS record: {value}",
            "Enrolled fingerprint: {value}",
            "Ten-print card: {value}",
            "Fingerprint match: {value}",
            "Fingerprint template: {value}",
            "Subject fingerprint: {value}",
        ],
    },

    "organization_name": {
        "generator": _company,
        "templates": [
            "Organization: {value}",
            "Employer: {value}",
            "Company: {value}",
            "Filed by {value}",
            "Authorized by {value}",
            "Entity: {value}",
            "Business: {value}",
            "The organization is {value}.",
            "Referred from {value}.",
            "The employer on record is {value}.",
        ],
    },

    "signature": {
        "generator": _first_last,
        "templates": [
            "Electronic signature: {value}",
            "Signed by: {value}",
            "Signature of {value} on file.",
            "Authorized signature: {value}",
            "E-sign: {value}",
            "Digital signature: {value}",
            "Wet signature: {value}",
            "Countersigned by {value}.",
            "Signatory: {value}",
            "Executed by {value}.",
            "Signature block: {value}",
            "Consent signed by {value}.",
        ],
    },

    # ── PCI-DSS ──────────────────────────────────────────────────────────────

    "credit_card_number": {
        "generator": _credit_card,
        "templates": [
            "Card number: {value}",
            "Credit card: {value}",
            "PAN: {value}",
            "Charged to card {value}",
            "Payment card: {value}",
            "Debit card: {value}",
            "Card No: {value}",
            "The card number used is {value}.",
            "Full card number: {value}",
            "Account PAN: {value}",
        ],
    },

    "card_expiration_date": {
        "generator": _card_exp,
        "templates": [
            "Expiry: {value}",
            "Expiration: {value}",
            "Card expiry: {value}",
            "Exp: {value}",
            "Expiry Date: {value}",
            "Valid thru: {value}",
            "Card expires: {value}",
            "Exp Date: {value}",
            "Expiration date: {value}",
            "Valid through: {value}",
            "Card valid until: {value}",
            "Expires: {value}",
        ],
    },

    "card_service_code": {
        "generator": _cvv,
        "templates": [
            "CVV2: {value}",
            "CVV: {value}",
            "CVC: {value}",
            "Security Code: {value}",
            "CID: {value}",
            "CAV2: {value}",
            "Card security code: {value}",
            "3-digit code: {value}",
            "Verification code: {value}",
            "Card verification: {value}",
            "CVC2: {value}",
            "Back of card code: {value}",
        ],
    },

    "card_track_data": {
        "generator": _card_track,
        "templates": [
            "Track data: {value}",
            "Magnetic stripe: {value}",
            "Track 1: {value}",
            "Swipe data: {value}",
            "MSR data: {value}",
            "Track 2 data: {value}",
            "Stripe read: {value}",
            "Full track: {value}",
            "Card magnetic data: {value}",
            "Track data captured: {value}",
            "Magnetic read: {value}",
            "Swiped data: {value}",
        ],
    },

    "card_pin": {
        "generator": lambda: str(random.randint(1000,9999)),
        "templates": [
            "PIN: {value}",
            "PIN Block: {value}",
            "Card PIN: {value}",
            "Debit PIN: {value}",
            "ATM PIN: {value}",
            "Entered PIN: {value}",
            "Cardholder PIN: {value}",
            "PIN code: {value}",
            "Security PIN: {value}",
            "Personal Identification Number: {value}",
            "Verified PIN: {value}",
            "Customer PIN: {value}",
        ],
    },

    "card_cryptogram": {
        "generator": _cryptogram,
        "templates": [
            "ARQC: {value}",
            "Cryptogram: {value}",
            "Dynamic Auth: {value}",
            "TC: {value}",
            "Chip cryptogram: {value}",
            "EMV cryptogram: {value}",
            "AAC: {value}",
            "Chip auth value: {value}",
            "ICC cryptogram: {value}",
            "Cryptogram value: {value}",
            "Auth response cryptogram: {value}",
            "Transaction cryptogram: {value}",
        ],
    },

    "card_iin_bin": {
        "generator": lambda: str(random.randint(400000,499999)),
        "templates": [
            "BIN: {value}",
            "IIN: {value}",
            "Issuer Number: {value}",
            "Issuer Identification Number: {value}",
            "Bank Identification Number: {value}",
            "Card BIN: {value}",
            "Issuer BIN: {value}",
            "Network IIN: {value}",
            "Card prefix: {value}",
            "BIN lookup: {value}",
            "Card issuer BIN: {value}",
            "First 6 digits: {value}",
        ],
    },

    "bank_account_number": {
        "generator": _bank_account,
        "templates": [
            "Bank Account Number: {value}",
            "Account Number: {value}",
            "Checking Account: {value}",
            "Savings Account: {value}",
            "Acct No: {value}",
            "Account #: {value}",
            "The bank account is {value}.",
            "Direct deposit account: {value}",
            "DDA: {value}",
            "Deposit to account: {value}",
            "Account on file: {value}",
            "Beneficiary account: {value}",
        ],
    },

    "bank_routing_number": {
        "generator": _routing,
        "templates": [
            "Routing Number: {value}",
            "ABA Number: {value}",
            "Routing No: {value}",
            "Bank routing: {value}",
            "ABA routing: {value}",
            "RTN: {value}",
            "The routing number is {value}.",
            "Wire routing: {value}",
            "ACH routing: {value}",
            "Federal routing: {value}",
            "Routing transit: {value}",
            "ABA transit number: {value}",
        ],
    },

    "iban": {
        "generator": _iban,
        "templates": [
            "IBAN: {value}",
            "International Bank Account: {value}",
            "Account IBAN: {value}",
            "IBAN Number: {value}",
            "International account: {value}",
            "Transfer to IBAN {value}",
            "Beneficiary IBAN: {value}",
            "Wire IBAN: {value}",
            "Remittance IBAN: {value}",
            "Recipient account IBAN: {value}",
            "SEPA account: {value}",
            "The IBAN is {value}.",
        ],
    },

    "swift_bic_code": {
        "generator": _swift,
        "templates": [
            "SWIFT Code: {value}",
            "BIC Code: {value}",
            "SWIFT: {value}",
            "BIC: {value}",
            "Bank SWIFT: {value}",
            "Wire BIC: {value}",
            "SWIFT/BIC: {value}",
            "Routing SWIFT: {value}",
            "Correspondent BIC: {value}",
            "Beneficiary SWIFT: {value}",
            "Bank identifier code: {value}",
            "The SWIFT code is {value}.",
        ],
    },

    "merchant_id": {
        "generator": _merchant_id,
        "templates": [
            "Merchant ID: {value}",
            "MID: {value}",
            "Merchant identifier: {value}",
            "Merchant number: {value}",
            "The merchant ID is {value}.",
            "Acquiring MID: {value}",
            "Merchant account: {value}",
            "Seller MID: {value}",
            "Payment facilitator MID: {value}",
            "MID on file: {value}",
            "Store merchant ID: {value}",
            "MID registered: {value}",
        ],
    },

    "terminal_id": {
        "generator": lambda: f"TID-{random.randint(10000000,99999999)}",
        "templates": [
            "Terminal ID: {value}",
            "TID: {value}",
            "POS terminal: {value}",
            "Terminal number: {value}",
            "Device TID: {value}",
            "POS TID: {value}",
            "Register TID: {value}",
            "Kiosk terminal: {value}",
            "Payment terminal: {value}",
            "Terminal identifier: {value}",
            "ATM terminal ID: {value}",
            "Lane terminal: {value}",
        ],
    },

    "transaction_id": {
        "generator": _transaction_id,
        "templates": [
            "Transaction ID: {value}",
            "TXN: {value}",
            "Transaction No: {value}",
            "Reference: {value}",
            "Auth reference: {value}",
            "TXN ID: {value}",
            "The transaction reference is {value}.",
            "Payment transaction: {value}",
            "Settlement ID: {value}",
            "Batch transaction: {value}",
            "Approval reference: {value}",
            "Transaction record: {value}",
        ],
    },

    "card_type": {
        "generator": lambda: random.choice(["Visa","Mastercard","American Express","Discover","AmEx"]),
        "templates": [
            "Card type: {value}",
            "Paid via {value}",
            "Payment method: {value}",
            "{value} card ending in 4521.",
            "Charged to {value}.",
            "Card brand: {value}",
            "Network: {value}",
            "Payment card: {value}",
            "Credit card type: {value}",
            "Card network: {value}",
            "Card association: {value}",
            "Card scheme: {value}",
        ],
    },

    "card_last4": {
        "generator": lambda: str(random.randint(1000,9999)),
        "templates": [
            "card ending in {value}",
            "Account ending in {value}",
            "last four digits: {value}",
            "Last 4: {value}",
            "Card last 4: {value}",
            "ending {value}",
            "Charged to card xxxx-{value}",
            "Card x-{value} on file",
            "card ****{value}",
            "Payment card ending {value}",
            "Debit card ending {value}",
            "Last four: {value}",
        ],
    },

    "bank_name": {
        "generator": _bank_name,
        "templates": [
            "Bank: {value}",
            "Financial institution: {value}",
            "Account at {value}",
            "Wire sent from {value}",
            "Issued by {value}",
            "Depository: {value}",
            "The bank is {value}.",
            "Lender: {value}",
            "Banking institution: {value}",
            "Account held at {value}.",
            "Funds from {value}.",
            "Credit union: {value}",
        ],
    },

    # ── General PII ──────────────────────────────────────────────────────────

    "password": {
        "generator": _password,
        "templates": [
            "password: {value}",
            "Password: {value}",
            "pwd: {value}",
            "passwd: {value}",
            "Your new password is {value}",
            "Temporary password: {value}",
            "System password: {value}",
            "Login password: {value}",
            "Account password: {value}",
            "Reset password: {value}",
            "Credentials password: {value}",
            "Auth password: {value}",
        ],
    },

    "username": {
        "generator": _username,
        "templates": [
            "username: {value}",
            "Username: {value}",
            "Login: {value}",
            "user_id: {value}",
            "userid: {value}",
            "Account name: {value}",
            "User: {value}",
            "Screen name: {value}",
            "Handle: {value}",
            "User login: {value}",
            "System user: {value}",
            "Network username: {value}",
        ],
    },

    "confidential": {
        "generator": lambda: random.choice(["CONFIDENTIAL","TOP SECRET","RESTRICTED","SENSITIVE PII"]),
        "templates": [
            "{value} — do not distribute.",
            "Marked as {value}.",
            "Classification: {value}",
            "{value} document enclosed.",
            "Document marked {value}.",
            "This record is {value}.",
            "Handling: {value}",
            "Security level: {value}",
            "Status: {value} — authorized personnel only.",
            "Clearance required: {value}",
            "Dissemination: {value}",
            "{value} — internal use only.",
        ],
    },

    "cookie_session_token": {
        "generator": _api_key,
        "templates": [
            "API Key: {value}",
            "Access Token: {value}",
            "Bearer Token: {value}",
            "Session Token: {value}",
            "Authorization: {value}",
            "Auth token: {value}",
            "JWT: {value}",
            "Cookie: {value}",
            "OAuth token: {value}",
            "Refresh token: {value}",
            "Session key: {value}",
            "Auth header: Bearer {value}",
        ],
    },

    "racial_ethnic_origin": {
        "generator": _dept_color,
        "templates": [
            "Race/Ethnicity: {value}",
            "Ethnic background: {value}",
            "Patient identified as {value}.",
            "Racial group: {value}",
            "Ethnicity: {value}",
            "Self-identified as {value}.",
            "Race: {value}",
            "Ethnic origin: {value}",
            "Cultural background: {value}",
            "Heritage: {value}",
            "Ancestry: {value}",
            "Demographic group: {value}",
        ],
    },

    "physical_characteristics": {
        "generator": _physical_desc,
        "templates": [
            "Physical description: {value}",
            "Subject description: {value}",
            "Characteristics: {value}",
            "Appearance: {value}",
            "Physical: {value}",
            "Descriptors: {value}",
            "Officer noted physical characteristics: {value}",
            "Suspect physical description: {value}",
            "Patient physical attributes: {value}",
            "Description of subject: {value}",
            "Known physical characteristics: {value}",
            "Medical physical description: {value}",
        ],
    },

    "race": {
        "generator": lambda: random.choice(["White","Black","Hispanic","Asian",
                                             "Native American","Pacific Islander","Multiracial"]),
        "templates": [
            "Race: {value}",
            "Patient race: {value}",
            "Race/Origin: {value}",
            "Identified race: {value}",
            "Reported race: {value}",
            "Subject race: {value}",
            "Race code: {value}",
            "Race self-reported: {value}",
            "Demographic race: {value}",
            "Patient identified race: {value}",
            "Race on record: {value}",
            "Race as reported: {value}",
        ],
    },

    "religion": {
        "generator": _religion_val,
        "templates": [
            "Religion: {value}",
            "Religious affiliation: {value}",
            "Faith: {value}",
            "Patient religion: {value}",
            "Religious belief: {value}",
            "Faith tradition: {value}",
            "The patient identifies as {value}.",
            "Religious preference: {value}",
            "Spiritual background: {value}",
            "Belief system: {value}",
            "Religious designation: {value}",
            "Observance: {value}",
        ],
    },

    "employment_history": {
        "generator": lambda: f"{_job_title()} at {_employer()}",
        "templates": [
            "Employment: {value}",
            "Previous employer: {value}",
            "Work history: {value}",
            "Job: {value}",
            "Occupation: {value}",
            "Current employment: {value}",
            "Position: {value}",
            "Former employment: {value}",
            "Job history: {value}",
            "Employer on record: {value}",
            "Past employment: {value}",
            "Career history: {value}",
        ],
    },

    "performance_evaluation": {
        "generator": lambda: random.choice(["Exceeds Expectations 4.5/5","Meets Expectations 3.2/5",
                                             "Needs Improvement 2.0/5","Outstanding 5/5"]),
        "templates": [
            "Performance rating: {value}",
            "Annual review score: {value}",
            "Appraisal result: {value}",
            "Performance evaluation: {value}",
            "Review outcome: {value}",
            "Staff evaluation: {value}",
            "Employee review: {value}",
            "Performance appraisal: {value}",
            "HR evaluation: {value}",
            "Year-end review: {value}",
            "Supervisor rating: {value}",
            "Performance score: {value}",
        ],
    },

    "student_records_ferpa": {
        "generator": lambda: f"GPA {round(random.uniform(1.5,4.0),2)}, enrolled in {random.randint(12,18)} credit hours",
        "templates": [
            "Student record: {value}",
            "Academic transcript: {value}",
            "Enrollment data: {value}",
            "FERPA record: {value}",
            "Academic standing: {value}",
            "Student academic data: {value}",
            "Academic record shows {value}.",
            "Transcript entry: {value}",
            "Student file: {value}",
            "Academic information: {value}",
            "Education record: {value}",
            "University record: {value}",
        ],
    },

    "state_id_number": {
        "generator": _state_id_num,
        "templates": [
            "State ID: {value}",
            "State Identification Number: {value}",
            "ID Number: {value}",
            "Government ID: {value}",
            "State issued ID: {value}",
            "The state ID number is {value}.",
            "ID presented: {value}",
            "Non-driver ID: {value}",
            "State photo ID: {value}",
            "State-issued identification: {value}",
            "Identification card number: {value}",
            "Photo ID number: {value}",
        ],
    },

    "employee_id": {
        "generator": _employee_id,
        "templates": [
            "Employee ID: {value}",
            "Staff ID: {value}",
            "Employee Number: {value}",
            "EMP ID: {value}",
            "Staff Number: {value}",
            "The employee ID is {value}.",
            "Worker ID: {value}",
            "Badge number: {value}",
            "Personnel ID: {value}",
            "HR employee number: {value}",
            "Payroll ID: {value}",
            "Staff badge: {value}",
        ],
    },

    "medication_name": {
        "generator": _medication,
        "templates": [
            "Prescribed: {value}",
            "Medication: {value}",
            "Rx: {value}",
            "Currently taking {value}",
            "Dispensed: {value}",
            "Current medications: {value}",
            "Drug: {value}",
            "Medication on file: {value}",
            "The patient is on {value}.",
            "Active prescription: {value}",
            "Administered: {value}",
            "Treating with {value}.",
        ],
    },

    "vehicle_registration": {
        "generator": _vehicle_reg,
        "templates": [
            "Vehicle Registration: {value}",
            "License Plate: {value}",
            "Plate No: {value}",
            "Registration No: {value}",
            "Reg Number: {value}",
            "Vehicle reg: {value}",
            "The vehicle registration is {value}.",
            "DMV registration: {value}",
            "Vehicle plate: {value}",
            "Registered plate: {value}",
            "License tag: {value}",
            "Vehicle ID tag: {value}",
        ],
    },

    "student_id": {
        "generator": _student_id,
        "templates": [
            "Student ID: {value}",
            "Student Number: {value}",
            "STU ID: {value}",
            "Student ID Number: {value}",
            "The student ID is {value}.",
            "Student identification: {value}",
            "University ID: {value}",
            "Campus ID: {value}",
            "Student account: {value}",
            "Registrar ID: {value}",
            "Student card number: {value}",
            "Student enrolled with ID: {value}",
        ],
    },

    "tax_id_number": {
        "generator": _tax_id,
        "templates": [
            "Tax ID: {value}",
            "TIN: {value}",
            "EIN: {value}",
            "Tax Identification Number: {value}",
            "Federal TIN: {value}",
            "The tax ID is {value}.",
            "Employer EIN: {value}",
            "Federal tax ID: {value}",
            "Business TIN: {value}",
            "FEIN: {value}",
            "Tax identification: {value}",
            "IRS number: {value}",
        ],
    },

    "flight_number": {
        "generator": _flight_number,
        "templates": [
            "Flight {value}",
            "Flt {value}",
            "Booked on flight {value}",
            "Departure: flight {value}",
            "Flight number: {value}",
            "Operating as flight {value}.",
            "Connecting flight {value}.",
            "Scheduled flight: {value}",
            "Passenger on flight {value}.",
            "Ticketed flight: {value}",
            "Airline flight: {value}",
            "Arrival flight: {value}",
        ],
    },

    "booking_reference": {
        "generator": _booking_ref,
        "templates": [
            "Booking Reference: {value}",
            "Reservation ID: {value}",
            "PNR: {value}",
            "Booking ID: {value}",
            "Confirmation Number: {value}",
            "Reservation code: {value}",
            "The booking reference is {value}.",
            "Itinerary number: {value}",
            "Ticket reference: {value}",
            "Travel booking: {value}",
            "Booking code: {value}",
            "Reservation number: {value}",
        ],
    },

    "claim_number": {
        "generator": _claim_number,
        "templates": [
            "Claim Number: {value}",
            "CLM: {value}",
            "Claim ID: {value}",
            "Insurance claim: {value}",
            "Claim Reference: {value}",
            "The claim number is {value}.",
            "Claim #: {value}",
            "Filed claim: {value}",
            "Claim record: {value}",
            "Adjuster claim: {value}",
            "Claim tracking: {value}",
            "Open claim: {value}",
        ],
    },

    "application_id": {
        "generator": lambda: f"GOV-{random.randint(100000,999999)}",
        "templates": [
            "Application Number: {value}",
            "Reference Number: {value}",
            "Application ID: {value}",
            "GOV: {value}",
            "Reference ID: {value}",
            "The application number is {value}.",
            "App ID: {value}",
            "Case application: {value}",
            "Filing reference: {value}",
            "Government application: {value}",
            "Request ID: {value}",
            "Benefit application: {value}",
        ],
    },

    "hospital_name": {
        "generator": _hospital,
        "templates": [
            "Facility: {value}",
            "Referred to {value}",
            "Admitted at {value}",
            "Hospital: {value}",
            "Treatment at {value}",
            "Healthcare facility: {value}",
            "Discharged from {value}.",
            "The treating facility is {value}.",
            "Medical center: {value}",
            "Inpatient at {value}.",
            "Clinical facility: {value}",
            "Patient transferred to {value}.",
        ],
    },

    "insurance_company_name": {
        "generator": _insurance_co,
        "templates": [
            "Insurance company: {value}",
            "Insurer: {value}",
            "Covered by {value}",
            "Policy with {value}",
            "Claim filed with {value}",
            "Insurance carrier: {value}",
            "Coverage through {value}.",
            "Provider: {value}",
            "Underwriter: {value}",
            "The insurer is {value}.",
            "Payer: {value}",
            "Health plan: {value}",
        ],
    },

    "university_name": {
        "generator": _university,
        "templates": [
            "University: {value}",
            "Institution: {value}",
            "School: {value}",
            "Enrolled at {value}",
            "Degree from {value}",
            "College: {value}",
            "Student at {value}.",
            "Transcript from {value}.",
            "Attended {value}.",
            "Academic institution: {value}",
            "The university is {value}.",
            "Higher education: {value}",
        ],
    },

    "law_firm_name": {
        "generator": _law_firm,
        "templates": [
            "Law firm: {value}",
            "Represented by {value}",
            "Counsel: {value}",
            "Attorney firm: {value}",
            "Legal representation: {value}",
            "Defense counsel: {value}",
            "Plaintiff counsel: {value}",
            "Filed by {value}.",
            "Retained firm: {value}",
            "The law firm is {value}.",
            "Legal firm: {value}",
            "Firm of record: {value}",
        ],
    },

    "court_name": {
        "generator": _court_name,
        "templates": [
            "Court: {value}",
            "Filed in {value}",
            "Jurisdiction: {value}",
            "Presiding court: {value}",
            "Case heard in {value}.",
            "Tribunal: {value}",
            "Venue: {value}",
            "Court of record: {value}",
            "Adjudicated in {value}.",
            "Hearing at {value}.",
            "The court is {value}.",
            "Judicial venue: {value}",
        ],
    },

    "hotel_name": {
        "generator": _hotel_name,
        "templates": [
            "Hotel: {value}",
            "Accommodation: {value}",
            "Staying at {value}",
            "Reserved at {value}",
            "Property: {value}",
            "Lodging: {value}",
            "Check-in at {value}.",
            "Booked at {value}.",
            "Guest at {value}.",
            "Hotel reservation: {value}",
            "The hotel is {value}.",
            "Resort: {value}",
        ],
    },

    "financial_amount": {
        "generator": _amount,
        "templates": [
            "Amount: {value}",
            "Total: {value}",
            "Balance: {value}",
            "Transaction amount: {value}",
            "Payment of {value} received.",
            "Outstanding balance: {value}",
            "Claim amount: {value}",
            "Transfer amount: {value}",
            "Settlement amount: {value}",
            "Invoice total: {value}",
            "Refund amount: {value}",
            "Due amount: {value}",
        ],
    },

    # ── Law Enforcement / CJIS ───────────────────────────────────────────────

    "fbi_number": {
        "generator": lambda: f"FBI-{random.randint(1000000,9999999)}",
        "templates": [
            "FBI number: {value}",
            "FBI ID: {value}",
            "Federal record: {value}",
            "FBI record: {value}",
            "NCIC FBI number: {value}",
            "Federal bureau number: {value}",
            "FBI reference: {value}",
            "Federal criminal ID: {value}",
            "FBI criminal record: {value}",
            "UCN number: {value}",
            "FBI file number: {value}",
            "Bureau record: {value}",
        ],
    },

    "chri": {
        "generator": lambda: f"CHRI-{random.randint(1000000,9999999)}",
        "templates": [
            "Criminal history: {value}",
            "CHRI Record: {value}",
            "Criminal record: {value}",
            "History record: {value}",
            "Rap sheet: {value}",
            "Criminal history reference: {value}",
            "CHRI number: {value}",
            "Criminal history information: {value}",
            "Background check: {value}",
            "Criminal file: {value}",
            "CJIS record: {value}",
            "Criminal history ID: {value}",
        ],
    },

    "arrest_record": {
        "generator": lambda: f"AR-{random.randint(2018,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Arrest record: {value}",
            "Booking number: {value}",
            "Arrest ID: {value}",
            "Custody record: {value}",
            "Arrest file: {value}",
            "The arrest record is {value}.",
            "Arrest reference: {value}",
            "Apprehension record: {value}",
            "Booking record: {value}",
            "Detention record: {value}",
            "Arrest log: {value}",
            "Police arrest: {value}",
        ],
    },

    "case_number": {
        "generator": _case_number,
        "templates": [
            "Case number: {value}",
            "Docket: {value}",
            "Case ID: {value}",
            "Criminal case: {value}",
            "Docket number: {value}",
            "Case file: {value}",
            "The case number is {value}.",
            "Court case: {value}",
            "NCIC case: {value}",
            "Case reference: {value}",
            "Investigation case: {value}",
            "Open case: {value}",
        ],
    },

    "warrant_data": {
        "generator": lambda: f"W-{random.randint(2018,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Warrant number: {value}",
            "Active warrant: {value}",
            "Warrant ID: {value}",
            "NCIC warrant: {value}",
            "Warrant on file: {value}",
            "The warrant number is {value}.",
            "Outstanding warrant: {value}",
            "Arrest warrant: {value}",
            "Search warrant: {value}",
            "Bench warrant: {value}",
            "Warrant issued: {value}",
            "Court warrant: {value}",
        ],
    },

    "incident_report": {
        "generator": lambda: f"IR-{random.randint(2018,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Incident report: {value}",
            "Case report: {value}",
            "IR Number: {value}",
            "Police report: {value}",
            "Incident number: {value}",
            "The incident report is {value}.",
            "Field incident: {value}",
            "Report number: {value}",
            "Complaint number: {value}",
            "Event number: {value}",
            "Incident ID: {value}",
            "CAD number: {value}",
        ],
    },

    "incarceration_info": {
        "generator": lambda: f"INCAR-{random.randint(100000,999999)}",
        "templates": [
            "Incarceration record: {value}",
            "Booking reference: {value}",
            "Detention ID: {value}",
            "Inmate number: {value}",
            "Custody file: {value}",
            "The incarceration record is {value}.",
            "Prison record: {value}",
            "Jail record: {value}",
            "Inmate ID: {value}",
            "DOC number: {value}",
            "Facility inmate: {value}",
            "Confinement record: {value}",
        ],
    },

    "missing_person_report": {
        "generator": lambda: f"MP-{random.randint(2020,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Missing Person Report: {value}",
            "NCIC missing: {value}",
            "Missing person report #: {value}",
            "MP Report: {value}",
            "Missing person case: {value}",
            "Filed missing person: {value}",
            "Endangered missing: {value}",
            "Missing child report: {value}",
            "NCIC MP entry: {value}",
            "Missing report number: {value}",
            "MP case file: {value}",
            "Missing person ID: {value}",
        ],
    },

    "wanted_person_report": {
        "generator": lambda: f"WP-{random.randint(2020,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Wanted person report: {value}",
            "Fugitive record: {value}",
            "NCIC wanted: {value}",
            "WP Report: {value}",
            "Fugitive case: {value}",
            "Active wanted: {value}",
            "Wanted bulletin: {value}",
            "NCIC WP entry: {value}",
            "Wanted person file: {value}",
            "Wanted report: {value}",
            "Fugitive notice: {value}",
            "Wanted person ID: {value}",
        ],
    },

    "sex_offender_report": {
        "generator": lambda: f"SO-{random.randint(10000,999999)}",
        "templates": [
            "Sex offender registration: {value}",
            "SORA record: {value}",
            "Registered offender ID: {value}",
            "SO Registration: {value}",
            "Sex offender record: {value}",
            "Registered sex offender: {value}",
            "SO registry entry: {value}",
            "NCIC SO: {value}",
            "Offender registration: {value}",
            "Megan's Law record: {value}",
            "Sex offender file: {value}",
            "Offender ID: {value}",
        ],
    },

    "protection_orders": {
        "generator": lambda: f"PO-{random.randint(2020,2024)}-{random.randint(10000,99999)}",
        "templates": [
            "Restraining order: {value}",
            "Protection order: {value}",
            "TRO: {value}",
            "Protection Order Number: {value}",
            "Order of protection: {value}",
            "Protective order: {value}",
            "CPO: {value}",
            "No contact order: {value}",
            "Civil protection: {value}",
            "Domestic violence order: {value}",
            "Injunctive order: {value}",
            "NCIC PO: {value}",
        ],
    },

    "foreign_fugitives": {
        "generator": lambda: f"FF-{random.randint(10000,999999)}",
        "templates": [
            "Foreign fugitive: {value}",
            "Interpol record: {value}",
            "Extradition number: {value}",
            "Foreign fugitive record: {value}",
            "Interpol notice: {value}",
            "Red notice: {value}",
            "Extradition request: {value}",
            "International warrant: {value}",
            "NCIC foreign fugitive: {value}",
            "International fugitive: {value}",
            "Cross-border warrant: {value}",
            "Foreign wanted: {value}",
        ],
    },

    "identity_theft_victims": {
        "generator": lambda: f"IDT-{random.randint(10000,999999)}",
        "templates": [
            "Identity theft report: {value}",
            "ID theft victim record: {value}",
            "IDT Report: {value}",
            "Identity theft case: {value}",
            "ID theft record: {value}",
            "Victim identity file: {value}",
            "FTC ID theft report: {value}",
            "Identity fraud report: {value}",
            "ID fraud case: {value}",
            "Impersonation report: {value}",
            "Identity compromise: {value}",
            "Theft of identity: {value}",
        ],
    },

    "gang_terrorist_member": {
        "generator": lambda: f"GT-{random.randint(10000,999999)}",
        "templates": [
            "Gang member record: {value}",
            "Terrorist watchlist entry: {value}",
            "Watchlist record: {value}",
            "GT Record: {value}",
            "Gang affiliation record: {value}",
            "Watchlist ID: {value}",
            "Terrorist screening: {value}",
            "NCIC gang file: {value}",
            "Known gang member: {value}",
            "Terrorist database entry: {value}",
            "Extremist record: {value}",
            "Gang intelligence file: {value}",
        ],
    },

    "supervised_release": {
        "generator": lambda: f"SR-{random.randint(10000,999999)}",
        "templates": [
            "Supervised release: {value}",
            "Supervision record: {value}",
            "SR Record: {value}",
            "Post-release supervision: {value}",
            "Supervised release number: {value}",
            "Supervised release file: {value}",
            "Release supervision: {value}",
            "Federal supervised release: {value}",
            "Supervision case: {value}",
            "Post-incarceration supervision: {value}",
            "Release conditions: {value}",
            "SR case number: {value}",
        ],
    },

    "probation_record": {
        "generator": lambda: f"PROB-{random.randint(10000,999999)}",
        "templates": [
            "Probation record: {value}",
            "Probation number: {value}",
            "PROB Record: {value}",
            "Probation case: {value}",
            "Probation file: {value}",
            "Probation order: {value}",
            "Community supervision: {value}",
            "Probation case number: {value}",
            "Probation officer file: {value}",
            "Misdemeanor probation: {value}",
            "Felony probation: {value}",
            "Court probation: {value}",
        ],
    },

    "parole_record": {
        "generator": lambda: f"PAR-{random.randint(10000,999999)}",
        "templates": [
            "Parole record: {value}",
            "Parole number: {value}",
            "PAR Record: {value}",
            "Parole case: {value}",
            "Parole file: {value}",
            "Parole order: {value}",
            "Conditional release: {value}",
            "Parole board record: {value}",
            "Parole officer file: {value}",
            "Parole case number: {value}",
            "Early release record: {value}",
            "State parole: {value}",
        ],
    },

    "stolen_vehicle": {
        "generator": lambda: f"SV-{random.randint(10000,999999)}",
        "templates": [
            "Stolen vehicle report: {value}",
            "NCIC stolen vehicle: {value}",
            "SV Report: {value}",
            "Stolen car report: {value}",
            "Stolen vehicle record: {value}",
            "Vehicle theft report: {value}",
            "NCIC SV entry: {value}",
            "Auto theft report: {value}",
            "Stolen auto: {value}",
            "Vehicle theft file: {value}",
            "Stolen car file: {value}",
            "NCIC vehicle theft: {value}",
        ],
    },

    "stolen_guns": {
        "generator": lambda: f"SG-{random.randint(10000,999999)}",
        "templates": [
            "Stolen gun report: {value}",
            "Stolen firearm record: {value}",
            "NCIC firearm: {value}",
            "SG Report: {value}",
            "Stolen weapon report: {value}",
            "Firearm theft report: {value}",
            "NCIC SG entry: {value}",
            "Gun theft record: {value}",
            "Weapon theft file: {value}",
            "Stolen pistol report: {value}",
            "Stolen firearm file: {value}",
            "Firearm recovery: {value}",
        ],
    },

    "stolen_license_plate": {
        "generator": lambda: f"SLP-{random.randint(1000,9999)}",
        "templates": [
            "Stolen plate: {value}",
            "Stolen license plate: {value}",
            "SLP Report: {value}",
            "Stolen plate report: {value}",
            "Plate theft record: {value}",
            "NCIC stolen plate: {value}",
            "License plate theft: {value}",
            "Plate theft file: {value}",
            "Stolen tag: {value}",
            "Tag theft report: {value}",
            "Plate theft ID: {value}",
            "NCIC SLP entry: {value}",
        ],
    },

    "stolen_boats": {
        "generator": lambda: f"SB-{random.randint(10000,999999)}",
        "templates": [
            "Stolen boat report: {value}",
            "Stolen vessel record: {value}",
            "SB Report: {value}",
            "Stolen watercraft report: {value}",
            "Boat theft record: {value}",
            "Marine theft report: {value}",
            "Vessel theft file: {value}",
            "NCIC SB entry: {value}",
            "Watercraft theft: {value}",
            "Boat theft ID: {value}",
            "Marine theft record: {value}",
            "Stolen yacht report: {value}",
        ],
    },

    "stolen_securities": {
        "generator": lambda: f"SS-{random.randint(10000,999999)}",
        "templates": [
            "Stolen securities report: {value}",
            "Stolen bonds record: {value}",
            "SS Report: {value}",
            "Stolen stocks report: {value}",
            "Securities theft record: {value}",
            "Bond theft report: {value}",
            "NCIC securities theft: {value}",
            "Stolen stock certificate: {value}",
            "Securities fraud report: {value}",
            "Financial theft record: {value}",
            "Stock theft file: {value}",
            "NCIC SS entry: {value}",
        ],
    },

    "stolen_articles": {
        "generator": lambda: f"SA-{random.randint(10000,999999)}",
        "templates": [
            "Stolen property report: {value}",
            "Stolen articles record: {value}",
            "SA Report: {value}",
            "Stolen goods report: {value}",
            "Property theft record: {value}",
            "Property theft report: {value}",
            "Stolen item record: {value}",
            "NCIC SA entry: {value}",
            "Stolen article ID: {value}",
            "Property crime file: {value}",
            "Theft report: {value}",
            "Stolen item ID: {value}",
        ],
    },

    # ── Transportation ───────────────────────────────────────────────────────

    "driver_history": {
        "generator": lambda: f"DMV-{random.randint(100000,9999999)}",
        "templates": [
            "Driving record: {value}",
            "Driver history: {value}",
            "DMV record: {value}",
            "Driver History Record: {value}",
            "MVR: {value}",
            "The MVR number is {value}.",
            "Motor vehicle record: {value}",
            "Driver abstract: {value}",
            "DMV file: {value}",
            "Traffic record: {value}",
            "License history: {value}",
            "Driver license record: {value}",
        ],
    },

    "bart_employee_id": {
        "generator": _bart_emp_id,
        "templates": [
            "BART Employee ID: {value}",
            "Employee ID: {value}",
            "Emp ID: {value}",
            "Employee No: {value}",
            "BART ID: {value}",
            "The BART employee number is {value}.",
            "Transit employee ID: {value}",
            "BART badge number: {value}",
            "Agency employee ID: {value}",
            "Operator ID: {value}",
            "Staff ID number: {value}",
            "BART worker ID: {value}",
        ],
    },
}


# ---------------------------------------------------------------------------
# Hard negatives — sentences that look like PII context but contain no entity
# ---------------------------------------------------------------------------

HARD_NEGATIVES: list[str] = [
    "The SSN field was left blank on the form.",
    "Please enter your date of birth in the field below.",
    "Employee ID is required for system access.",
    "Contact your insurance provider for your member ID.",
    "The credit card number has not been provided.",
    "Please provide a valid phone number to continue.",
    "The patient name field is required.",
    "Enter your email address to receive the confirmation.",
    "A Social Security Number is needed for verification.",
    "The routing number must be exactly nine digits.",
    "Please upload a copy of your driver's license.",
    "Passport number is required for international travel.",
    "The VIN should be 17 alphanumeric characters.",
    "Your Medicare number can be found on your Medicare card.",
    "The bank account number was not included in the form.",
    "Date of birth is required for age verification.",
    "Your health plan member ID is printed on your insurance card.",
    "The NPI number was not found in the provider directory.",
    "Please fax your records to the billing department.",
    "The prescription number appears on the medication label.",
    "IP address logging is enabled for security purposes.",
    "The MAC address is printed on the device label.",
    "Please provide a valid ZIP code.",
    "Your policy number can be found on your insurance card.",
    "The medical record number was not located in the system.",
    "Please enter a valid credit card number.",
    "The CVV is the 3-digit code on the back of your card.",
    "Your card expiration date is printed on the front of the card.",
    "A tax ID number is required for business registration.",
    "The claim number will be assigned after submission.",
    "Please provide your student ID to access the library.",
    "The booking reference will be emailed to you.",
    "Your flight number is printed on your boarding pass.",
    "The case number will be assigned by the clerk of court.",
    "A warrant number is issued by the presiding judge.",
    "The incident report number is assigned at the scene.",
    "Please contact human resources for your employee ID.",
    "The DEA number is required on all controlled substance orders.",
    "Your IBAN can be found on your bank statement.",
    "The SWIFT code is used for international wire transfers.",
    "Please provide a valid email address.",
    "The password must be at least eight characters.",
    "Enter your username to log in to the system.",
    "The transaction ID will appear on your receipt.",
    "Your merchant ID is assigned by your payment processor.",
    "The terminal ID is printed on the POS device label.",
    "Please enter the last four digits of your card.",
    "The BIN identifies the card issuing bank.",
    "Track data must not be stored after authorization.",
    "The PIN should never be written down or shared.",
    "Please do not share your authentication token.",
    "An API key is required to access the service.",
    "The session will expire after 30 minutes of inactivity.",
    "Race and ethnicity data is collected for reporting purposes.",
    "Religious accommodations must be requested in writing.",
    "Employment history is verified during the background check.",
    "Performance evaluations are conducted annually.",
    "Student records are protected under FERPA.",
    "DNA samples are collected at booking per state law.",
    "Fingerprints are required for the background check.",
    "Facial recognition is used for facility access control.",
    "The voiceprint template is stored in an encrypted format.",
    "Iris scan enrollment requires in-person verification.",
    "The arrest record will be expunged after five years.",
    "Probation conditions are outlined in the court order.",
    "Parole eligibility is determined by the board.",
    "The stolen vehicle report must include the VIN.",
    "A missing person report can be filed at any police station.",
    "The protection order was not yet entered in the system.",
    "Please contact the FBI for the identification number.",
    "Criminal history records are restricted under CJIS policy.",
    "The gang affiliation information is not publicly available.",
    "Supervised release conditions will be reviewed quarterly.",
    "The driving record will be requested from the DMV.",
    "BART employee IDs are issued by HR upon hiring.",
    "The hotel confirmation number will be emailed to you.",
    "A booking reference is required to modify your reservation.",
    "The court name and division appear on the filing receipt.",
    "Law firm contact information is available on the bar website.",
    "University name and degree must be verified by the registrar.",
    "Insurance company details are on the back of the card.",
    "The hospital name appears on the admission paperwork.",
    "Financial amounts are reported in US dollars unless noted.",
    "The application reference number will be provided at submission.",
    "Please verify your identity before accessing the record.",
    "No personal information should be shared over unencrypted channels.",
    "The form requires all fields to be completed before submission.",
    "Sensitive data must be encrypted at rest and in transit.",
    "The patient consented to the collection of their health information.",
    "Do not leave patient data visible on unattended screens.",
    "All PHI must be de-identified before use in research.",
    "The organization follows HIPAA privacy rule requirements.",
    "PCI-DSS compliance requires quarterly security scans.",
    "Card data must be masked when displayed on receipts.",
    "The data breach notification was sent to affected individuals.",
    "Personal information collected is used solely for the stated purpose.",
    "You have the right to request access to your medical records.",
    "The privacy officer handles all HIPAA complaints.",
    "Employees must complete annual privacy training.",
    "All visitors must sign in and present photo ID.",
    "The records retention schedule is posted on the intranet.",
    "Destruction of records must be documented and witnessed.",
    "Third-party vendors must sign a business associate agreement.",
    "The minimum necessary standard applies to all PHI disclosures.",
    "Patients may opt out of the facility directory.",
    "The authorization must be signed before releasing records.",
    "De-identified data does not require a HIPAA authorization.",
    "The limited data set may be used for research purposes.",
    "A data use agreement is required for limited data sets.",
    "The covered entity must designate a privacy official.",
    "Workforce members who violate the privacy rule face sanctions.",
    "The security rule applies to electronic protected health information.",
    "Physical safeguards include locked storage and access controls.",
    "Technical safeguards include encryption and audit logs.",
    "Administrative safeguards include policies and training.",
    "The contingency plan must be tested annually.",
    "Risk analysis must be conducted and documented.",
    "Encryption of ePHI is an addressable implementation specification.",
    "The breach must be reported within 60 days of discovery.",
    "Small breaches affecting fewer than 500 individuals are logged.",
    "The notice of privacy practices must be provided at first visit.",
    "Patients have the right to amend their health information.",
    "The right of access allows patients to obtain their records.",
    "Accounting of disclosures is available upon patient request.",
    "The complaint must be filed within 180 days of the violation.",
    "CMS oversees the enforcement of the HIPAA privacy rule.",
    "Civil monetary penalties may be imposed for violations.",
    "Criminal penalties apply to willful disclosure of PHI.",
    "The Safe Harbor method requires removal of 18 identifiers.",
    "The expert determination method uses statistical methods.",
    "Dates other than year must be removed for de-identification.",
    "Geographic data smaller than state must be removed.",
    "Ages over 89 must be aggregated for de-identification.",
    "Unique identifiers must be removed under Safe Harbor.",
    "The covered entity certifies that re-identification is unlikely.",
    "The minimum necessary standard limits incidental disclosure.",
    "Treatment, payment, and operations are permitted purposes.",
    "Psychotherapy notes require a separate authorization.",
    "The right to restrict applies to out-of-pocket paid services.",
    "Electronic access must be provided within 30 days.",
    "Fees for copies must be cost-based and reasonable.",
    "Third-party access is allowed with a valid authorization.",
    "The privacy notice must describe uses and disclosures.",
    "Marketing communications require a valid authorization.",
    "Sale of PHI requires a separate authorization.",
    "Research uses may require a waiver of authorization.",
    "IRB oversight is required for research involving PHI.",
    "Business associates are subject to the same privacy obligations.",
    "Subcontractors must sign a business associate agreement.",
    "The covered entity is responsible for its business associates.",
    "Breach notification applies to unsecured PHI.",
    "Secured PHI includes encrypted and destroyed information.",
    "Unauthorized access by a workforce member is a breach.",
    "Lost or stolen devices containing unencrypted PHI are breaches.",
    "The breach assessment uses a four-factor analysis.",
    "Notification must include what happened and what was exposed.",
    "Substitute notice may be used when contact information is outdated.",
    "Media notice is required for breaches affecting over 500 residents.",
    "All breach investigations must be documented.",
    "The covered entity retains breach documentation for six years.",
    "The privacy rule preempts state law unless state law is stricter.",
    "California law provides additional privacy protections.",
    "The CCPA grants consumers the right to know and delete.",
    "GDPR applies to EU residents regardless of where data is processed.",
    "The right to erasure allows individuals to request deletion.",
    "Data controllers must respond to subject access requests within one month.",
    "The lawful basis for processing must be documented.",
    "Consent must be freely given, specific, informed, and unambiguous.",
    "Data protection impact assessments are required for high-risk processing.",
    "The data protection officer must be appointed in certain cases.",
    "Cross-border transfers require appropriate safeguards.",
    "Standard contractual clauses are a valid transfer mechanism.",
    "Supervisory authorities enforce GDPR compliance.",
    "Individuals must be notified without undue delay.",
    "Data minimization requires collecting only what is necessary.",
    "Purpose limitation restricts use to the original collection purpose.",
    "Storage limitation requires deletion when no longer needed.",
    "Integrity and confidentiality require appropriate security.",
    "Accountability requires demonstrating compliance.",
    "Automated decision-making rights apply to significant decisions.",
    "Special categories of data require explicit consent or exception.",
    "Health data is a special category under GDPR.",
    "Biometric data processed for identification is a special category.",
    "Genetic data is explicitly listed as a special category.",
    "Criminal records require authorization under national law.",
    "Children's data requires parental consent in most jurisdictions.",
    "Privacy by design must be integrated into system development.",
    "Privacy by default means the most privacy-protective settings apply.",
    "Records of processing activities must be maintained.",
    "Processors must act only on instructions from the controller.",
    "The field for the patient date of birth was intentionally left blank.",
    "The SSN was redacted from the document prior to disclosure.",
    "No identifying information was included in the summary report.",
    "The patient opted out of including their name in the registry.",
    "The record was anonymized before being used in the study.",
    "Personal identifiers were removed from the dataset.",
    "The research cohort does not include any identifying information.",
    "The claim was submitted without the patient SSN by mistake.",
    "The employee ID field is auto-populated by the HR system.",
    "The tax ID number will be provided upon request.",
    "A valid government ID must be presented at the front desk.",
    "The phone number field accepts only US formats.",
    "Please do not include your password in email communications.",
    "The username must be unique across the system.",
    "Two-factor authentication is required for privileged accounts.",
    "Session tokens expire after 60 minutes of inactivity.",
    "API keys should be rotated every 90 days.",
    "The form does not collect financial account information.",
    "Cardholder data must not be stored after authorization.",
    "The PAN must be masked when displayed to non-privileged users.",
    "The CVV must never be stored under any circumstances.",
    "PIN blocks must be encrypted using approved algorithms.",
    "Track data retention is prohibited under PCI-DSS.",
    "Merchant IDs are assigned by the acquiring bank.",
    "Terminal IDs must be verified before deployment.",
    "IBAN format varies by country.",
    "ACH transactions use ABA routing numbers for domestic transfers.",
    "The bank account number was masked in the statement.",
    "Wire transfers require both routing and account numbers.",
    "The financial amount field accepts values up to six figures.",
    "The claim amount is determined by the fee schedule.",
    "Prior authorization is required for amounts over a certain threshold.",
    "The deductible must be met before benefits apply.",
    "Out-of-pocket maximums reset at the start of each plan year.",
    "The explanation of benefits shows what the insurer paid.",
    "The subrogation clause allows recovery from third parties.",
    "The preferred provider organization offers in-network discounts.",
    "High-deductible health plans qualify for health savings accounts.",
    "The empty field awaits a valid social security number.",
    "No name was provided in the referral paperwork.",
    "The form was returned because the date of birth was missing.",
    "The claim was rejected due to an invalid member ID.",
    "The address field must include both street and city.",
    "Please contact IT to reset your system password.",
    "The username field does not accept special characters.",
    "Enter the last known IP address of the device.",
    "The access log will record the originating IP address.",
    "Card data was not present in this transaction.",
    "The routing number check failed validation.",
    "The account number format was incorrect.",
    "A valid IBAN is required for international transfers.",
    "The SWIFT code must be eight or eleven characters.",
    "Please present a valid photo identification at the counter.",
    "The fingerprint scanner was offline during enrollment.",
    "Biometric authentication requires initial enrollment.",
    "No genetic data has been submitted for this individual.",
    "The DNA sample was excluded from the report.",
    "The facial recognition system did not produce a match.",
    "Voice authentication is not enabled for this account.",
    "The iris scan file was corrupted and could not be read.",
    "No driving violations are on record for this individual.",
    "The DMV record is pending from the state.",
    "The warrant has been recalled and is no longer active.",
    "No open warrants were found for this individual.",
    "The missing person report was closed after the individual was located.",
    "The fugitive was returned to jurisdiction and the report was cleared.",
    "The watchlist entry was removed following adjudication.",
    "No gang affiliation was found during the background investigation.",
    "The protection order expired and was not renewed.",
    "The probation term concluded without violations.",
    "Parole was successfully completed and the case was closed.",
    "The supervised release term ended without incident.",
    "The stolen vehicle was recovered and the report was closed.",
    "The firearm was recovered and returned to the owner.",
    "The license plate was found and returned to the registered owner.",
    "The stolen securities were recovered by law enforcement.",
    "The stolen property was inventoried and returned to the claimant.",
    "The arrest record was expunged pursuant to court order.",
]


# ---------------------------------------------------------------------------
# GLiNER example builder
# ---------------------------------------------------------------------------

def make_gliner_example(text: str, entity_value: str, label: str) -> dict | None:
    """Build a GLiNER training record with whitespace-token indices."""
    tokens = text.split()
    ev_tokens = entity_value.split()
    if not ev_tokens:
        return None
    for i in range(len(tokens) - len(ev_tokens) + 1):
        if tokens[i: i + len(ev_tokens)] == ev_tokens:
            return {"tokenized_text": tokens, "ner": [[i, i + len(ev_tokens) - 1, label]]}
    return None


# ---------------------------------------------------------------------------
# Multi-entity document generation — 8 realistic document types
# ---------------------------------------------------------------------------

def _find_spans(tokens: list[str], candidates: list[tuple[str, str]]) -> list[list]:
    spans: list[list] = []
    used: set[int] = set()
    for val, lbl in candidates:
        if not val:
            continue
        ev = val.split()
        for i in range(len(tokens) - len(ev) + 1):
            if tokens[i: i + len(ev)] == ev:
                pos = set(range(i, i + len(ev)))
                if not pos & used:
                    spans.append([i, i + len(ev) - 1, lbl])
                    used |= pos
                break
    return spans


def generate_multi_entity_documents(
    entity_defs: dict,
    gliner_label_map: dict,
    n: int,
    rng: random.Random,
) -> list[dict]:

    def _g(eid: str) -> str:
        d = entity_defs.get(eid)
        return d["generator"]() if d else ""

    def _lbl(eid: str) -> str:
        return gliner_label_map.get(eid, "")

    templates = [
        # 1. Medical record
        lambda: (
            f"Patient Name: {_g('person_name')} | DOB: {_g('date_of_birth')} | "
            f"SSN: {_g('ssn')} | MRN: {_g('medical_record_number')} | "
            f"Phone: {_g('phone_number')} | Medication: {_g('medication_name')} | "
            f"Facility: {_g('hospital_name')} | Member ID: {_g('health_plan_beneficiary_number')} | "
            f"Service Date: {_g('clinical_date')}",
            ["person_name","date_of_birth","ssn","medical_record_number","phone_number",
             "medication_name","hospital_name","health_plan_beneficiary_number","clinical_date"]
        ),
        # 2. Financial record
        lambda: (
            f"Name: {_g('person_name')} | Account Number: {_g('bank_account_number')} | "
            f"Routing Number: {_g('bank_routing_number')} | Card Number: {_g('credit_card_number')} | "
            f"Expiry: {_g('card_expiration_date')} | Bank: {_g('bank_name')} | "
            f"Transaction ID: {_g('transaction_id')} | Amount: {_g('financial_amount')} | "
            f"SSN: {_g('ssn')}",
            ["person_name","bank_account_number","bank_routing_number","credit_card_number",
             "card_expiration_date","bank_name","transaction_id","financial_amount","ssn"]
        ),
        # 3. Law enforcement record
        lambda: (
            f"Suspect: {_g('person_name')} | DOB: {_g('date_of_birth')} | SSN: {_g('ssn')} | "
            f"Case Number: {_g('case_number')} | Arrest Record: {_g('arrest_record')} | "
            f"Court: {_g('court_name')} | Warrant Number: {_g('warrant_data')} | "
            f"Incident Report {_g('incident_report')} | Driver's License: {_g('drivers_license')}",
            ["person_name","date_of_birth","ssn","case_number","arrest_record",
             "court_name","warrant_data","incident_report","drivers_license"]
        ),
        # 4. HR/Employee record
        lambda: (
            f"Employee: {_g('person_name')} | Employee ID: {_g('employee_id')} | "
            f"SSN: {_g('ssn')} | Email: {_g('email_address')} | Phone: {_g('phone_number')} | "
            f"Employment: {_g('employment_history')} | Tax ID: {_g('tax_id_number')} | "
            f"Performance rating: {_g('performance_evaluation')}",
            ["person_name","employee_id","ssn","email_address","phone_number",
             "employment_history","tax_id_number","performance_evaluation"]
        ),
        # 5. Travel record
        lambda: (
            f"Traveler: {_g('person_name')} | Passport Number: {_g('passport_number')} | "
            f"Flight {_g('flight_number')} | Booking Reference: {_g('booking_reference')} | "
            f"Hotel: {_g('hotel_name')} | Email: {_g('email_address')} | "
            f"Phone: {_g('phone_number')}",
            ["person_name","passport_number","flight_number","booking_reference",
             "hotel_name","email_address","phone_number"]
        ),
        # 6. Insurance claim
        lambda: (
            f"Claimant: {_g('person_name')} | Claim Number: {_g('claim_number')} | "
            f"Policy Number: {_g('insurance_policy_number')} | "
            f"Insurance company: {_g('insurance_company_name')} | SSN: {_g('ssn')} | "
            f"MRN: {_g('medical_record_number')} | Service Date: {_g('clinical_date')} | "
            f"Amount: {_g('financial_amount')}",
            ["person_name","claim_number","insurance_policy_number","insurance_company_name",
             "ssn","medical_record_number","clinical_date","financial_amount"]
        ),
        # 7. Student record
        lambda: (
            f"Student: {_g('person_name')} | Student ID: {_g('student_id')} | "
            f"University: {_g('university_name')} | Email: {_g('email_address')} | "
            f"DOB: {_g('date_of_birth')} | SSN: {_g('ssn')} | "
            f"Student record: {_g('student_records_ferpa')}",
            ["person_name","student_id","university_name","email_address",
             "date_of_birth","ssn","student_records_ferpa"]
        ),
        # 8. Payment transaction
        lambda: (
            f"Card Number: {_g('credit_card_number')} | card ending in {_g('card_last4')} | "
            f"Card type: {_g('card_type')} | Expiry: {_g('card_expiration_date')} | "
            f"Merchant ID: {_g('merchant_id')} | Terminal ID: {_g('terminal_id')} | "
            f"Transaction ID: {_g('transaction_id')} | Amount: {_g('financial_amount')}",
            ["credit_card_number","card_last4","card_type","card_expiration_date",
             "merchant_id","terminal_id","transaction_id","financial_amount"]
        ),
    ]

    results: list[dict] = []
    for _ in range(n):
        tmpl_fn = rng.choice(templates)
        try:
            text, eids = tmpl_fn()
        except Exception:
            continue

        # Re-generate to get consistent values (tmpl_fn already called _g() inline)
        # We need to extract values from the generated text — find each entity's value
        # from a fresh generation matching text
        tokens = text.split()
        # Build candidates from a re-scan: find labels whose values appear in text
        # Since values were inlined in the template, we need to regenerate with tracking
        # Use the simpler approach: parse structured text (field: value | field: value)
        candidate_spans: list[tuple[str, str]] = []
        for eid in eids:
            lbl = _lbl(eid)
            if not lbl:
                continue
            d = entity_defs.get(eid)
            if not d:
                continue
            # Try multiple generated values to find one that matches a subsequence
            for _ in range(5):
                try:
                    val = d["generator"]()
                except Exception:
                    continue
                val_toks = val.split()
                for i in range(len(tokens) - len(val_toks) + 1):
                    if tokens[i: i + len(val_toks)] == val_toks:
                        candidate_spans.append((val, lbl))
                        break

        spans = _find_spans(tokens, candidate_spans)
        if spans:
            results.append({"tokenized_text": tokens, "ner": spans})

    # Fallback: generate structured field:value pairs for guaranteed span matching
    field_templates = [
        ("person_name", "Patient Name: {v}"),
        ("ssn", "SSN: {v}"),
        ("date_of_birth", "DOB: {v}"),
        ("medical_record_number", "MRN: {v}"),
        ("credit_card_number", "Card Number: {v}"),
        ("bank_account_number", "Account Number: {v}"),
        ("email_address", "Email: {v}"),
        ("phone_number", "Phone: {v}"),
        ("employee_id", "Employee ID: {v}"),
        ("case_number", "Case Number: {v}"),
    ]

    needed = n - len(results)
    if needed > 0:
        for _ in range(needed):
            chosen = rng.sample(field_templates, k=min(4, len(field_templates)))
            parts = []
            candidates2: list[tuple[str, str]] = []
            for eid, tmpl in chosen:
                d = entity_defs.get(eid)
                lbl = _lbl(eid)
                if not d or not lbl:
                    continue
                try:
                    val = d["generator"]()
                    parts.append(tmpl.format(v=val))
                    candidates2.append((val, lbl))
                except Exception:
                    continue
            if not parts:
                continue
            text2 = " | ".join(parts)
            tokens2 = text2.split()
            spans2 = _find_spans(tokens2, candidates2)
            if spans2:
                results.append({"tokenized_text": tokens2, "ner": spans2})

    return results


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(
    entities_yaml: Path,
    out_path: Path,
    samples_per_entity: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    fake.seed_instance(seed)

    with open(entities_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    gliner_label_map: dict[str, str] = {}
    for raw in data.get("entities", []):
        if not raw.get("enabled", True):
            continue
        eid = raw["id"]
        gliner_label_map[eid] = raw.get("gliner_label") or raw.get("display_name", eid)

    all_examples: list[dict] = []
    entity_counts: dict[str, int] = {}

    for entity_id, defn in ENTITY_DEFS.items():
        label = gliner_label_map.get(entity_id)
        if label is None:
            continue

        generator: Callable = defn["generator"]
        templates: list[str] = defn["templates"]
        generated = 0

        for _ in range(samples_per_entity * 4):
            if generated >= samples_per_entity:
                break
            try:
                value = generator()
            except Exception:
                continue
            template = rng.choice(templates)
            try:
                text = template.format(value=value)
            except (KeyError, IndexError):
                continue
            ex = make_gliner_example(text, value, label)
            if ex is None:
                continue
            all_examples.append(ex)
            generated += 1

        entity_counts[entity_id] = generated

    # Auto-fallback: generate data for yaml entities not in ENTITY_DEFS.
    # Any new entity added to entities_config.yaml gets covered automatically
    # using generic field-value templates. No code change required.
    auto_covered: list[str] = []
    for entity_id in gliner_label_map:
        if entity_id in ENTITY_DEFS:
            continue  # already handled above
        label = gliner_label_map[entity_id]
        display = label

        fallback_templates = [
            f"{display}: {{value}}",
            f"The {display} is {{value}}.",
            f"Patient {display}: {{value}} is on record.",
            f"Please verify {display}: {{value}} before proceeding.",
            f"Record shows {display} as {{value}}.",
            f"Note: {display} = {{value}}",
            f"The patient's {display} is listed as {{value}}.",
            f"Document {display}: {{value}}",
            f"Verify {display} {{value}} with the issuing authority.",
            f"Field {display} contains {{value}}.",
            f"System: {display} updated to {{value}}.",
            f"The {display} on file is {{value}}.",
            f"Claim {display}: {{value}} — pending review.",
            f"Reference {display} {{value}} when contacting support.",
            f"Authorization required for {display} {{value}}.",
        ]

        def _fallback_value(r: random.Random) -> str:
            kind = r.choice(["hex", "num", "alpha", "mixed"])
            if kind == "hex":
                return "".join(r.choices("ABCDEF0123456789", k=r.randint(6, 12)))
            elif kind == "num":
                return str(r.randint(100000, 99999999))
            elif kind == "alpha":
                return "".join(r.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=r.randint(4, 8)))
            else:
                prefix = "".join(r.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
                return f"{prefix}-{r.randint(10000, 999999)}"

        generated = 0
        for _ in range(samples_per_entity * 4):
            if generated >= samples_per_entity:
                break
            value = _fallback_value(rng)
            template = rng.choice(fallback_templates)
            text = template.format(value=value)
            ex = make_gliner_example(text, value, label)
            if ex is None:
                continue
            all_examples.append(ex)
            generated += 1

        entity_counts[entity_id] = generated
        auto_covered.append(entity_id)

    if auto_covered:
        print(f"  [auto-fallback] {len(auto_covered)} new entities covered: {', '.join(auto_covered)}")

    # Hard negatives
    for sentence in HARD_NEGATIVES:
        all_examples.append({"tokenized_text": sentence.split(), "ner": []})

    # Multi-entity documents
    multi = generate_multi_entity_documents(ENTITY_DEFS, gliner_label_map, n=10000, rng=rng)
    all_examples.extend(multi)

    rng.shuffle(all_examples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    total = len(all_examples)
    single = total - len(HARD_NEGATIVES) - len(multi)
    print(f"\nGenerated {total:,} training examples")
    print(f"  Single-entity:  {single:,}")
    print(f"  Hard negatives: {len(HARD_NEGATIVES):,}")
    print(f"  Multi-entity:   {len(multi):,}")
    print(f"  Saved to: {out_path}")
    print("\nEntity coverage:")
    covered = sum(1 for c in entity_counts.values() if c >= samples_per_entity * 0.8)
    print(f"  {covered}/{len(entity_counts)} entities at >=80% target")
    for eid, cnt in sorted(entity_counts.items()):
        mark = "OK" if cnt >= samples_per_entity * 0.8 else "!!"
        print(f"  [{mark}] {eid}: {cnt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GLiNER PII training data")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Target examples per entity (default: 1000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--yaml", type=Path, default=YAML_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.yaml, args.out, args.samples, args.seed)


if __name__ == "__main__":
    main()
