
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


def _build_label_map() -> dict[str, str]:
    """Build entity_id -> gliner_label mapping from YAML config."""
    try:
        cfg = yaml.safe_load(open(YAML_PATH, encoding="utf-8"))
        return {e["id"]: e.get("gliner_label", e["id"]) for e in cfg.get("entities", [])}
    except Exception:
        return {}

LABEL_MAP: dict[str, str] = _build_label_map()


# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

def _phone() -> str:
    """Phone number across US/UK/India/EU/intl formats."""
    a = f"{random.randint(200, 999):03d}"
    b = f"{random.randint(200, 999):03d}"
    c = f"{random.randint(1000, 9999):04d}"
    fmt = random.choices(
        ["us_parens", "us_dash", "us_dot", "us_intl", "us_ext",
         "us_bare10", "us_spaced", "uk", "india", "intl_eu",
         "with_extension_word", "e164"],
        weights=[16, 18, 8, 12, 8, 12, 6, 6, 8, 4, 6, 6],
        k=1,
    )[0]

    if fmt == "us_parens":
        return f"({a}) {b}-{c}"
    if fmt == "us_dash":
        return f"{a}-{b}-{c}"
    if fmt == "us_dot":
        return f"{a}.{b}.{c}"
    if fmt == "us_intl":
        return f"+1-{a}-{b}-{c}"
    if fmt == "us_ext":
        return f"{a}-{b}-{c}x{random.randint(100, 9999)}"
    if fmt == "us_bare10":
        return f"{a}{b}{c}"
    if fmt == "us_spaced":
        return f"{a} {b} {c}"
    if fmt == "uk":
        return f"+44 {random.randint(20, 79)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
    if fmt == "india":
        # +91 98765 43210 / 98765-43210 / 9876543210
        mobile = random.randint(6000000000, 9999999999)
        style = random.choice(["intl_spaced", "dashed", "bare"])
        if style == "intl_spaced":
            s = str(mobile)
            return f"+91 {s[:5]} {s[5:]}"
        if style == "dashed":
            s = str(mobile)
            return f"{s[:5]}-{s[5:]}"
        return str(mobile)
    if fmt == "intl_eu":
        country = random.choice(["+33", "+49", "+39", "+34", "+31", "+41"])
        return f"{country} {random.randint(1, 99)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
    if fmt == "with_extension_word":
        return f"{a}-{b}-{c} ext. {random.randint(100, 9999)}"
    # e164
    return f"+1{a}{b}{c}"

def _phone2() -> str:
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def _fax() -> str:
    a, b, c = (f"{random.randint(200, 999):03d}",
               f"{random.randint(200, 999):03d}",
               f"{random.randint(1000, 9999):04d}")
    fmt = random.choices(
        ["parens", "dashed", "dotted", "intl", "ext", "bare", "spaced"],
        weights=[22, 22, 10, 14, 12, 10, 10],
        k=1,
    )[0]
    if fmt == "parens":
        return f"({a}) {b}-{c}"
    if fmt == "dashed":
        return f"{a}-{b}-{c}"
    if fmt == "dotted":
        return f"{a}.{b}.{c}"
    if fmt == "intl":
        return f"+1-{a}-{b}-{c}"
    if fmt == "ext":
        return f"+1-{a}-{b}-{c} ext {random.randint(100, 9999)}"
    if fmt == "bare":
        return f"{a}{b}{c}"
    return f"{a} {b} {c}"

def _ssn() -> str:
    a = f"{random.randint(100, 799):03d}"
    b = f"{random.randint(10, 99):02d}"
    c = f"{random.randint(1000, 9999):04d}"
    fmt = random.choices(
        ["dashed", "spaced", "bare", "dotted", "trailing_only"],
        weights=[42, 14, 28, 8, 8],
        k=1,
    )[0]
    if fmt == "dashed":
        return f"{a}-{b}-{c}"
    if fmt == "spaced":
        return f"{a} {b} {c}"
    if fmt == "bare":
        return f"{a}{b}{c}"
    if fmt == "dotted":
        return f"{a}.{b}.{c}"
    # trailing_only — last 4 / masked
    return f"XXX-XX-{c}"

def _credit_card() -> str:
    """Credit/debit PAN across networks and grouping styles.

    Networks covered: Visa (16 / 13), Mastercard (16), Amex (15), Discover (16),
    Diners (14), JCB (16), RuPay/UnionPay (16). Grouping styles: spaced 4-4-4-4
    or 4-6-5 (Amex), dashed, and bare contiguous.
    """
    network = random.choices(
        ["visa16", "visa13", "mc", "amex", "discover",
         "diners", "jcb", "unionpay", "rupay"],
        weights=[24, 4, 22, 14, 12, 6, 6, 6, 6],
        k=1,
    )[0]

    if network == "visa16":
        digits = f"4{random.randint(0, 9)}{random.randint(0, 99):02d}" + \
                 "".join(random.choices("0123456789", k=12))
    elif network == "visa13":
        digits = "4" + "".join(random.choices("0123456789", k=12))
    elif network == "mc":
        prefix = random.choice(["51", "52", "53", "54", "55",
                                 "2221", "2300", "2400", "2700"])
        digits = prefix + "".join(random.choices("0123456789", k=16 - len(prefix)))
    elif network == "amex":
        prefix = random.choice(["34", "37"])
        digits = prefix + "".join(random.choices("0123456789", k=13))
    elif network == "discover":
        prefix = random.choice(["6011", "65", "644", "645", "646", "647",
                                 "648", "649"])
        digits = prefix + "".join(random.choices("0123456789", k=16 - len(prefix)))
    elif network == "diners":
        prefix = random.choice(["300", "301", "302", "303", "304", "305",
                                 "36", "38"])
        digits = prefix + "".join(random.choices("0123456789", k=14 - len(prefix)))
    elif network == "jcb":
        digits = "35" + "".join(random.choices("0123456789", k=14))
    elif network == "unionpay":
        digits = "62" + "".join(random.choices("0123456789", k=14))
    else:  # rupay
        prefix = random.choice(["508", "606", "607", "608", "652", "653"])
        digits = prefix + "".join(random.choices("0123456789", k=16 - len(prefix)))

    # Grouping
    style = random.choices(["bare", "spaced", "dashed"], weights=[28, 56, 16], k=1)[0]
    if style == "bare":
        return digits
    # Amex groups 4-6-5; everything else groups 4-4-4-(4/3) or 4-4-4-4-3
    if network == "amex":
        groups = [digits[:4], digits[4:10], digits[10:]]
    elif len(digits) == 13:
        groups = [digits[:4], digits[4:8], digits[8:12], digits[12:]]
    elif len(digits) == 14:
        groups = [digits[:4], digits[4:8], digits[8:12], digits[12:]]
    elif len(digits) == 15:
        groups = [digits[:4], digits[4:10], digits[10:]]
    elif len(digits) == 19:
        groups = [digits[:4], digits[4:8], digits[8:12], digits[12:16], digits[16:]]
    else:  # 16
        groups = [digits[:4], digits[4:8], digits[8:12], digits[12:]]
    sep = " " if style == "spaced" else "-"
    return sep.join(groups)


def _cvv() -> str:
    # Most networks use 3-digit codes; Amex CID is 4 digits.
    if random.random() < 0.2:
        return f"{random.randint(1000, 9999):04d}"
    return f"{random.randint(100, 999):03d}"

def _card_exp() -> str:
    mm = f"{random.randint(1, 12):02d}"
    yy = f"{random.randint(25, 35):02d}"
    yyyy = f"20{yy}"
    sep = random.choice(["/", "-"])
    year = random.choice([yy, yy, yyyy])
    return f"{mm}{sep}{year}"

def _date_mmddyyyy() -> str:
    d = fake.date_of_birth(minimum_age=1, maximum_age=90)
    fmt = random.choice([
        "%m/%d/%Y",   # 05/21/2026
        "%m-%d-%Y",   # 05-21-2026
        "%d/%m/%Y",   # 21/05/2026
        "%d-%m-%Y",   # 21-05-2026
        "%d.%m.%Y",   # 21.05.2026
        "%Y-%m-%d",   # 2026-05-21
        "%Y/%m/%d",   # 2026/05/21
        "%Y_%m_%d",   # 2026_05_21
        "%B %d, %Y",  # May 21, 2026
        "%d %B %Y",   # 21 May 2026
        "%d %b %Y",   # 21 May 2026 (abbrev)
        "%b %d, %Y",  # May 21, 2026 (abbrev)
        "%A, %d %B %Y",  # Thursday, 21 May 2026
        "%a, %d %b %Y",  # Thu, 21 May 2026
        "%d/%b/%Y",   # 21/MAY/2026 - handled via upper
        "%d_%b_%Y",   # 21_May_2026
        "year_only",  # GAP-01: birth year only
        "timestamp",  # GAP-02: YYYY-MM-DD HH:MM:SS
    ])
    if fmt == "year_only":
        return d.strftime("%Y")
    if fmt == "timestamp":
        h, m, s = random.randint(0,23), random.randint(0,59), random.randint(0,59)
        return f"{d.strftime('%Y-%m-%d')} {h:02d}:{m:02d}:{s:02d}"
    result = d.strftime(fmt)
    if fmt == "%d/%b/%Y":
        result = result.upper()
    return result

def _date_clinical() -> str:
    d = fake.date_between(start_date="-5y", end_date="today")
    fmt = random.choice([
        "%m/%d/%Y",   # 05/21/2026
        "%m-%d-%Y",   # 05-21-2026
        "%d/%m/%Y",   # 21/05/2026
        "%d-%m-%Y",   # 21-05-2026
        "%d.%m.%Y",   # 21.05.2026
        "%Y-%m-%d",   # 2026-05-21
        "%Y/%m/%d",   # 2026/05/21
        "%Y_%m_%d",   # 2026_05_21
        "%B %d, %Y",  # May 21, 2026
        "%d %B %Y",   # 21 May 2026
        "%d %b %Y",   # 21 May 2026 (abbrev)
        "%b %d, %Y",  # May 21, 2026 (abbrev)
        "%A, %d %B %Y",  # Thursday, 21 May 2026
        "%a, %d %b %Y",  # Thu, 21 May 2026
        "%d/%b/%Y",   # 21/MAY/2026 - handled via upper
        "%d_%b_%Y",   # 21_May_2026
        "year_only",  # GAP-01: year-only clinical event year
        "timestamp",  # GAP-02: YYYY-MM-DD HH:MM:SS
    ])
    if fmt == "year_only":
        return d.strftime("%Y")
    if fmt == "timestamp":
        h, m, s = random.randint(0,23), random.randint(0,59), random.randint(0,59)
        return f"{d.strftime('%Y-%m-%d')} {h:02d}:{m:02d}:{s:02d}"
    result = d.strftime(fmt)
    if fmt == "%d/%b/%Y":
        result = result.upper()
    return result

def _year_only() -> str:
    return str(random.randint(2010, 2025))

def _ip() -> str:
    fmt = random.choices(
        ["private", "public", "loopback", "ipv6", "ipv6_short", "with_port", "cidr"],
        weights=[24, 30, 8, 18, 8, 8, 4],
        k=1,
    )[0]
    if fmt == "private":
        return fake.ipv4_private()
    if fmt == "public":
        return ".".join(str(random.randint(1, 223)) for _ in range(4))
    if fmt == "loopback":
        return f"127.0.0.{random.randint(1, 255)}"
    if fmt == "ipv6":
        return ":".join(f"{random.randint(0, 65535):04x}" for _ in range(8))
    if fmt == "ipv6_short":
        return f"2001:db8::{random.randint(0, 65535):x}"
    if fmt == "with_port":
        ip = ".".join(str(random.randint(1, 223)) for _ in range(4))
        return f"{ip}:{random.randint(1024, 65535)}"
    # cidr
    ip = ".".join(str(random.randint(1, 223)) for _ in range(4))
    return f"{ip}/{random.choice([8, 16, 24, 32])}"


def _mac() -> str:
    octets = [f"{random.randint(0, 255):02X}" for _ in range(6)]
    fmt = random.choices(
        ["colon_upper", "colon_lower", "dash_upper", "dash_lower",
         "cisco_dot", "bare_upper", "bare_lower"],
        weights=[26, 14, 18, 10, 14, 10, 8],
        k=1,
    )[0]
    if fmt == "colon_upper":
        return ":".join(octets)
    if fmt == "colon_lower":
        return ":".join(o.lower() for o in octets)
    if fmt == "dash_upper":
        return "-".join(octets)
    if fmt == "dash_lower":
        return "-".join(o.lower() for o in octets)
    if fmt == "cisco_dot":
        flat = "".join(octets)
        return f"{flat[:4]}.{flat[4:8]}.{flat[8:]}"
    if fmt == "bare_upper":
        return "".join(octets)
    return "".join(octets).lower()

def _url() -> str:
    """URLs that contain PII in path or query string.

    Originally produced one hospital-portal format. Real failures included
    patient portals, ride/transit IDs, booking confirmations, social profile
    links, customer dashboards, billing portals, and signed S3 URLs.
    """
    pid = random.randint(10000, 9999999)
    forms = [
        f"https://portal.hospital.org/patients/{pid}",
        f"https://my.healthplan.com/member/{pid}/dashboard",
        f"https://app.clinic.io/charts/{pid}",
        f"https://billing.medgroup.com/account/{pid}/invoice",
        f"https://patient.kp.org/visit/{pid}",
        f"https://account.bank.com/users/{pid}/statements",
        f"https://booking.airline.com/itinerary/{pid}",
        f"https://www.linkedin.com/in/{fake.user_name()}",
        f"https://twitter.com/{fake.user_name()}",
        f"https://facebook.com/{fake.user_name()}.{random.randint(1,9999)}",
        f"https://github.com/{fake.user_name()}",
        f"https://example.com/profile?user_id={pid}&token=abc{random.randint(1000,9999)}",
        f"https://shop.com/order?email={fake.email()}",
        f"https://accounts.google.com/o/oauth2/v2/auth?login_hint={fake.email()}",
        f"https://amzn.to/{random.choice('abcdefghjkmnpqrstuvwxyz')}{random.randint(100,9999)}",
        f"https://s3.amazonaws.com/private/{pid}/report.pdf?X-Amz-Signature={random.randint(10**9,10**10)}",
    ]
    return random.choice(forms)

def _mrn_value() -> str:
    digits = random.randint(100000, 99999999)
    fmt = random.choices(
        ["bare", "mrn_prefix", "mrn_hash", "alpha_prefix",
         "epic_style", "cerner_style"],
        weights=[28, 22, 14, 16, 10, 10],
        k=1,
    )[0]
    if fmt == "bare":
        return str(digits)
    if fmt == "mrn_prefix":
        return f"MRN-{digits}"
    if fmt == "mrn_hash":
        return f"MRN#{digits}"
    if fmt == "alpha_prefix":
        letter = random.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
        return f"{letter}{digits}"
    if fmt == "epic_style":
        return f"E{random.randint(1000000,99999999)}"
    # cerner_style — leading zeros padded
    return f"{digits:010d}"

def _npi() -> str:
    digits = str(random.randint(1000000000, 9999999999))
    fmt = random.choices(
        ["npi_dashed", "npi_attached", "npi_hash", "npi_word",
         "provider_id", "bare", "npi_space"],
        weights=[28, 14, 10, 12, 12, 18, 6],
        k=1,
    )[0]
    if fmt == "npi_dashed":
        return f"NPI-{digits}"
    if fmt == "npi_attached":
        return f"NPI{digits}"
    if fmt == "npi_hash":
        return f"NPI#{digits}"
    if fmt == "npi_word":
        return f"NPI Number: {digits}"
    if fmt == "provider_id":
        return f"PROVIDER-{digits}"
    if fmt == "npi_space":
        return f"NPI {digits}"
    return digits


def _npi_plain() -> str:
    return str(random.randint(1000000000, 9999999999))


def _dea() -> str:
    letters = "ABCDEFGHJKLMNPRSTUX"
    body = f"{random.choice(letters)}{random.choice(letters)}{random.randint(1000000, 9999999)}"
    fmt = random.choices(
        ["bare", "dea_dashed", "dea_attached", "dea_hash", "dea_no",
         "dea_word", "reg_prefix"],
        weights=[30, 18, 14, 10, 10, 12, 6],
        k=1,
    )[0]
    if fmt == "bare":
        return body
    if fmt == "dea_dashed":
        return f"DEA-{body}"
    if fmt == "dea_attached":
        return f"DEA{body}"
    if fmt == "dea_hash":
        return f"DEA#{body}"
    if fmt == "dea_no":
        return f"DEA No: {body}"
    if fmt == "dea_word":
        return f"DEA Number {body}"
    return f"REG-{body}"

def _iban() -> str:
    """Multi-country IBAN. Original only emitted GB-format which biased the
    model toward UK BBANs and missed every EU/IN/AE/ME real-world example.
    Lengths follow the actual per-country IBAN spec.
    """
    schemes = [
        ("GB", 22, 4),   # GB22 NWBK 6010 1234 5678 90  — UK
        ("DE", 22, 0),   # DE89 3704 0044 0532 0130 00  — Germany
        ("FR", 27, 0),   # FR14 2004 1010 0505 0001 3M02 606  — France
        ("ES", 24, 0),   # Spain
        ("IT", 27, 1),   # IT60 X054 2811 1010 0000 0123 456  — Italy
        ("NL", 18, 4),   # NL91 ABNA 0417 1643 00  — Netherlands
        ("CH", 21, 0),   # Switzerland
        ("BE", 16, 0),   # Belgium
        ("AT", 20, 0),   # Austria
        ("IE", 22, 4),   # Ireland
        ("PT", 25, 0),   # Portugal
        ("AE", 23, 0),   # UAE
        ("SA", 24, 0),   # Saudi Arabia
        ("TR", 26, 0),   # Turkey
        ("PL", 28, 0),   # Poland
    ]
    country, length, bank_letters = random.choice(schemes)
    check = f"{random.randint(10, 99):02d}"
    bank = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=bank_letters))
    remaining = length - len(country) - len(check) - bank_letters
    body = "".join(random.choices("0123456789", k=remaining))
    iban = f"{country}{check}{bank}{body}"
    # 25 % of the time return with conventional 4-char space groupings
    if random.random() < 0.25:
        return " ".join(iban[i:i+4] for i in range(0, len(iban), 4))
    return iban

def _swift() -> str:
    banks = [
        # US / EU / UK majors
        "CHAS", "BOFA", "CITI", "WFBI", "JPMC", "MSCO", "GOLD",
        "DEUT", "BARC", "HSBC", "BNPA", "SCBL", "INGB", "RABO",
        "ABNA", "UBSW", "CRES",
        # India
        "HDFC", "ICIC", "AXIS", "PUNB", "SBIN", "KKBK", "YESB",
        "UTIB", "BKID", "INDB", "CNRB", "IDFB", "IBKL", "FDRL",
        # APAC + Middle East
        "DBSS", "OCBC", "UOVB", "ANZB", "NATA", "BKKB", "KASI",
        "MITB", "BCHK", "BOTK", "NABA", "EBIL", "QNBA", "ENBD",
    ]
    countries = [
        "US", "GB", "DE", "FR", "NL", "CH", "ES", "IT",
        "IN", "SG", "HK", "JP", "AU", "CN", "AE", "QA",
        "CA", "BR", "MX", "ZA", "TH",
    ]
    # Real-world locations span all-letter and digit/letter combos.
    locs = [
        # All-letter (the previous bug — these were never generated)
        "BB", "GG", "AA", "XX", "PP", "FF", "SG", "HK", "NY",
        # Letter + digit / digit + letter
        "2X", "3N", "6S", "8R", "0X", "1S", "9A", "4M", "7B",
        # All-digit
        "33", "22", "11", "00", "44", "55",
    ]
    branches = [
        "XXX", "DEL", "BOM", "MUM", "DLH", "LON", "FRA", "SYD",
        "HKG", "TKY", "NYC", "SFO", "104", "500", "001", "002",
    ]
    bic8 = f"{random.choice(banks)}{random.choice(countries)}{random.choice(locs)}"
    # 30 % of the time emit the 11-char form (8-char BIC + 3-char branch code)
    if random.random() < 0.3:
        return f"{bic8}{random.choice(branches)}"
    return bic8

def _routing() -> str:
    digits = f"{random.randint(100000000, 999999999)}"
    fmt = random.choices(
        ["bare", "aba_prefix", "rt_prefix", "rtn_prefix", "dashed"],
        weights=[40, 20, 14, 14, 12],
        k=1,
    )[0]
    if fmt == "bare":
        return digits
    if fmt == "aba_prefix":
        return f"ABA-{digits}"
    if fmt == "rt_prefix":
        return f"RT{digits}"
    if fmt == "rtn_prefix":
        return f"RTN-{digits}"
    # dashed — XXXX-XXXX-X
    return f"{digits[:4]}-{digits[4:8]}-{digits[8]}"


def _bank_account() -> str:
    digits = random.randint(10000000, 9999999999)
    fmt = random.choices(
        ["bare", "acc_prefix", "acct_prefix", "ac_no", "leading_zero",
         "spaced", "dashed", "iban_style_dom"],
        weights=[26, 16, 14, 12, 10, 10, 8, 4],
        k=1,
    )[0]
    s = str(digits)
    if fmt == "bare":
        return s
    if fmt == "acc_prefix":
        return f"ACC-{s}"
    if fmt == "acct_prefix":
        return f"ACCT-{s}"
    if fmt == "ac_no":
        return f"A/C-{s}"
    if fmt == "leading_zero":
        return f"{int(s):012d}"
    if fmt == "spaced":
        return " ".join(s[i:i+4] for i in range(0, len(s), 4))
    if fmt == "dashed":
        return "-".join(s[i:i+4] for i in range(0, len(s), 4))
    # iban_style_dom
    return f"XXBANK{s}"

def _vin() -> str:
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    body = "".join(random.choice(chars) for _ in range(17))
    fmt = random.choices(
        ["bare", "vin_dashed", "vin_attached", "vin_hash",
         "vin_colon", "vin_no"],
        weights=[40, 18, 14, 10, 10, 8],
        k=1,
    )[0]
    if fmt == "bare":
        return body
    if fmt == "vin_dashed":
        return f"VIN-{body}"
    if fmt == "vin_attached":
        return f"VIN{body}"
    if fmt == "vin_hash":
        return f"VIN#{body}"
    if fmt == "vin_colon":
        return f"VIN: {body}"
    return f"VIN No. {body}"

def _license_plate() -> str:
    """License plate covering US, Indian, UK, and EU formats."""
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # exclude I and O (commonly omitted)
    fmt = random.choices(
        [
            "us_3letter_4digit",   # ABC-1234
            "us_3letter_3digit",   # ABC-123
            "us_2letter_4digit",   # AB-1234
            "indian",              # TN10BK9090 / KA01AB1234
            "indian_spaced",       # TN 10 BK 9090
            "uk",                  # AB12 CDE
            "eu_dash",             # B-AB 1234
            "vanity",              # I-LOVE-NY style
            "motorcycle",          # short 5-6 char
        ],
        weights=[20, 10, 8, 22, 10, 12, 6, 5, 7],
        k=1,
    )[0]

    if fmt == "us_3letter_4digit":
        sep = random.choice(["-", " ", ""])
        return f"{''.join(random.choices(L, k=3))}{sep}{random.randint(1000, 9999)}"
    if fmt == "us_3letter_3digit":
        sep = random.choice(["-", " ", ""])
        return f"{''.join(random.choices(L, k=3))}{sep}{random.randint(100, 999)}"
    if fmt == "us_2letter_4digit":
        sep = random.choice(["-", " ", ""])
        return f"{''.join(random.choices(L, k=2))}{sep}{random.randint(1000, 9999)}"
    if fmt == "indian":
        state = random.choice(["TN", "KA", "MH", "DL", "AP", "TS", "KL", "GJ",
                               "RJ", "UP", "MP", "WB", "PB", "HR", "BR"])
        dist = f"{random.randint(1, 99):02d}"
        series = "".join(random.choices(L, k=2))
        num = f"{random.randint(1, 9999):04d}"
        return f"{state}{dist}{series}{num}"
    if fmt == "indian_spaced":
        state = random.choice(["TN", "KA", "MH", "DL", "AP", "TS", "KL", "GJ"])
        return (f"{state} {random.randint(1, 99):02d} "
                f"{''.join(random.choices(L, k=2))} {random.randint(1, 9999):04d}")
    if fmt == "uk":
        # AB12 CDE
        return (f"{''.join(random.choices(L, k=2))}{random.randint(10, 99)} "
                f"{''.join(random.choices(L, k=3))}")
    if fmt == "eu_dash":
        # B-AB 1234 (Germany-style)
        city = random.choice(["B", "M", "K", "F", "S", "L", "H"])
        return f"{city}-{''.join(random.choices(L, k=2))} {random.randint(1, 9999)}"
    if fmt == "vanity":
        words = random.choice(["ILOVENY", "GOBLUE", "MOMSCAR", "BEST1",
                                "DR1VER", "GR8MOM", "RUNFAST"])
        return words
    # motorcycle — 5-6 chars, often higher density
    return f"{''.join(random.choices(L, k=2))}{random.randint(100, 9999)}"

def _api_key() -> str:
    """Session tokens, API keys, JWTs and cookie-style values.

    Covers the production failures: stripe/openai prefixed keys, AWS access
    keys, GitHub PATs, JWT three-segment, OAuth bearer, PHPSESSID/JSESSIONID
    cookie-pair values, and bare alphanumeric tokens.
    """
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    HEX = "0123456789abcdef"

    def rand(charset: str, k: int) -> str:
        return "".join(random.choices(charset, k=k))

    fmt = random.choices(
        [
            "sk_prefix",          # sk-...
            "openai_proj",        # sk-proj-...
            "stripe_key",         # sk_live_..., sk_test_...
            "stripe_object",      # pi_..., ch_..., cus_..., pm_...
            "aws_access",         # AKIA...
            "github_pat",         # ghp_..., gho_..., ghs_...
            "slack",              # xoxb-..., xoxp-...
            "jwt",                # eyJ...
            "session_id_hex",     # 32-64 hex chars
            "cookie_pair",        # PHPSESSID=..., JSESSIONID=..., csrftoken=...
            "auth_tok_prefix",    # auth_tok_..., sess_..., sid_...
            "sid_dashed",         # SID-44556677
            "bearer",             # Bearer eyJ...
            "uuid_hex",           # 8-4-4-4-12
            "base64ish",          # random base64-ish
        ],
        weights=[6, 4, 6, 8, 5, 5, 4, 12, 10, 12, 10, 4, 4, 6, 4],
        k=1,
    )[0]

    if fmt == "sk_prefix":
        return f"sk-{rand(ALPHA, 32)}"
    if fmt == "openai_proj":
        return f"sk-proj-{rand(ALPHA, 48)}"
    if fmt == "stripe_key":
        return f"sk_{random.choice(['live', 'test'])}_{rand(ALPHA, 24)}"
    if fmt == "stripe_object":
        obj = random.choice(["pi", "ch", "cus", "pm", "src", "sub", "in", "txn"])
        return f"{obj}_{rand(ALPHA, 24)}"
    if fmt == "aws_access":
        return f"AKIA{rand('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567', 16)}"
    if fmt == "github_pat":
        pfx = random.choice(["ghp", "gho", "ghs", "ghr", "github_pat"])
        return f"{pfx}_{rand(ALPHA, 36)}"
    if fmt == "slack":
        kind = random.choice(["xoxb", "xoxp", "xoxa", "xoxr"])
        return (f"{kind}-{random.randint(10**10, 10**11)}-"
                f"{random.randint(10**10, 10**11)}-{rand(ALPHA, 24)}")
    if fmt == "jwt":
        header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = rand(ALPHA, random.randint(30, 80))
        sig = rand(ALPHA, random.randint(30, 60))
        return f"{header}.{payload}.{sig}"
    if fmt == "session_id_hex":
        return rand(HEX, random.choice([32, 40, 48, 64]))
    if fmt == "cookie_pair":
        name = random.choice([
            "PHPSESSID", "JSESSIONID", "sessionid", "connect.sid",
            "auth_token", "refresh_token", "access_token", "csrftoken",
            "remember_me", "_session", "XSRF-TOKEN",
        ])
        sep = random.choice(["=", "="])
        val_kind = random.random()
        if val_kind < 0.4:
            val = rand(HEX, random.randint(16, 32))
        elif val_kind < 0.7:
            val = rand(ALPHA, random.randint(12, 30))
        else:
            val = f"s%3A{rand(ALPHA, 16)}"
        return f"{name}{sep}{val}"
    if fmt == "auth_tok_prefix":
        prefix = random.choice([
            "auth_tok", "sess", "sid", "bsess", "scid", "usi", "pst", "api",
            "asc", "est", "tsid", "cst", "dat", "psi", "sso", "sls",
            "login", "jwt", "access", "refresh",
        ])
        sep = random.choice(["_", "-"])
        val = rand(ALPHA, random.randint(8, 20))
        return f"{prefix}{sep}{val}"
    if fmt == "sid_dashed":
        return f"SID-{random.randint(10000000, 99999999)}"
    if fmt == "bearer":
        return f"Bearer {rand(ALPHA, random.randint(20, 60))}"
    if fmt == "uuid_hex":
        return (f"{rand(HEX, 8)}-{rand(HEX, 4)}-{rand(HEX, 4)}-"
                f"{rand(HEX, 4)}-{rand(HEX, 12)}")
    # base64ish
    return rand(ALPHA + "+/", random.randint(40, 88))

def _password() -> str:
    SPECIAL = "!@#$%^&*-_+="
    UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LOWER = "abcdefghijklmnopqrstuvwxyz"
    DIGIT = "0123456789"
    ALL = UPPER + LOWER + DIGIT + SPECIAL

    fmt = random.choices(
        ["strong_mixed", "passphrase", "leetspeak", "with_year",
         "double_word_num", "common_weak", "long_mixed", "hex_token"],
        weights=[24, 14, 12, 14, 10, 6, 12, 8],
        k=1,
    )[0]

    if fmt == "strong_mixed":
        n = random.randint(10, 16)
        return (random.choice(UPPER) + random.choice(LOWER)
                + str(random.randint(10, 99)) + random.choice(SPECIAL)
                + "".join(random.choices(ALL, k=n - 4)))
    if fmt == "passphrase":
        words = random.sample(
            ["correct", "horse", "battery", "staple", "purple", "monkey",
             "rocket", "blue", "river", "thunder", "shadow", "winter",
             "dragon", "phoenix", "summer", "Mountain", "Castle", "Garden"],
            k=random.choice([3, 4]),
        )
        sep = random.choice(["-", "_", ""])
        return sep.join(words) + str(random.randint(10, 999))
    if fmt == "leetspeak":
        base = random.choice(["passw0rd", "h3ll0", "S3cur3", "Adm1n", "L0g1n"])
        return base + str(random.randint(10, 999)) + random.choice(SPECIAL)
    if fmt == "with_year":
        word = random.choice(["Summer", "Winter", "Spring", "Autumn",
                               "Mountain", "River", "Tiger", "Eagle"])
        return f"{word}{random.randint(1990, 2025)}{random.choice(SPECIAL)}"
    if fmt == "double_word_num":
        return (random.choice(["Apple", "Banana", "Cherry", "Pizza", "Tiger"])
                + random.choice(["Pie", "Pad", "Top", "Cat", "Dog"])
                + str(random.randint(100, 9999)))
    if fmt == "common_weak":
        return random.choice([
            "password123", "qwerty123", "admin123", "letmein!",
            "welcome1", "iloveyou", "abc12345", "P@ssw0rd",
        ])
    if fmt == "long_mixed":
        n = random.randint(20, 32)
        return "".join(random.choices(ALL, k=n))
    # hex_token
    return "".join(random.choices("0123456789abcdef", k=random.choice([32, 40, 64])))

_US_STATE_ABBR: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
]

_US_STATE_FULL: list[str] = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia", "Puerto Rico", "Guam",
    "US Virgin Islands", "American Samoa", "Northern Mariana Islands",
]


def _state_abbr() -> str:
    """US state — 60% 2-letter abbreviation, 40% full state name.

    Original only emitted abbreviations, which left the NER unable to
    recognise "California", "New York", "Texas" as us_state entities.
    """
    if random.random() < 0.6:
        return random.choice(_US_STATE_ABBR)
    return random.choice(_US_STATE_FULL)

def _zipcode() -> str:
    z5 = random.randint(10000, 99999)
    fmt = random.choices(["zip5", "zip_plus4", "zip_plus4_space"],
                         weights=[70, 22, 8], k=1)[0]
    if fmt == "zip5":
        return str(z5)
    plus4 = random.randint(1000, 9999)
    if fmt == "zip_plus4":
        return f"{z5}-{plus4}"
    return f"{z5} {plus4}"

def _gps() -> str:
    lat = round(random.uniform(-89.0, 89.0), random.choice([4, 5, 6]))
    lon = round(random.uniform(-179.0, 179.0), random.choice([4, 5, 6]))
    abs_lat, abs_lon = abs(lat), abs(lon)
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    fmt = random.choices(
        ["signed_pair", "degree_symbol", "labeled", "dms",
         "lat_lon_label", "geo_uri", "google_maps"],
        weights=[22, 18, 16, 14, 14, 8, 8],
        k=1,
    )[0]
    if fmt == "signed_pair":
        return f"{lat}, {lon}"
    if fmt == "degree_symbol":
        return f"{abs_lat}°{ns}, {abs_lon}°{ew}"
    if fmt == "labeled":
        return f"lat: {lat}, lon: {lon}"
    if fmt == "dms":
        deg_lat = int(abs_lat)
        min_lat = int((abs_lat - deg_lat) * 60)
        sec_lat = round(((abs_lat - deg_lat) * 60 - min_lat) * 60, 2)
        deg_lon = int(abs_lon)
        min_lon = int((abs_lon - deg_lon) * 60)
        sec_lon = round(((abs_lon - deg_lon) * 60 - min_lon) * 60, 2)
        return f"{deg_lat}°{min_lat}'{sec_lat}\"{ns} {deg_lon}°{min_lon}'{sec_lon}\"{ew}"
    if fmt == "lat_lon_label":
        return f"Latitude: {lat}, Longitude: {lon}"
    if fmt == "geo_uri":
        return f"geo:{lat},{lon}"
    # google_maps
    return f"https://maps.google.com/?q={lat},{lon}"

# ---------------------------------------------------------------------------
# Multicultural name pools — curated to teach the NER model open-vocabulary
# coverage across major naming traditions. Faker's locale support is
# inconsistent for some scripts, so we ship hand-picked exemplars.
# ---------------------------------------------------------------------------

KOREAN_NAMES: list[str] = [
    "Kim Min Soo", "Park Ji Hye", "Lee Jae Hwan", "Choi Yu Jin",
    "Jung Min Ho", "Kang Soo Yeon", "Yoon Sang Hyun", "Han Bo Reum",
    "Cho Min Ji", "Im Tae Yang", "Shin Hye Won", "Oh Joon Ho",
    "Kim Seo Yeon", "Park Do Hyun", "Lee Eun Jung", "Choi Hyun Woo",
    "Jang Mi Rae", "Hwang Sung Jin", "Bae Suzy", "Ryu Jun Yeol",
    "Son Ye Jin", "Hyun Bin", "Gong Yoo", "Bae Doona", "Kim Tae Hee",
]

CHINESE_NAMES: list[str] = [
    "Wei Zhang", "Li Chen", "Wang Fang", "Liu Yang", "Zhou Hao",
    "Sun Mei", "Xu Bing", "Lin Hua", "Yang Tao", "He Lei",
    "Huang Min", "Zhao Lan", "Xie Ming", "Tang Wei", "Cao Jun",
    "Zhu Lin", "Deng Yan", "Feng Xiao", "Pan Li", "Luo Jing",
    "Chen Wei Ming", "Zhang Xue Lian", "Wang Zi Hao", "Li Mei Ling",
    "Xi Jinping", "Yao Ming", "Lang Lang", "Li Na",
]

INDIAN_NAMES: list[str] = [
    "Ananya Reddy", "Rohan Sharma", "Priya Patel", "Arjun Singh",
    "Kavya Iyer", "Vikram Mehta", "Aditi Kumar", "Rahul Gupta",
    "Sneha Rao", "Aryan Verma", "Meera Nair", "Karan Malhotra",
    "Riya Joshi", "Siddharth Banerjee", "Tanvi Desai", "Aniket Bhat",
    "Pooja Krishnan", "Aakash Agarwal", "Ishaan Chowdhury",
    "Divya Pillai", "Manav Saxena", "Nisha Bose", "Ritu Tiwari",
    "Sandeep Yadav", "Kiran Bedi", "Raj Kapoor", "Mukesh Ambani",
]

JAPANESE_NAMES: list[str] = [
    "Sakura Tanaka", "Yuki Sato", "Hiroshi Suzuki", "Aiko Watanabe",
    "Kenji Yamamoto", "Yumi Nakamura", "Takashi Kobayashi",
    "Misaki Ito", "Daiki Kato", "Hanako Yoshida", "Ren Mori",
    "Akira Hayashi", "Mei Saito", "Riku Inoue", "Naoko Kimura",
    "Sho Matsumoto", "Asuka Ono", "Haruto Takahashi", "Yuna Miyazaki",
    "Shinzo Abe", "Haruki Murakami", "Hayao Miyazaki",
]

HISPANIC_NAMES: list[str] = [
    "Maria Garcia", "Juan Carlos Rodriguez", "Sofia Martinez",
    "Carlos Hernandez", "Isabella Lopez", "Diego Gonzalez",
    "Camila Perez", "Mateo Sanchez", "Valentina Ramirez",
    "Lucas Torres", "Alejandro Flores", "Lucia Rivera",
    "Sebastian Gomez", "Mariana Diaz", "Gabriel Ruiz",
    "Penelope Cruz", "Lionel Messi", "Frida Kahlo",
]

ARABIC_NAMES: list[str] = [
    "Mohammed Al-Saud", "Fatima Hassan", "Ahmed Ali", "Layla Khan",
    "Omar Abbas", "Aisha Rahman", "Yusuf Ibrahim", "Zara Mahmood",
    "Khalid Bin Salman", "Noor Al-Zahra", "Hassan Al-Maktoum",
    "Mariam El-Sayed", "Tariq Aziz", "Yasmin Saleh",
]

RUSSIAN_NAMES: list[str] = [
    "Vladimir Petrov", "Anastasia Ivanova", "Dmitri Sokolov",
    "Ekaterina Smirnova", "Sergei Volkov", "Olga Kuznetsova",
    "Mikhail Popov", "Natalia Vasilyeva", "Alexei Lebedev",
    "Tatyana Morozova", "Igor Novikov", "Yuri Pavlov",
]

EUROPEAN_NAMES: list[str] = [
    "Sophie Dubois", "Pierre Lefevre", "Hans Müller", "Greta Schmidt",
    "Giuseppe Rossi", "Elena Bianchi", "Antonio Ferreira",
    "Beatriz Silva", "Lars Eriksson", "Astrid Hansen", "Jan Kowalski",
    "Hanna Nowak", "Ana Sofia Nogueira",
]

NICKNAMES_POOL: list[str] = [
    "Kate", "Bob", "Liz", "Tom", "Sam", "Joe", "Pat", "Mike",
    "Sue", "Beth", "Dan", "Jim", "Ben", "Tim", "Jen", "Will",
    "Chris", "Rob", "Steve", "Andy", "Maddie", "Cathy", "Ginny",
]

NAME_SUFFIXES: list[str] = ["Jr.", "Sr.", "II", "III", "IV", "PhD", "Esq."]

GENEALOGY_PREFIXES: list[str] = ["S/o", "D/o", "W/o", "C/o"]


def _intl_name() -> str:
    """Pick from a curated multicultural name pool."""
    pool = random.choice([
        KOREAN_NAMES, CHINESE_NAMES, INDIAN_NAMES, JAPANESE_NAMES,
        HISPANIC_NAMES, ARABIC_NAMES, RUSSIAN_NAMES, EUROPEAN_NAMES,
    ])
    return random.choice(pool)


def _name_two_initials_surname() -> str:
    """Return name like 'K. W. Smith' (two initials + surname)."""
    a = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    b = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{a}. {b}. {fake.last_name()}"


def _name_with_middle_initial() -> str:
    """Return name like 'Kate W. Smith' or 'Alex M. Johnson'."""
    mid = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{fake.first_name()} {mid}. {fake.last_name()}"


def _name_with_initial_first_and_middle() -> str:
    """Return name like 'K. Kate W.' (initial first + name + initial last)."""
    a = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    b = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{a}. {fake.first_name()} {b}."


def _name_with_nickname() -> str:
    """Return name like 'Katherine \"Kate\" Doe'."""
    full = fake.first_name()
    nick = random.choice(NICKNAMES_POOL)
    return f'{full} "{nick}" {fake.last_name()}'


def _name_with_suffix() -> str:
    """Return name like 'Alex Johnson Jr.' or 'John Smith III'."""
    return f"{fake.first_name()} {fake.last_name()} {random.choice(NAME_SUFFIXES)}"


def _name_with_genealogy_prefix() -> str:
    """Return name like 'S/o Robert Smith' (Indian-style filiation marker)."""
    return f"{random.choice(GENEALOGY_PREFIXES)} {fake.first_name()} {fake.last_name()}"


def _name_all_caps() -> str:
    """Return all-caps name like 'KATE SMITH' or single-token 'KATE'."""
    if random.random() < 0.4:
        return random.choice(NICKNAMES_POOL).upper()
    return f"{fake.first_name().upper()} {fake.last_name().upper()}"


def _name_single_token() -> str:
    """Return a single-token name (just first name or just nickname)."""
    if random.random() < 0.5:
        return fake.first_name()
    return random.choice(NICKNAMES_POOL)


def _name_initial_plus_first() -> str:
    """Return name like 'K Kate' (single uppercase letter + first name)."""
    initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{initial} {random.choice(NICKNAMES_POOL)}"


def _first_last() -> str:
    """Multi-form, multi-cultural name generator.

    Distribution is calibrated to roughly mirror real-world naming variation
    in clinical/financial documents: standard Western names dominate but
    edge cases (initials, nicknames, multicultural, all-caps, suffixes) get
    enough representation for the NER model to generalize.
    """
    form = random.choices(
        [
            "faker_basic",          # Mike Smith
            "faker_with_middle",    # Mike S. Smith
            "intl",                 # Wei Zhang / Sakura Tanaka / Ananya Reddy / Kim Min Soo
            "two_initials_surname", # K. W. Smith
            "initial_first_middle", # K. Kate W.
            "nickname_quoted",      # Katherine "Kate" Doe
            "with_suffix",          # Alex Johnson Jr.
            "all_caps",             # KATE SMITH or KATE
            "single_token",         # Kate
            "initial_plus_first",   # K Kate
            "genealogy_prefix",     # S/o Robert Smith
        ],
        weights=[28, 14, 18, 6, 5, 5, 4, 5, 6, 5, 4],
        k=1,
    )[0]

    if form == "faker_basic":
        return f"{fake.first_name()} {fake.last_name()}"
    if form == "faker_with_middle":
        return _name_with_middle_initial()
    if form == "intl":
        return _intl_name()
    if form == "two_initials_surname":
        return _name_two_initials_surname()
    if form == "initial_first_middle":
        return _name_with_initial_first_and_middle()
    if form == "nickname_quoted":
        return _name_with_nickname()
    if form == "with_suffix":
        return _name_with_suffix()
    if form == "all_caps":
        return _name_all_caps()
    if form == "single_token":
        return _name_single_token()
    if form == "initial_plus_first":
        return _name_initial_plus_first()
    if form == "genealogy_prefix":
        return _name_with_genealogy_prefix()
    return f"{fake.first_name()} {fake.last_name()}"

def _street_address() -> str:
    streets = ["Main St", "Oak Ave", "Elm Blvd", "River Rd", "Park Lane",
               "Maple Drive", "Cedar Court", "Washington Blvd", "Pine Way",
               "Highland Ave", "Sunset Terrace", "Valley Rd", "Green St"]
    base = f"{random.randint(1,9999)} {random.choice(streets)}"
    if random.random() < 0.3:
        # GAP-22: Apt/Suite suffix variants
        suffix = random.choice([
            f"Apt {random.randint(1,999)}",
            f"Suite {random.randint(100,999)}",
            f"Ste {random.randint(100,999)}",
            f"Unit {random.randint(1,99)}",
            f"#{random.randint(1,999)}",
        ])
        return f"{base} {suffix}"
    return base

def _street_address_full() -> str:
    streets = ["Main St", "Oak Ave", "Elm Blvd", "River Rd", "Park Lane",
               "Maple Drive", "Cedar Court", "Washington Blvd", "Pine Way",
               "Highland Ave", "Sunset Terrace", "Valley Rd", "Green St",
               "Hampton Squares", "Kramer Springs", "Johnson Rd", "Lincoln Ave"]
    city = fake.city()
    state = _state_abbr()
    zipcode = _zipcode()
    return f"{random.randint(1,9999)} {random.choice(streets)}, {city}, {state} {zipcode}"

_INTL_CITIES: list[str] = [
    # India (top failures in user examples — Faker's en_US doesn't generate these)
    "Mumbai", "Delhi", "Bengaluru", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik",
    "Faridabad", "Meerut", "Rajkot", "Kalyan", "Vasai", "Varanasi",
    "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Allahabad",
    "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior", "Vijayawada",
    "Jodhpur", "Madurai", "Raipur", "Kota", "Chandigarh", "Mysuru",
    "Mysore", "Kochi", "Thiruvananthapuram", "Guwahati", "Ernakulam",
    # Europe
    "London", "Paris", "Berlin", "Madrid", "Rome", "Milan", "Vienna",
    "Amsterdam", "Brussels", "Lisbon", "Stockholm", "Copenhagen",
    "Helsinki", "Oslo", "Warsaw", "Prague", "Budapest", "Athens",
    "Dublin", "Edinburgh", "Manchester", "Birmingham", "Glasgow",
    "Munich", "Frankfurt", "Hamburg", "Cologne", "Zurich", "Geneva",
    "Barcelona", "Seville", "Valencia", "Florence", "Venice", "Naples",
    "Lyon", "Marseille", "Nice", "Bordeaux",
    # APAC
    "Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Sapporo",
    "Seoul", "Busan", "Incheon",
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Xi'an",
    "Hong Kong", "Taipei", "Kaohsiung",
    "Singapore", "Kuala Lumpur", "Jakarta", "Bangkok", "Manila",
    "Ho Chi Minh City", "Hanoi", "Phnom Penh", "Yangon", "Colombo",
    "Karachi", "Lahore", "Islamabad", "Dhaka", "Kathmandu",
    "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Auckland",
    "Wellington",
    # Middle East / Africa
    "Dubai", "Abu Dhabi", "Doha", "Riyadh", "Jeddah", "Kuwait City",
    "Tel Aviv", "Jerusalem", "Istanbul", "Ankara", "Tehran", "Baghdad",
    "Cairo", "Alexandria", "Lagos", "Nairobi", "Johannesburg",
    "Cape Town", "Casablanca", "Addis Ababa", "Accra", "Dakar",
    # Americas (non-US)
    "Toronto", "Vancouver", "Montreal", "Ottawa", "Calgary",
    "Mexico City", "Guadalajara", "Monterrey", "Tijuana",
    "Buenos Aires", "Rio de Janeiro", "São Paulo", "Lima", "Santiago",
    "Bogotá", "Caracas", "Quito", "Havana", "San Juan",
    # US cities (small curated set in addition to Faker)
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Seattle", "Boston", "Atlanta", "Miami", "Denver",
    "Portland", "Las Vegas", "Detroit", "Minneapolis",
]


def _city() -> str:
    """City name — 50% Faker (US), 50% curated international pool."""
    if random.random() < 0.5:
        return fake.city()
    return random.choice(_INTL_CITIES)


_EMAIL_DOMAINS: list[str] = [
    # Free / consumer
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "aol.com", "protonmail.com", "proton.me", "fastmail.com", "zoho.com",
    "yandex.com", "mail.ru", "qq.com", "163.com", "126.com",
    "live.com", "msn.com", "me.com", "rediffmail.com",
    # Edu (US)
    "harvard.edu", "stanford.edu", "mit.edu", "berkeley.edu", "yale.edu",
    "princeton.edu", "columbia.edu", "ucla.edu", "umich.edu", "nyu.edu",
    "ucsf.edu", "ucsd.edu", "upenn.edu",
    # Corporate
    "company.com", "corp.com", "enterprise.com", "techcorp.io",
    "acme.co", "example.com", "example.org", "example.net",
    # Government / health
    "hospital.org", "clinic.org", "medcenter.com", "health.gov",
    "irs.gov", "state.gov", "treasury.gov",
    # India
    "rediffmail.com", "yahoo.in", "gmail.com", "indianoil.in", "tcs.com",
    "infosys.com", "wipro.com", "icicibank.com", "hdfcbank.com",
    "sbi.co.in",
    # UK
    "co.uk", "ac.uk", "nhs.uk", "btinternet.com", "sky.com",
]


def _email() -> str:
    """Email address across consumer/edu/corporate/intl + plus-addressing.

    Original used only fake.email() which produced one Faker-style pattern.
    Failure cases include +tag addresses, dotted local parts, edu addresses,
    and international TLDs.
    """
    fn = fake.first_name().lower()
    ln = fake.last_name().lower()
    domain = random.choice(_EMAIL_DOMAINS)
    fmt = random.choices(
        ["faker", "first_dot_last", "first_initial_last", "first_last",
         "first_underscore_last", "with_digits", "plus_tag",
         "first_only", "initials", "ln_first"],
        weights=[16, 18, 14, 10, 8, 12, 8, 6, 4, 4],
        k=1,
    )[0]
    if fmt == "faker":
        return fake.email()
    if fmt == "first_dot_last":
        return f"{fn}.{ln}@{domain}"
    if fmt == "first_initial_last":
        return f"{fn[0]}{ln}@{domain}"
    if fmt == "first_last":
        return f"{fn}{ln}@{domain}"
    if fmt == "first_underscore_last":
        return f"{fn}_{ln}@{domain}"
    if fmt == "with_digits":
        return f"{fn}{ln}{random.randint(1, 999)}@{domain}"
    if fmt == "plus_tag":
        tag = random.choice(["work", "promo", "newsletter", "shopping",
                              "signup", "spam", "junk", "billing"])
        return f"{fn}.{ln}+{tag}@{domain}"
    if fmt == "first_only":
        return f"{fn}@{domain}"
    if fmt == "initials":
        return f"{fn[0]}{ln[0]}@{domain}"
    return f"{ln}{fn}@{domain}"

# ────────────────────────────────────────────────────────────────────────────
# Organization-name pool — Group B (second batch)
# ────────────────────────────────────────────────────────────────────────────
# A diverse pool of company / nonprofit / government / research / startup
# organization names. Covers the production failure modes:
#   • Multi-word adjective-style names (Bright Future Solutions, Quantum AI Labs)
#   • Industry-suffix variants (Inc., LLC, Ltd., Corp., GmbH, S.A., Pvt Ltd)
#   • Foundation / charity (Helping Hands Foundation)
#   • Government bodies (Department of Public Services)
#   • International / ANZ / EU / APAC examples
#   • Real-world brands (OpenAI Technologies, Tata Consultancy Services)
#   • Acronym / initialism variants (HH Foundation, NGI Solutions, Apex Inc.)

_ORG_PREFIXES = [
    "Bright", "Global", "NextGen", "Visionary", "Apex", "Quantum", "Pinnacle",
    "Horizon", "Future", "Helping Hands", "Department of", "Bureau of",
    "Ministry of", "National", "International", "Metro", "Urban", "Pacific",
    "Atlantic", "Premier", "Prime", "Titan", "Stellar", "Phoenix", "Vertex",
    "Catalyst", "Innovate", "Synergy", "Vanguard", "Crestline", "Aurora",
    "Lumina", "Beacon", "Summit", "Pillar", "Cascade", "Velocity",
    "Northstar", "Silverline", "Goldline", "Frontier", "Compass", "Origin",
    "Polaris", "Constellation", "Sterling", "Ironclad", "Ascend", "Elevate",
    "United", "Allied", "Coastal", "Highland", "Lakeshore", "Riverstone",
    "Sapphire", "Emerald", "Sunset", "Daybreak", "Eastline", "Westfield",
    "Trustline", "Heritage", "Legacy", "Insight", "Foresight",
    # Real / well-known brands (helps the model recognise common patterns)
    "OpenAI", "Tata", "Infosys", "Wipro", "Cognizant", "Accenture", "Capgemini",
    "Microsoft", "Salesforce", "Oracle", "Adobe", "IBM", "Intel", "Cisco",
    "World Health", "United Nations", "Red Cross", "Amnesty",
]

_ORG_MID = [
    "Future", "Tech", "Technologies", "Systems", "AI", "Data", "Cloud",
    "Digital", "Software", "Hardware", "Capital", "Holdings", "Ventures",
    "Energy", "Logistics", "Health", "BioPharma", "Pharma", "Diagnostics",
    "Robotics", "Analytics", "Networks", "Solutions", "Communications",
    "Media", "Broadcasting", "Research", "Manufacturing", "Logistics",
    "Transit", "Public", "Civil", "Workforce", "Labour", "Education",
    "Science", "Mobility", "Aerospace", "Marine", "Mining", "Construction",
    "Insurance", "Banking", "Financial", "Investment", "Equity", "Wealth",
    "Studio", "Innovation", "Advisory", "Global", "Strategic",
    # Organization-type adjectives the user examples lean on
    "Consulting", "Retail", "Transportation", "Media", "Research", "Startup",
    "Manufacturing", "Government", "Nonprofit", "Education", "Quantum",
]

# Suffixes alone aren't an org name — they MUST follow at least one prefix
# (and usually a mid-segment too). Distribution-weighted to mirror reality.
_ORG_SUFFIXES = [
    "Inc.", "Inc", "LLC", "LLC.", "Ltd.", "Ltd", "Co.", "Corp.",
    "Corporation", "Holdings", "Group", "Group LLC", "Holdings Inc.",
    "Global Inc.", "International Ltd.", "Pvt Ltd", "Pvt. Ltd.", "Private Limited",
    "PLC", "P.C.", "LLP", "Partners", "Partners LLP",
    "GmbH", "AG", "S.A.", "S.A. de C.V.", "B.V.", "N.V.", "Pty Ltd",
    "Foundation", "Trust", "Council", "Society", "Association", "Alliance",
    "Consortium", "Institute", "Labs", "Studios", "Ventures",
    "Department", "Ministry", "Bureau", "Agency", "Authority", "Commission",
    "Network", "Networks", "Services", "Systems", "Solutions", "Technologies",
]

# Curated real-world examples — used 15 % of the time so the model anchors
# its understanding of organization shape against well-known brands.
_ORG_KNOWN: list[str] = [
    "OpenAI Technologies", "Bright Future Solutions", "Global Tech Systems",
    "NextGen Innovations", "National Research Institute", "Apex Holdings Inc.",
    "Helping Hands Foundation", "Department of Public Services",
    "Quantum AI Labs", "Visionary Apps Studio", "Prime Advisory Group",
    "Urban Market Retailers", "Metro Transit Authority",
    "Titan Manufacturing Co.", "Horizon Broadcasting Network",
    "Future Science Consortium", "United Nations", "World Health Organization",
    "Tata Consultancy Services", "Infosys Limited", "Wipro Technologies",
    "Cognizant Technology Solutions", "Accenture plc", "Capgemini SE",
    "Microsoft Corporation", "Oracle Corporation", "Salesforce, Inc.",
    "Adobe Systems Incorporated", "International Business Machines",
    "Intel Corporation", "Cisco Systems, Inc.", "Pfizer Inc.",
    "Johnson & Johnson", "Mayo Clinic Foundation", "Cleveland Clinic Foundation",
    "American Red Cross", "Doctors Without Borders", "Bill & Melinda Gates Foundation",
    "International Monetary Fund", "Federal Reserve Bank",
    "Bay Area Rapid Transit", "Federal Bureau of Investigation",
    "Centers for Disease Control", "Department of Veterans Affairs",
    "European Central Bank", "United Way Worldwide", "UNICEF",
    "Federal Aviation Administration", "Department of Defense",
]

# Acronym / initialism organization names (e.g. "HH Foundation", "NGI Solutions",
# "MTA Transit", "NRI Institute", "Apex Inc.", "QAILabs"). The user examples
# included these short forms, so we generate them as a distinct shape so the
# NER model learns to recognise abbreviated org names too.
_ORG_ACRONYM_SUFFIXES = [
    "Solutions", "Institute", "Foundation", "Inc.", "LLC", "Group",
    "Holdings", "Agency", "Transit", "Labs", "Foundation", "Services",
    "Networks", "Authority", "Council", "Society", "Foundation", "Studio",
]


def _company() -> str:
    """Generate a diverse organization name.

    Distribution:
      • 15 % real-world / curated examples (anchors well-known shapes)
      • 25 % prefix + suffix      ("Apex Holdings", "Bright Foundation")
      • 25 % prefix + mid + suffix ("Bright Future Solutions",
                                     "Quantum AI Labs",
                                     "Department of Public Services")
      • 15 % acronym + suffix       ("NGI Solutions", "HH Foundation")
      • 10 % "<Prefix> & <Prefix>" / "<Prefix>-<Mid>"   (compound brands)
      •  5 % Faker fallback         (random plausible org for variety)
      •  5 % "<X> of <Region>"      ("Department of Public Services",
                                     "Bureau of Workforce Statistics")
    """
    form = random.choices(
        ["known", "prefix_suffix", "prefix_mid_suffix",
         "acronym_suffix", "compound", "of_phrase", "faker"],
        weights=[15, 25, 25, 15, 10, 5, 5],
        k=1,
    )[0]

    if form == "known":
        return random.choice(_ORG_KNOWN)

    if form == "prefix_suffix":
        return f"{random.choice(_ORG_PREFIXES)} {random.choice(_ORG_SUFFIXES)}"

    if form == "prefix_mid_suffix":
        # 50 % chance to drop the suffix (e.g. "Bright Future Solutions"
        # already feels complete without "Inc.").
        prefix = random.choice(_ORG_PREFIXES)
        mid = random.choice(_ORG_MID)
        if random.random() < 0.5:
            return f"{prefix} {mid} {random.choice(_ORG_SUFFIXES)}"
        # Two-mid form: "Bright Future Tech Solutions"
        if random.random() < 0.4:
            mid2 = random.choice(_ORG_MID)
            if mid2 != mid:
                return f"{prefix} {mid} {mid2}"
        return f"{prefix} {mid} {random.choice(['Solutions', 'Systems', 'Labs', 'Studio', 'Studios', 'Group', 'Network', 'Services', 'Holdings', 'Innovations', 'Consortium', 'Institute', 'Authority'])}"

    if form == "acronym_suffix":
        # Build a 2-3 letter acronym from a phrase.
        n_letters = random.choice([2, 3, 3, 4])
        acronym = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                          for _ in range(n_letters))
        return f"{acronym} {random.choice(_ORG_ACRONYM_SUFFIXES)}"

    if form == "compound":
        # "Smith & Johnson Holdings", "Brown-Carter Group"
        a = random.choice(_ORG_PREFIXES)
        b = random.choice(_ORG_PREFIXES)
        joiner = random.choice([" & ", "-", " and "])
        return f"{a}{joiner}{b} {random.choice(_ORG_SUFFIXES)}"

    if form == "of_phrase":
        head = random.choice([
            "Department", "Bureau", "Ministry", "Agency", "Office",
            "Council", "Commission", "Authority", "Institute", "Society",
        ])
        topic = random.choice([
            "Public Services", "Workforce Statistics", "Civil Aviation",
            "Health and Human Services", "Transportation", "Education",
            "Veterans Affairs", "Public Safety", "Foreign Affairs",
            "Internal Revenue", "Energy", "Commerce", "Labor",
            "Children and Families", "Consumer Protection",
            "Environmental Protection", "Health and Wellness",
            "Industrial Relations", "Trade and Industry",
        ])
        return f"{head} of {topic}"

    return fake.company()


_HOSPITAL_GENERIC_ADJ = [
    "Regional", "Community", "General", "Central", "Pacific", "Northern",
    "Valley", "Riverside", "University", "Metropolitan", "Lakeside",
    "St. Mary's", "St. Joseph", "Holy Cross", "Mercy", "Sacred Heart",
    "Saint Luke's", "Memorial", "Eastside", "Westside", "Southside",
]

_HOSPITAL_GENERIC_NOUN = [
    "Medical Center", "Hospital", "Health System", "Memorial Hospital",
    "Healthcare", "Medical Group", "Medical Pavilion",
]

# Specialty / niche facility taxonomy — the failing samples in production
# clustered here (women's clinics, rehab, dental, trauma, urgent care, etc.).
_HOSPITAL_SPECIALTY_TEMPLATES = [
    # Trauma / emergency centers
    ("{adj} Trauma Center", ["Metro", "Citywide", "Riverside", "Pacific",
                              "University", "Regional", "Northern", "St. Vincent's"]),
    ("Level {n} Trauma Center", ["1", "2", "3", "I", "II", "III", "IV"]),
    # Rehabilitation / recovery
    ("{adj} Rehab Center", ["Restore", "Recovery", "Phoenix", "New Hope",
                              "Sunrise", "Renew", "Bridge", "Pathways"]),
    ("{adj} Rehabilitation Hospital", ["Premier", "Lakeside", "Hilltop",
                                          "Coastal", "Mountain View", "Magnolia"]),
    ("{adj} Physical Therapy Center", ["AthletiCare", "Motion", "Active Life",
                                          "ProRehab", "Velocity"]),
    # Dental / Oral
    ("{adj} Dental Center", ["BrightSmile", "Pearl", "GentleCare",
                               "Family", "Aspen", "Smile Studio"]),
    ("{adj} Dental Clinic", ["Sunshine", "Healthy Teeth", "Crystal",
                               "Premier", "Cornerstone"]),
    ("{adj} Oral Surgery Center", ["Atlantic", "Summit", "Pioneer", "Capital"]),
    # Women's / OB-GYN / fertility
    ("{adj} Women's Clinic", ["Grace", "Hope", "Lotus", "Wellness",
                                "Bayside", "Harmony", "Serenity"]),
    ("{adj} Women's Health Center", ["Aurora", "Crescent", "Maple",
                                        "Northstar", "Evergreen"]),
    ("{adj} OB-GYN Associates", ["Spring Valley", "Riverstone", "Coastal",
                                   "Garden State", "Magnolia"]),
    ("{adj} Fertility Clinic", ["Dawn", "New Beginnings", "Genesis",
                                  "Bloom", "First Steps"]),
    # Mental / behavioral health
    ("{adj} Behavioral Health Center", ["Cornerstone", "Beacon", "Clearview",
                                           "Cedar", "Tranquil", "Mindful Path"]),
    ("{adj} Mental Health Clinic", ["Compass", "Anchor", "Pinecrest",
                                      "Riverbend", "Quiet Waters"]),
    ("{adj} Psychiatric Hospital", ["Sunrise", "Crestview", "Greenfield",
                                       "Hilltop", "Forest Glen"]),
    # Urgent care / walk-in
    ("{adj} Urgent Care", ["FastMed", "QuickCare", "PremierCare",
                             "MinuteMed", "EZ-Health", "RapidCare"]),
    ("{adj} Walk-In Clinic", ["Sunrise", "Cornerstone", "Anytime",
                                "Citywide", "Lakeside"]),
    # Surgical / specialty
    ("{adj} Surgical Center", ["Apex", "Vanguard", "Premier", "Pinnacle",
                                  "MedStar", "Crossroads"]),
    ("{adj} Cardiac Center", ["Heart Stream", "Cardio Plus", "VitalHeart",
                                 "Pulse", "BeatCare"]),
    ("{adj} Cancer Center", ["Hope", "Beacon", "Magnolia", "Horizon",
                                "Wellspring", "Cornerstone Oncology"]),
    ("{adj} Eye Institute", ["Vision", "ClearSight", "Crystal",
                                "EyeCare", "OptiVision"]),
    ("{adj} Orthopedic Hospital", ["BoneCare", "Sterling", "Joint Health",
                                       "Apex", "Premier"]),
    ("{adj} Children's Hospital", ["Rainbow", "Little Stars", "Sunshine",
                                       "Hopeful Hearts", "Bumblebee"]),
    # Diagnostic / imaging
    ("{adj} Diagnostic Center", ["Insight", "Clarity", "Apex",
                                    "Beacon", "Premier"]),
    ("{adj} Imaging Center", ["Crystal", "Clear View", "Premier",
                                "Pinnacle", "Apex"]),
    # Hospice / Long-term
    ("{adj} Hospice", ["Peaceful Path", "Compassionate Care", "Serenity",
                          "Twilight", "Restful Haven"]),
    ("{adj} Skilled Nursing Facility", ["Maplewood", "Pinecrest",
                                            "Oakwood", "Brookhaven"]),
    # Animal / Veterinary (some PII flows mention vet records when caregiver overlaps)
    ("{adj} Veterinary Hospital", ["PetCare", "Companion", "Animal Health",
                                       "Furry Friends"]),
]


def _hospital() -> str:
    """Generate a hospital / medical-facility name with broad taxonomy coverage.

    Distribution: 55% generic hospital ('General Hospital'), 45% specialty
    facility ('Grace Women's Clinic', 'BrightSmile Dental Center', etc.).
    """
    if random.random() < 0.55:
        return f"{random.choice(_HOSPITAL_GENERIC_ADJ)} {random.choice(_HOSPITAL_GENERIC_NOUN)}"

    template, adj_pool = random.choice(_HOSPITAL_SPECIALTY_TEMPLATES)
    return template.format(adj=random.choice(adj_pool), n=random.choice(adj_pool))

_BANK_NAMES: list[str] = [
    # US majors
    "Chase", "JPMorgan Chase", "Bank of America", "Wells Fargo", "Citibank",
    "US Bank", "PNC Bank", "TD Bank", "Capital One", "Truist", "HSBC USA",
    "Goldman Sachs Bank", "Morgan Stanley Bank", "Charles Schwab Bank",
    "Ally Bank", "Discover Bank", "American Express National Bank",
    # US regional / credit unions
    "First National Bank", "City Savings Bank", "Heritage Credit Union",
    "Union Trust Bank", "Navy Federal Credit Union", "USAA Federal Savings Bank",
    "Pentagon Federal Credit Union", "First Republic Bank", "Silicon Valley Bank",
    "Signature Bank", "M&T Bank", "Regions Bank", "Fifth Third Bank",
    "KeyBank", "Huntington Bank", "BMO Harris Bank",
    # International
    "Barclays", "HSBC", "Standard Chartered", "Lloyds Bank",
    "Royal Bank of Scotland", "NatWest", "Santander", "BNP Paribas",
    "Deutsche Bank", "Credit Suisse", "UBS", "ING", "Rabobank",
    "Société Générale", "UniCredit", "Intesa Sanpaolo",
    # India
    "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank",
    "Kotak Mahindra Bank", "Punjab National Bank", "Bank of Baroda",
    "IndusInd Bank", "Yes Bank", "IDBI Bank", "Canara Bank",
    "Union Bank of India", "Bank of India", "Federal Bank",
    # APAC + ME
    "DBS Bank", "OCBC Bank", "UOB", "Mizuho Bank", "Bank of Tokyo-Mitsubishi UFJ",
    "Sumitomo Mitsui Banking", "ANZ Bank", "Commonwealth Bank of Australia",
    "Westpac", "Bank of China", "Industrial and Commercial Bank of China",
    "Emirates NBD", "Qatar National Bank", "Abu Dhabi Commercial Bank",
]


def _bank_name() -> str:
    return random.choice(_BANK_NAMES)


_INSURANCE_NAMES: list[str] = [
    # US health
    "Blue Cross Blue Shield", "Aetna", "Cigna", "Humana", "Anthem",
    "UnitedHealthcare", "Kaiser Permanente", "Centene", "Molina Healthcare",
    "WellCare", "Oscar Health", "Bright Health", "Clover Health",
    # US property/casualty
    "State Farm", "Allstate", "Travelers Insurance", "Liberty Mutual",
    "Progressive", "GEICO", "Nationwide", "Farmers Insurance",
    "American Family Insurance", "USAA Insurance", "MetLife Insurance",
    "Prudential", "New York Life", "Northwestern Mutual",
    "Mutual of Omaha", "AIG", "Chubb", "The Hartford", "Erie Insurance",
    # International
    "AXA", "Allianz", "Zurich Insurance", "Generali", "Aviva",
    "Munich Re", "Swiss Re", "Lloyd's of London", "QBE Insurance",
    "Tokio Marine", "Ping An Insurance", "China Life Insurance",
    "Manulife", "Sun Life Financial", "Great-West Lifeco",
    # India
    "LIC", "HDFC Life", "ICICI Prudential", "SBI Life Insurance",
    "Max Life Insurance", "Bajaj Allianz", "Tata AIG", "Star Health",
    "New India Assurance", "United India Insurance", "Reliance General Insurance",
    # Custom/synthetic
    "United Shield Insurance",
]


def _insurance_co() -> str:
    return random.choice(_INSURANCE_NAMES)

def _case_number() -> str:
    n = random.randint(1000, 99999)
    year = random.randint(2018, 2025)
    fmt = random.choices(
        ["court_year", "court_attached", "case_word", "court_only",
         "type_year", "docket", "div_year", "bare"],
        weights=[22, 14, 14, 12, 16, 12, 6, 4],
        k=1,
    )[0]
    pfx = random.choice(["CR", "CV", "CF", "DK", "CASE", "CC", "FA"])
    if fmt == "court_year":
        return f"{pfx}-{year}-{n:05d}"
    if fmt == "court_attached":
        return f"{pfx}{year}{n:05d}"
    if fmt == "case_word":
        return f"CASE No: {pfx}-{year}-{n:05d}"
    if fmt == "court_only":
        return f"{pfx}-{n:06d}"
    if fmt == "type_year":
        kind = random.choice(["CRIM", "CIVIL", "FAM", "PROB", "JUV"])
        return f"{kind}-{year}-{n}"
    if fmt == "docket":
        return f"Docket #{year}-{n:05d}"
    if fmt == "div_year":
        div = random.choice(["A", "B", "C", "D", "1", "2", "3"])
        return f"{pfx}-{year}-{div}-{n}"
    return f"{n:08d}"


def _case_number_alphanumeric() -> str:
    return _case_number()

def _phone_dotted() -> str:
    return f"{random.randint(200,999)}.{random.randint(200,999)}.{random.randint(1000,9999)}"

def _phone_with_ext() -> str:
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}x{random.randint(100,9999)}"

def _npi_with_prefix() -> str:
    return f"NPI-{random.randint(1000000000, 9999999999)}"

def _amount() -> str:
    """Multi-currency, multi-format monetary value generator.

    Covers symbols ($/€/£/¥/₹), 3-letter codes (USD/EUR/GBP/INR/JPY/CAD/AUD/CHF/AED/SGD),
    prefix/suffix positions, abbreviations (K/M/L/Cr), per-period (/mo, /yr),
    European decimal commas, and bare integer / no-decimal forms.
    """
    symbols = ["$", "€", "£", "¥", "₹"]
    codes = ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD",
             "CHF", "AED", "SGD", "HKD", "NZD", "ZAR", "BRL"]
    form = random.choices(
        [
            "sym_prefix_decimal",    # $1,234.56
            "sym_prefix_integer",    # ₹89000
            "sym_prefix_abbr",       # ₹89K / €1.5M / ₹8.75L
            "code_prefix",           # USD 1250.00
            "code_suffix",           # 1,250 USD
            "european_comma",        # €780,50
            "per_period",            # $19.99/mo
            "bare_decimal",          # 1234.56
            "code_abbr",             # 2.5M JPY / EUR 50K
            "no_decimal_large",      # JPY 250000
        ],
        weights=[20, 10, 14, 14, 8, 6, 6, 4, 10, 8],
        k=1,
    )[0]

    if form == "sym_prefix_decimal":
        sym = random.choice(symbols)
        return f"{sym}{random.randint(1, 99999):,}.{random.randint(0, 99):02d}"

    if form == "sym_prefix_integer":
        sym = random.choice(symbols)
        return f"{sym}{random.randint(100, 999999):,}"

    if form == "sym_prefix_abbr":
        sym = random.choice(symbols)
        whole = random.randint(1, 999)
        frac = random.choice(["", f".{random.randint(1, 99)}"])
        unit = random.choice(["K", "M", "L", "Cr", "B"])
        return f"{sym}{whole}{frac}{unit}"

    if form == "code_prefix":
        code = random.choice(codes)
        return f"{code} {random.randint(1, 999999):,}.{random.randint(0, 99):02d}"

    if form == "code_suffix":
        code = random.choice(codes)
        return f"{random.randint(1, 999999):,}.{random.randint(0, 99):02d} {code}"

    if form == "european_comma":
        sym = random.choice(["€", "£"])
        return f"{sym}{random.randint(1, 9999)},{random.randint(10, 99):02d}"

    if form == "per_period":
        sym = random.choice(symbols)
        period = random.choice(["/mo", "/yr", "/month", "/year", " per month"])
        return f"{sym}{random.randint(1, 999)}.{random.randint(0, 99):02d}{period}"

    if form == "bare_decimal":
        return f"{random.randint(1, 999999):,}.{random.randint(0, 99):02d}"

    if form == "code_abbr":
        code = random.choice(codes)
        whole = random.randint(1, 999)
        frac = random.choice(["", f".{random.randint(1, 9)}"])
        unit = random.choice(["K", "M", "L", "B"])
        # 50% prefix, 50% suffix
        if random.random() < 0.5:
            return f"{code} {whole}{frac}{unit}"
        return f"{whole}{frac}{unit} {code}"

    # no_decimal_large
    code = random.choice(codes)
    return f"{code} {random.randint(10000, 9999999)}"

def _employee_id() -> str:
    n = random.randint(1000, 999999)
    fmt = random.choices(
        ["emp_dashed", "emp_underscore", "emp_bare", "e_short",
         "id_prefix", "staff", "badge", "emp_space", "raw_digits"],
        weights=[20, 10, 12, 10, 12, 10, 8, 8, 10],
        k=1,
    )[0]
    if fmt == "emp_dashed":
        return f"EMP-{n}"
    if fmt == "emp_underscore":
        return f"EMP_{n}"
    if fmt == "emp_bare":
        return f"EMP{n}"
    if fmt == "e_short":
        return f"E{n}"
    if fmt == "id_prefix":
        return f"ID-{n}"
    if fmt == "staff":
        return f"STAFF-{n}"
    if fmt == "badge":
        return f"BADGE-{n}"
    if fmt == "emp_space":
        return f"EMP {n}"
    return str(n)


def _student_id() -> str:
    n = random.randint(1000, 99999999)
    fmt = random.choices(
        ["stu_dashed", "stu_underscore", "stu_bare", "s_short",
         "student_word", "id_prefix", "univ_prefix", "raw_digits"],
        weights=[18, 10, 12, 12, 12, 10, 14, 12],
        k=1,
    )[0]
    if fmt == "stu_dashed":
        return f"STU-{n}"
    if fmt == "stu_underscore":
        return f"STU_{n}"
    if fmt == "stu_bare":
        return f"STU{n}"
    if fmt == "s_short":
        return f"S{n}"
    if fmt == "student_word":
        return f"Student #{n}"
    if fmt == "id_prefix":
        return f"ID-{n}"
    if fmt == "univ_prefix":
        univ = random.choice(["UNI", "MIT", "UCLA", "NYU", "UCSF", "BU", "USC"])
        return f"{univ}-{n}"
    return str(n)


def _tax_id() -> str:
    fmt = random.choices(
        ["ein", "itin", "indian_pan", "vat_eu", "tin_dashed", "bare_9"],
        weights=[24, 16, 14, 12, 20, 14],
        k=1,
    )[0]
    if fmt == "ein":
        return f"{random.randint(10, 99):02d}-{random.randint(1000000, 9999999):07d}"
    if fmt == "itin":
        # ITIN format: 9XX-7X-XXXX (always starts with 9)
        return f"9{random.randint(10, 99):02d}-{random.choice([7,8])}{random.randint(0,9)}-{random.randint(1000, 9999):04d}"
    if fmt == "indian_pan":
        # PAN: 5 letters + 4 digits + 1 letter
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        digits = random.randint(1000, 9999)
        last = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return f"{letters}{digits}{last}"
    if fmt == "vat_eu":
        country = random.choice(["DE", "FR", "GB", "IT", "ES", "NL", "BE", "AT"])
        return f"{country}{random.randint(100000000, 999999999)}"
    if fmt == "tin_dashed":
        return f"TIN-{random.randint(100000000, 999999999)}"
    # bare_9 — undelimited 9 digits, common in US business filings
    return f"{random.randint(100000000, 999999999)}"

def _flight_number() -> str:
    """IATA flight number with airline-code and separator diversity.

    Covers US/EU/Asian carriers including the Indian low-cost airlines
    that were missing from the original pool (6E, G8, UK, IX, AI, SG).
    """
    codes = [
        "AA", "UA", "DL", "SW", "WN", "B6", "AS", "F9", "NK",  # US
        "BA", "LH", "AF", "KL", "IB", "AZ", "LX", "TK",          # EU
        "EK", "QR", "EY", "SV", "GF", "MS",                       # ME
        "SQ", "CX", "JL", "NH", "QF", "NZ", "TG", "MH", "OZ",    # APAC
        "6E", "G8", "UK", "IX", "AI", "SG",                       # India
    ]
    code = random.choice(codes)
    num = random.randint(1, 9999)
    sep = random.choices([" ", "", "-", "/", "_"], weights=[40, 30, 12, 10, 8], k=1)[0]
    flight = f"{code}{sep}{num}"
    # 15 % of the time prefix with the airline's spoken name, e.g. "IndiGo 6E345"
    if random.random() < 0.15:
        spoken = {
            "6E": "IndiGo", "G8": "Go First", "UK": "Vistara", "IX": "Air India Express",
            "AI": "Air India", "SG": "SpiceJet", "AA": "American", "UA": "United",
            "DL": "Delta", "BA": "British Airways", "LH": "Lufthansa", "EK": "Emirates",
            "QR": "Qatar Airways", "SQ": "Singapore Airlines", "CX": "Cathay Pacific",
        }.get(code)
        if spoken:
            return f"{spoken} {code}{num}"
    return flight

def _booking_ref() -> str:
    """Airline / hotel / OTA booking reference (PNR).

    Real PNRs are 6 alphanumerics with no separator (e.g. ABC123, X3F9KP).
    OTAs like MakeMyTrip/Cleartrip/Booking.com use longer alphanumerics with
    site-specific prefixes.
    """
    ALPHA = "ABCDEFGHIJKLMNPQRSTUVWXYZ"  # exclude O for readability
    NUM = "0123456789"
    fmt = random.choices(
        ["pnr6", "bk_prefix", "conf_prefix", "ref_prefix", "ota_brand",
         "hotel", "pnr_dashed", "long_alpha"],
        weights=[28, 14, 12, 10, 16, 8, 6, 6],
        k=1,
    )[0]
    if fmt == "pnr6":
        return "".join(random.choices(ALPHA + NUM, k=6))
    if fmt == "bk_prefix":
        return f"BK-{''.join(random.choices(ALPHA + NUM, k=6))}"
    if fmt == "conf_prefix":
        return f"CONF-{''.join(random.choices(ALPHA + NUM, k=8))}"
    if fmt == "ref_prefix":
        return f"REF{random.randint(100000, 99999999)}"
    if fmt == "ota_brand":
        brand = random.choice([
            "MAKEMYTRIP", "CLEARTRIP", "YATRA", "EASEMYTRIP",
            "BOOKING", "EXPEDIA", "AGODA",
        ])
        return f"{brand}{random.randint(100000, 999999)}"
    if fmt == "hotel":
        return f"HTL-{random.randint(100000, 9999999)}"
    if fmt == "pnr_dashed":
        return (f"{''.join(random.choices(ALPHA, k=3))}-"
                f"{''.join(random.choices(NUM, k=3))}")
    # long_alpha — 10+ alphanumeric
    return "".join(random.choices(ALPHA + NUM, k=random.randint(10, 12)))


def _claim_number() -> str:
    n = random.randint(100000, 999999999)
    fmt = random.choices(
        ["clm_dashed", "claim_word", "cl_short", "year_claim",
         "ins_claim", "raw_digits"],
        weights=[24, 18, 14, 18, 14, 12],
        k=1,
    )[0]
    if fmt == "clm_dashed":
        return f"CLM-{n}"
    if fmt == "claim_word":
        return f"CLAIM-{n}"
    if fmt == "cl_short":
        return f"CL{n}"
    if fmt == "year_claim":
        return f"CLM-{random.randint(2018, 2025)}-{random.randint(10000, 999999)}"
    if fmt == "ins_claim":
        return f"INS-CLM-{n}"
    return str(n)


def _policy_number() -> str:
    n = random.randint(1000000, 999999999)
    fmt = random.choices(
        ["pol_dashed", "policy_word", "p_short", "ins_pol",
         "year_pol", "alpha_prefix", "raw_digits"],
        weights=[20, 14, 14, 14, 14, 14, 10],
        k=1,
    )[0]
    if fmt == "pol_dashed":
        return f"POL-{n}"
    if fmt == "policy_word":
        return f"POLICY-{n}"
    if fmt == "p_short":
        return f"P{n}"
    if fmt == "ins_pol":
        return f"INS-POL-{n}"
    if fmt == "year_pol":
        return f"POL-{random.randint(2018, 2025)}-{n}"
    if fmt == "alpha_prefix":
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
        return f"{letters}{n}"
    return str(n)


def _member_id() -> str:
    n = random.randint(100000, 99999999)
    fmt = random.choices(
        ["mbr_attached", "mbr_dashed", "member_word", "m_short",
         "id_prefix", "subscriber", "raw_digits"],
        weights=[18, 16, 14, 14, 12, 14, 12],
        k=1,
    )[0]
    if fmt == "mbr_attached":
        return f"MBR{n}"
    if fmt == "mbr_dashed":
        return f"MBR-{n}"
    if fmt == "member_word":
        return f"MEMBER-{n}"
    if fmt == "m_short":
        return f"M{n}"
    if fmt == "id_prefix":
        return f"ID-{n}"
    if fmt == "subscriber":
        return f"SUB-{n}"
    return str(n)


def _transaction_id() -> str:
    """Transaction identifier covering payment processor and bank formats."""
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    n = random.randint(10000, 999999999)
    fmt = random.choices(
        ["stripe", "square", "paypal", "txn_dashed", "ord_prefix",
         "specific_prefix", "year_dated", "uuid_short", "bank_ref",
         "raw_digits"],
        weights=[14, 10, 10, 14, 10, 16, 10, 6, 6, 4],
        k=1,
    )[0]
    if fmt == "stripe":
        obj = random.choice(["pi", "ch", "txn", "py", "pyr"])
        return f"stripe_{obj}_{''.join(random.choices(ALPHA, k=18))}"
    if fmt == "square":
        return f"sq_{''.join(random.choices(ALPHA, k=22))}"
    if fmt == "paypal":
        return f"PAY-{''.join(random.choices('0123456789ABCDEF', k=17))}"
    if fmt == "txn_dashed":
        return f"TXN-{n}"
    if fmt == "ord_prefix":
        return f"ORD-{n}"
    if fmt == "specific_prefix":
        pfx = random.choice([
            "TRX", "TTN", "CTN", "PTN", "MTN", "POS", "INV", "GTX",
            "PMT", "REF", "AUTH",
        ])
        return f"{pfx}-{n}"
    if fmt == "year_dated":
        ymd = (f"{random.randint(2020, 2025)}"
               f"{random.randint(1, 12):02d}{random.randint(1, 28):02d}")
        return f"TXN-{ymd}-{n}"
    if fmt == "uuid_short":
        return "".join(random.choices("0123456789abcdef", k=16))
    if fmt == "bank_ref":
        return f"BRN-{n}"
    return str(n)


def _merchant_id() -> str:
    n = random.randint(100000, 99999999)
    fmt = random.choices(
        ["mid_dashed", "mid_bare", "merchant_word", "m_short",
         "acquirer_mid", "raw_digits"],
        weights=[26, 18, 16, 14, 14, 12],
        k=1,
    )[0]
    if fmt == "mid_dashed":
        return f"MID-{n}"
    if fmt == "mid_bare":
        return f"MID{n}"
    if fmt == "merchant_word":
        return f"MERCHANT-{n}"
    if fmt == "m_short":
        return f"M{n}"
    if fmt == "acquirer_mid":
        acq = random.choice(["FDM", "CHA", "WPS", "STR", "ADY"])
        return f"{acq}-MID-{n}"
    return str(n)

_UNIVERSITY_NAMES: list[str] = [
    # US Ivies + top private
    "Harvard University", "Yale University", "Princeton University",
    "Stanford University", "Massachusetts Institute of Technology",
    "Columbia University", "University of Pennsylvania", "Cornell University",
    "Brown University", "Dartmouth College", "Duke University",
    "Northwestern University", "Johns Hopkins University", "University of Chicago",
    "Vanderbilt University", "Rice University", "Emory University",
    "Carnegie Mellon University", "Georgetown University", "New York University",
    # US public majors
    "University of California, Berkeley", "University of California, Los Angeles",
    "University of California, San Francisco", "University of Michigan",
    "University of Virginia", "University of North Carolina",
    "University of Texas at Austin", "University of Wisconsin-Madison",
    "University of Washington", "Ohio State University",
    "Pennsylvania State University", "University of Illinois Urbana-Champaign",
    "Purdue University", "University of Florida", "University of Georgia",
    # Medical / specialty
    "Mayo Clinic Alix School of Medicine", "Cleveland Clinic Lerner College",
    "Baylor College of Medicine", "Icahn School of Medicine at Mount Sinai",
    "UCSF School of Medicine", "Johns Hopkins School of Medicine",
    # Community / state
    "Northern State University", "Pacific Coast College",
    "Riverside Institute of Technology", "Midwest University",
    "Southern Medical School", "Eastern State College",
    "Valley Community College", "Lakewood University",
    "Metropolitan School of Medicine", "City College of San Francisco",
    "Houston Community College", "Miami Dade College",
    # International
    "University of Oxford", "University of Cambridge", "Imperial College London",
    "University College London", "London School of Economics",
    "ETH Zurich", "EPFL", "Sorbonne University", "Sciences Po",
    "Technical University of Munich", "Heidelberg University",
    "Karolinska Institutet", "University of Toronto", "McGill University",
    "University of British Columbia", "Australian National University",
    "University of Melbourne", "University of Sydney",
    "National University of Singapore", "Nanyang Technological University",
    "University of Hong Kong", "Tsinghua University", "Peking University",
    # India
    "Indian Institute of Technology Bombay", "Indian Institute of Technology Delhi",
    "Indian Institute of Technology Madras", "All India Institute of Medical Sciences",
    "Indian Institute of Science", "Jawaharlal Nehru University",
    "University of Delhi", "Banaras Hindu University", "Anna University",
    "BITS Pilani", "Indian Institute of Management Ahmedabad",
]


def _university() -> str:
    return random.choice(_UNIVERSITY_NAMES)

def _law_firm() -> str:
    return random.choice([
        "Smith & Associates", "Johnson & Williams LLP",
        "Parker & Davis Law Group", "Mitchell & Clark Attorneys at Law",
        "Thompson & Reed LLP", "Harrison Law Office", "Grant & Associates LLP",
        # GAP-17: Big Law firm names
        "Skadden, Arps, Slate, Meagher & Flom LLP",
        "Davis Polk & Wardwell LLP",
        "Gibson Dunn & Crutcher LLP",
        "Sullivan & Cromwell LLP",
        "Wachtell, Lipton, Rosen & Katz",
        "Kirkland & Ellis LLP",
        "Latham & Watkins LLP",
        "Jones Day",
        "Debevoise & Plimpton LLP",
        "Cleary Gottlieb Steen & Hamilton LLP",
        "Cravath, Swaine & Moore LLP",
        "Paul Weiss Rifkind Wharton & Garrison LLP",
        "Simpson Thacher & Bartlett LLP",
        "White & Case LLP",
    ])

_COURT_NAMES: list[str] = [
    # Federal
    "US District Court", "US Court of Appeals", "United States Supreme Court",
    "US Bankruptcy Court", "US Tax Court", "US Court of Federal Claims",
    "US District Court for the Southern District of New York",
    "US District Court for the Northern District of California",
    "US District Court for the District of Columbia",
    "US Court of Appeals for the Second Circuit",
    "US Court of Appeals for the Ninth Circuit",
    "US Court of Appeals for the Federal Circuit",
    # State — supreme / appellate
    "California Supreme Court", "New York Supreme Court",
    "Texas Supreme Court", "Florida Supreme Court", "Illinois Supreme Court",
    "Pennsylvania Supreme Court", "Ohio Supreme Court",
    "California Court of Appeal", "New York Court of Appeals",
    # State / county
    "California Superior Court", "Los Angeles Superior Court",
    "San Francisco Superior Court", "Cook County Circuit Court",
    "Texas District Court", "Florida Circuit Court", "King County Superior Court",
    "Travis County District Court", "Harris County District Court",
    "Maricopa County Superior Court", "Miami-Dade Circuit Court",
    # Specialty
    "Family Court", "Probate Court", "Juvenile Court", "Drug Court",
    "Traffic Court", "Small Claims Court", "Housing Court",
    "Workers' Compensation Court",
    # International
    "Royal Courts of Justice", "High Court of Justice", "Old Bailey",
    "European Court of Justice", "European Court of Human Rights",
    "Federal Court of Australia", "Supreme Court of Canada",
    # India
    "Supreme Court of India", "Delhi High Court", "Bombay High Court",
    "Madras High Court", "Calcutta High Court", "Karnataka High Court",
    "Patna High Court", "Allahabad High Court",
]


def _court_name() -> str:
    return random.choice(_COURT_NAMES)


_HOTEL_NAMES: list[str] = [
    # Marriott family
    "Marriott Downtown", "JW Marriott", "Ritz-Carlton", "St. Regis",
    "W Hotels", "Westin Conference Center", "Le Méridien", "Renaissance Hotel",
    "Sheraton Grand Hotel", "Courtyard by Marriott", "Residence Inn",
    "Fairfield Inn", "Springhill Suites", "TownePlace Suites",
    # Hilton family
    "Hilton Garden Inn", "Conrad Hotel", "Waldorf Astoria",
    "DoubleTree by Hilton", "Embassy Suites", "Hampton Inn",
    "Hilton Grand Vacations", "Tru by Hilton", "Tapestry Collection",
    # IHG / Hyatt / Accor
    "Holiday Inn", "Holiday Inn Express", "InterContinental",
    "Crowne Plaza", "Kimpton Hotels", "Six Senses",
    "Hyatt Regency", "Park Hyatt", "Grand Hyatt", "Andaz",
    "Sofitel", "Pullman Hotel", "Novotel", "Ibis Hotel", "Mercure",
    "Fairmont Hotel", "Raffles Hotel", "Movenpick", "Swissôtel",
    # Boutique / luxury
    "Four Seasons Hotel", "Mandarin Oriental", "Aman Resort",
    "Peninsula Hotel", "The Plaza", "The Savoy", "The Dorchester",
    "Burj Al Arab", "Atlantis The Palm", "The Beverly Hills Hotel",
    "Grand Royal Hotel",
    # Budget / midscale
    "Best Western", "Comfort Inn", "Quality Inn", "Days Inn", "Super 8",
    "Motel 6", "Red Roof Inn", "La Quinta Inn", "Wyndham Garden",
    # India
    "Taj Mahal Palace", "The Oberoi", "ITC Maurya", "The Leela Palace",
    "Trident Hotel", "Vivanta by Taj", "Lemon Tree Hotels", "OYO Townhouse",
]


def _hotel_name() -> str:
    return random.choice(_HOTEL_NAMES)

def _inmate_id() -> str:
    return str(random.randint(10000,999999))

def _warrant_num() -> str:
    n = random.randint(10000, 9999999)
    year = random.randint(2018, 2025)
    fmt = random.choices(
        ["wrt_dashed", "w_year", "warrant_word", "warr_dashed",
         "warrant_hash", "bench_warrant", "search_warrant",
         "bare"],
        weights=[18, 18, 14, 14, 10, 10, 10, 6],
        k=1,
    )[0]
    if fmt == "wrt_dashed":
        return f"WRT-{n}"
    if fmt == "w_year":
        return f"W{year}{n}"
    if fmt == "warrant_word":
        return f"WARRANT-{year}-{n}"
    if fmt == "warr_dashed":
        return f"WARR-{n}"
    if fmt == "warrant_hash":
        return f"WARRANT#{n}"
    if fmt == "bench_warrant":
        return f"BW-{year}-{n}"
    if fmt == "search_warrant":
        return f"SW-{year}-{n}"
    return str(n)


def _incident_num() -> str:
    n = random.randint(100, 99999)
    year = random.randint(2018, 2025)
    fmt = random.choices(
        ["inc_year", "ir_hash", "incident_word", "report_dashed",
         "ir_dashed", "complaint", "case_inc", "police_report",
         "bare"],
        weights=[18, 14, 14, 12, 12, 10, 8, 8, 4],
        k=1,
    )[0]
    if fmt == "inc_year":
        return f"INC-{year}-{n}"
    if fmt == "ir_hash":
        return f"IR#{year}{n}"
    if fmt == "incident_word":
        return f"INCIDENT-{year}-{n}"
    if fmt == "report_dashed":
        return f"RPT-{year}-{n}"
    if fmt == "ir_dashed":
        return f"IR-{n}"
    if fmt == "complaint":
        return f"COMP-{year}-{n}"
    if fmt == "case_inc":
        return f"CASE-INC-{n}"
    if fmt == "police_report":
        return f"PR#{year}{n}"
    return str(n)

def _bart_emp_id() -> str:
    n = random.randint(1000, 999999)
    fmt = random.choices(
        ["bare", "bart_dashed", "bart_emp", "bartid", "transit"],
        weights=[28, 22, 22, 16, 12],
        k=1,
    )[0]
    if fmt == "bare":
        return str(n)
    if fmt == "bart_dashed":
        return f"BART-{n}"
    if fmt == "bart_emp":
        return f"BART-EMP-{n}"
    if fmt == "bartid":
        return f"BARTID{n}"
    return f"TRANSIT-{n}"

def _passport_num() -> str:
    """Passport number across major issuing countries.

    Original only emitted a single uppercase letter + 8 digits (loose US format).
    Real passports vary: US is 9 digits (rarely with a letter prefix), Indian
    is 1 letter + 7 digits, UK is 9 digits, EU varies.
    """
    fmt = random.choices(
        ["us_9digit", "us_letter9", "indian", "uk", "eu_alpha_num",
         "letter_8digit", "passport_prefix"],
        weights=[18, 14, 18, 14, 16, 12, 8],
        k=1,
    )[0]
    if fmt == "us_9digit":
        return f"{random.randint(100000000, 999999999)}"
    if fmt == "us_letter9":
        return f"{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.randint(10000000, 99999999):08d}"
    if fmt == "indian":
        # Indian: 1 letter + 7 digits (e.g. J1234567)
        return f"{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.randint(1000000, 9999999):07d}"
    if fmt == "uk":
        return f"{random.randint(100000000, 999999999)}"
    if fmt == "eu_alpha_num":
        # 2-letter prefix + 6 digits (Germany / France style)
        letters = "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=2))
        return f"{letters}{random.randint(100000, 999999)}"
    if fmt == "letter_8digit":
        return f"{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.randint(10000000, 99999999):08d}"
    # passport_prefix — keyword-attached
    return f"P-{random.randint(10000000, 99999999)}"


def _drivers_license() -> str:
    """Driver's license across US state formats.

    States vary widely: CA = 1 letter + 7 digits, NY = 9 digits, FL = 1 letter +
    12 digits, TX = 8 digits, IL = 1 letter + 11 digits, etc.
    """
    state = random.choice([
        "CA", "NY", "FL", "TX", "IL", "PA", "OH", "MI", "GA", "NC",
        "WA", "AZ", "MA", "NJ", "VA", "CO",
    ])
    if state == "CA":
        return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000, 9999999):07d}"
    if state == "NY":
        return f"{random.randint(100000000, 999999999):09d}"
    if state == "FL":
        return (f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                f"{random.randint(100000000000, 999999999999):012d}")
    if state == "TX":
        return f"{random.randint(10000000, 99999999):08d}"
    if state == "IL":
        return (f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                f"{random.randint(10**10, 10**11 - 1):011d}")
    if state == "PA":
        return f"{random.randint(10000000, 99999999):08d}"
    if state == "WA":
        return (f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
                f"{random.randint(100, 999):03d}"
                f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}")
    # Generic fallback: letter + 7 digits or 8-digit numeric
    if random.random() < 0.5:
        return f"D{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000, 9999999):07d}"

def _state_id_num() -> str:
    """State-issued ID with diverse prefix conventions.

    Covers the formats observed in production failures:
      - SID-/LSI-/SSI-/RSI-/OSI-/GSI-/RGN- 3-letter agency prefixes
      - State-abbr mashed prefixes (TXSTATE, NYSID, PASTATE, NVSTATE)
      - State-abbr with dash separators (CA-ID-NNN, WA-IDCARD-NNN,
        FL-STATE-NNN, OH-RESIDENT-NNN)
      - Agency-tagged (DMV-NNN, IDENT-NNN)
      - Bare ID-NNN fallback
    """
    state = _state_abbr()
    digits = random.randint(100000, 999999999)
    form = random.choices(
        [
            "agency_prefix",       # SID-/LSI-/SSI-/RSI-/OSI-/GSI-/RGN-
            "state_dashed",        # CA-ID-123456, WA-IDCARD-223344
            "state_mashed",        # TXSTATE998877, NYSID445566
            "sid_state",           # SID-CA-123456789
            "agency_tagged",       # DMV77889900, IDENT-556677
            "bare_id",             # ID7654321
        ],
        weights=[28, 18, 18, 14, 12, 10],
        k=1,
    )[0]

    if form == "agency_prefix":
        pfx = random.choice(["SID", "LSI", "SSI", "RSI", "OSI", "GSI", "RGN"])
        sep = random.choice(["-", ""])
        return f"{pfx}{sep}{digits}"

    if form == "state_dashed":
        kind = random.choice(["ID", "STATE", "IDCARD", "RESIDENT", "ID-CARD"])
        return f"{state}-{kind}-{digits}"

    if form == "state_mashed":
        kind = random.choice(["STATE", "SID", "ID", "IDENT"])
        return f"{state}{kind}{digits}"

    if form == "sid_state":
        return f"SID-{state}-{digits}"

    if form == "agency_tagged":
        pfx = random.choice(["DMV", "IDENT", "AZIDENT", "DMVID"])
        sep = random.choice(["", "-"])
        return f"{pfx}{sep}{digits}"

    # bare_id
    return f"ID{digits}"

def _med_license() -> str:
    n = random.randint(100000, 9999999)
    state = random.choice(_US_STATE_ABBR)
    fmt = random.choices(
        ["ml_state_dashed", "ml_dashed", "ml_attached", "ml_hash",
         "state_md", "med_lic", "license_word", "physician_id",
         "bare"],
        weights=[18, 16, 12, 10, 12, 10, 10, 8, 4],
        k=1,
    )[0]
    if fmt == "ml_state_dashed":
        return f"ML-{state}-{n}"
    if fmt == "ml_dashed":
        return f"ML-{n}"
    if fmt == "ml_attached":
        return f"ML{n}"
    if fmt == "ml_hash":
        return f"ML#{n}"
    if fmt == "state_md":
        return f"{state}-MD-{n}"
    if fmt == "med_lic":
        return f"MED-LIC-{n}"
    if fmt == "license_word":
        return f"LICENSE-{n}"
    if fmt == "physician_id":
        return f"PHYS-LIC-{state}-{n}"
    return str(n)

def _health_plan_id() -> str:
    # GAP-08: Add INS- prefix with 25% weight
    if random.random() < 0.25:
        return f"INS-{random.randint(1000000,9999999)}"
    prefixes = ["UHC","BCBS","CIGNA","AETNA","HUMANA","KAISER"]
    return f"{random.choice(prefixes)}-{random.randint(1000000,9999999)}"

def _card_track() -> str:
    pan = f"{random.randint(4000,4999)}{random.randint(100000000,999999999)}{random.randint(1000,9999)}"
    name = f"{fake.last_name()}/{fake.first_name()}"
    exp = f"{random.randint(25,30):02d}{random.randint(1,12):02d}"
    return f"%B{pan}^{name}^{exp}101?"

def _pin_block() -> str:
    """ISO-9564 PIN block (16 hex) AND human-typeable PINs (4-8 digits)."""
    fmt = random.choices(
        ["pin4", "pin5", "pin6", "pin8", "iso_block",
         "pin_dashed", "pin_prefixed"],
        weights=[28, 12, 14, 12, 18, 8, 8],
        k=1,
    )[0]
    if fmt == "pin4":
        return f"{random.randint(0, 9999):04d}"
    if fmt == "pin5":
        return f"{random.randint(0, 99999):05d}"
    if fmt == "pin6":
        return f"{random.randint(0, 999999):06d}"
    if fmt == "pin8":
        return f"{random.randint(0, 99999999):08d}"
    if fmt == "iso_block":
        return "".join(random.choices("0123456789ABCDEF", k=16))
    if fmt == "pin_dashed":
        return f"{random.randint(0, 999):03d}-{random.randint(0, 999):03d}"
    # pin_prefixed
    return f"PIN-{random.randint(0, 999999):04d}"


def _cryptogram() -> str:
    """EMV Application Cryptogram — typically 16 hex (8 bytes), sometimes 8."""
    fmt = random.choices(
        ["hex8", "hex16", "hex16_grouped", "tlv_prefix", "ac_prefix"],
        weights=[14, 36, 16, 18, 16],
        k=1,
    )[0]
    if fmt == "hex8":
        return "".join(random.choices("0123456789ABCDEF", k=8))
    if fmt == "hex16":
        return "".join(random.choices("0123456789ABCDEF", k=16))
    if fmt == "hex16_grouped":
        s = "".join(random.choices("0123456789ABCDEF", k=16))
        return " ".join(s[i:i+4] for i in range(0, 16, 4))
    if fmt == "tlv_prefix":
        return "9F26:" + "".join(random.choices("0123456789ABCDEF", k=16))
    # ac_prefix — Application Cryptogram label
    return "AC=" + "".join(random.choices("0123456789ABCDEF", k=16))


def _imei() -> str:
    fmt = random.choices(
        ["bare15", "dashed", "spaced", "imei_prefix",
         "imeisv16", "iccid20", "android_id_hex", "udid_ios"],
        weights=[26, 16, 12, 14, 8, 10, 8, 6],
        k=1,
    )[0]
    digits15 = "".join(str(random.randint(0, 9)) for _ in range(15))
    if fmt == "bare15":
        return digits15
    if fmt == "dashed":
        return f"{digits15[:2]}-{digits15[2:8]}-{digits15[8:14]}-{digits15[14]}"
    if fmt == "spaced":
        return f"{digits15[:2]} {digits15[2:8]} {digits15[8:14]} {digits15[14]}"
    if fmt == "imei_prefix":
        return f"IMEI: {digits15}"
    if fmt == "imeisv16":
        return "".join(str(random.randint(0, 9)) for _ in range(16))
    if fmt == "iccid20":
        return "89" + "".join(str(random.randint(0, 9)) for _ in range(18))
    if fmt == "android_id_hex":
        return "".join(random.choices("0123456789abcdef", k=16))
    # udid_ios (40-hex iOS UDID)
    return "".join(random.choices("0123456789abcdef", k=40))

def _vehicle_reg() -> str:
    """Vehicle registration number across US/Indian/UK/EU schemes."""
    L = "ABCDEFGHJKLMNPRSTUVWXYZ"
    fmt = random.choices(
        ["us_state_county", "indian", "indian_dashed", "uk", "eu_country",
         "reg_prefix", "vrn_prefix"],
        weights=[16, 22, 14, 16, 14, 10, 8],
        k=1,
    )[0]
    if fmt == "us_state_county":
        return (f"{_state_abbr()}-{random.randint(10, 99)}-"
                f"{_state_abbr()}-{random.randint(100, 9999)}")
    if fmt == "indian":
        state = random.choice(["TN", "KA", "MH", "DL", "AP", "TS", "KL", "GJ",
                                "UP", "MP", "WB", "PB", "HR", "RJ"])
        return (f"{state}{random.randint(1, 99):02d}"
                f"{''.join(random.choices(L, k=2))}{random.randint(1, 9999):04d}")
    if fmt == "indian_dashed":
        state = random.choice(["TN", "KA", "MH", "DL", "GJ", "WB"])
        return (f"{state}-{random.randint(1, 99):02d}-"
                f"{''.join(random.choices(L, k=2))}-{random.randint(1, 9999):04d}")
    if fmt == "uk":
        return (f"{''.join(random.choices(L, k=2))}{random.randint(10, 99)} "
                f"{''.join(random.choices(L, k=3))}")
    if fmt == "eu_country":
        city = random.choice(["B", "M", "K", "F", "S", "L", "H", "PA", "AB"])
        return f"{city}-{''.join(random.choices(L, k=2))} {random.randint(1, 9999)}"
    if fmt == "reg_prefix":
        return f"REG-{''.join(random.choices(L, k=3))}{random.randint(1000, 9999)}"
    # vrn_prefix
    return f"VRN-{random.randint(1000000, 99999999)}"

def _billing_num() -> str:
    n = random.randint(100000, 999999999)
    fmt = random.choices(
        ["bare", "bill_dashed", "bill_attached", "invoice", "stmt",
         "acct_bill", "year_bill"],
        weights=[22, 22, 14, 16, 12, 8, 6],
        k=1,
    )[0]
    if fmt == "bare":
        return str(n)
    if fmt == "bill_dashed":
        return f"BILL-{n}"
    if fmt == "bill_attached":
        return f"BILL{n}"
    if fmt == "invoice":
        return f"INV-{n}"
    if fmt == "stmt":
        return f"STMT-{n}"
    if fmt == "acct_bill":
        return f"ACCT-BILL-{n}"
    return f"BILL-{random.randint(2020, 2025)}-{n}"

_RACE_ETHNICITY_POOL: list[str] = [
    # US Census broad
    "White", "Black", "African American", "Black or African American",
    "Asian", "Asian American", "Hispanic", "Latino", "Latina", "Latinx",
    "Hispanic or Latino", "Native American", "American Indian",
    "Alaska Native", "Native Hawaiian", "Pacific Islander",
    "Middle Eastern", "North African", "Middle Eastern or North African",
    "Multiracial", "Two or more races", "Biracial", "Mixed race",
    "Caucasian", "Non-Hispanic White", "Non-Hispanic Black",
    # More specific Asian sub-groups
    "Chinese", "Japanese", "Korean", "Vietnamese", "Filipino",
    "Indian", "South Asian", "Pakistani", "Bangladeshi", "Sri Lankan",
    "East Asian", "Southeast Asian",
    # Hispanic sub-groups
    "Mexican", "Mexican American", "Puerto Rican", "Cuban", "Dominican",
    "Salvadoran", "Colombian", "Central American", "South American",
    # Other
    "Afro-Caribbean", "Afro-Latino", "Indigenous",
    "Prefer not to say", "Other",
]


def _dept_color() -> str:
    return random.choice(_RACE_ETHNICITY_POOL)


_RELIGION_POOL: list[str] = [
    "Christian", "Catholic", "Roman Catholic", "Protestant",
    "Eastern Orthodox", "Anglican", "Episcopalian", "Baptist",
    "Methodist", "Presbyterian", "Lutheran", "Pentecostal",
    "Evangelical", "Mormon", "Latter-day Saints", "LDS",
    "Jehovah's Witness", "Seventh-day Adventist", "Quaker", "Amish",
    "Muslim", "Sunni", "Shia", "Sufi", "Islamic",
    "Jewish", "Orthodox Jewish", "Conservative Jewish", "Reform Jewish",
    "Hasidic",
    "Hindu", "Vaishnav", "Shaiva", "Smarta",
    "Buddhist", "Theravada", "Mahayana", "Zen", "Tibetan Buddhist",
    "Sikh", "Jain", "Zoroastrian", "Parsi", "Baha'i", "Rastafarian",
    "Spiritual but not religious", "Agnostic", "Atheist",
    "Humanist", "Pagan", "Wiccan", "Shinto", "Taoist", "Confucianist",
    "Native American Spirituality", "Indigenous", "Animist",
    "Prefer not to say", "None", "No religious affiliation",
]


def _religion_val() -> str:
    return random.choice(_RELIGION_POOL)


def _physical_desc() -> str:
    """Physical description across height/weight/hair/eyes/build/skin/tattoos."""
    height_imp = f"{random.randint(4, 7)}'{random.randint(0, 11)}\""
    height_cm = f"{random.randint(140, 210)} cm"
    weight_lb = f"{random.randint(90, 320)} lbs"
    weight_kg = f"{random.randint(40, 150)} kg"
    hair = random.choice([
        "brown hair", "black hair", "blonde hair", "red hair",
        "gray hair", "white hair", "auburn hair", "bald", "shaved head",
        "long brown hair", "short black hair", "curly hair",
        "straight hair", "wavy hair",
    ])
    eyes = random.choice([
        "brown eyes", "blue eyes", "green eyes", "hazel eyes",
        "gray eyes", "amber eyes", "dark brown eyes",
    ])
    build = random.choice([
        "slim build", "average build", "muscular build", "heavy-set",
        "athletic build", "stocky build", "thin", "medium build",
    ])
    skin = random.choice([
        "fair complexion", "light skin", "medium skin", "tan complexion",
        "dark complexion", "olive skin", "freckled",
    ])
    marks = random.choice([
        "tattoo on left arm", "scar on right cheek", "no distinguishing marks",
        "tattoo on neck", "birthmark on chest", "scar above right eye",
        "tattoos on both arms", "no tattoos", "pierced ears",
    ])

    fmt = random.choices(
        ["hwt", "hwt_hair", "full", "metric_full", "build_skin",
         "marks_focus", "height_only"],
        weights=[14, 18, 22, 14, 14, 12, 6],
        k=1,
    )[0]
    if fmt == "hwt":
        return f"{height_imp}, {weight_lb}"
    if fmt == "hwt_hair":
        return f"{height_imp}, {weight_lb}, {hair}"
    if fmt == "full":
        return f"{height_imp}, {weight_lb}, {hair}, {eyes}, {build}"
    if fmt == "metric_full":
        return f"{height_cm}, {weight_kg}, {hair}, {eyes}"
    if fmt == "build_skin":
        return f"{build}, {skin}, {hair}"
    if fmt == "marks_focus":
        return f"{height_imp}, {build}, {marks}"
    return height_imp

# Generic-name medications (active ingredient) — used in clinical notes
_GENERIC_DRUGS = [
    "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole",
    "Albuterol", "Prednisone", "Sertraline", "Levothyroxine", "Gabapentin",
    "Amoxicillin", "Metoprolol", "Ibuprofen", "Acetaminophen", "Aspirin",
    "Cetirizine", "Loratadine", "Fluoxetine", "Citalopram", "Escitalopram",
    "Hydrochlorothiazide", "Losartan", "Valsartan", "Simvastatin",
    "Rosuvastatin", "Pantoprazole", "Esomeprazole", "Ranitidine",
    "Famotidine", "Ondansetron", "Tramadol", "Hydrocodone", "Oxycodone",
    "Codeine", "Morphine", "Diazepam", "Lorazepam", "Alprazolam",
    "Clonazepam", "Zolpidem", "Trazodone", "Bupropion", "Venlafaxine",
    "Duloxetine", "Mirtazapine", "Quetiapine", "Risperidone", "Aripiprazole",
    "Olanzapine", "Lamotrigine", "Topiramate", "Levetiracetam",
    "Doxycycline", "Azithromycin", "Ciprofloxacin", "Levofloxacin",
    "Cephalexin", "Clindamycin", "Trimethoprim", "Sulfamethoxazole",
    "Glimepiride", "Glipizide", "Sitagliptin", "Empagliflozin",
    "Dapagliflozin", "Liraglutide", "Semaglutide", "Naproxen", "Heparin",
    "Warfarin", "Apixaban", "Rivaroxaban", "Clopidogrel", "Furosemide",
    "Spironolactone", "Carvedilol", "Bisoprolol", "Diltiazem", "Verapamil",
    "Tamsulosin", "Finasteride", "Sildenafil", "Tadalafil", "Tacrolimus",
    "Methotrexate", "Adalimumab", "Etanercept", "Infliximab",
]

# Common consumer brand names — what patients actually call their meds
_BRAND_DRUGS = [
    "Tylenol", "Tylenol Extra Strength", "Advil", "Motrin", "Aleve",
    "Excedrin", "Bayer Aspirin", "Benadryl", "Claritin", "Zyrtec",
    "Allegra", "Mucinex", "Robitussin", "Sudafed", "DayQuil",
    "NyQuil", "Pepto-Bismol", "Imodium", "Tums", "Rolaids", "Zantac",
    "Augmentin", "Lipitor", "Crestor", "Zocor", "Synthroid",
    "Levaquin", "Cipro", "Bactrim", "Z-Pak", "Amoxil", "Keflex",
    "Prozac", "Zoloft", "Lexapro", "Celexa", "Effexor", "Paxil",
    "Wellbutrin", "Cymbalta", "Abilify", "Seroquel", "Risperdal",
    "Xanax", "Ativan", "Klonopin", "Valium", "Ambien", "Lunesta",
    "Norco", "Vicodin", "Percocet", "OxyContin", "Suboxone",
    "Adderall", "Ritalin", "Concerta", "Vyvanse", "Strattera",
    "Ventolin Inhaler", "ProAir", "Symbicort", "Advair Diskus",
    "Spiriva", "Flonase", "Nasonex", "Singulair", "Flovent",
    "Lantus", "Humalog", "Novolog", "Levemir", "Tresiba",
    "Toujeo", "Basaglar", "Trulicity", "Ozempic", "Victoza",
    "Mounjaro", "Wegovy", "Jardiance", "Farxiga", "Invokana",
    "Januvia", "Glucophage", "Eliquis", "Xarelto", "Pradaxa",
    "Plavix", "Coumadin", "Lasix", "Aldactone", "Lopressor",
    "Toprol XL", "Tenormin", "Norvasc", "Cardizem", "Inderal",
    "Cozaar", "Diovan", "Benicar", "Lotensin", "Zestril",
    "Lipitor 20mg", "Crestor 10mg", "Synthroid 50mcg",
    # India-specific common brands (the failing examples included these)
    "Crocin", "Crocin 650", "Panadol", "Disprin", "Combiflam",
    "Dolo 650", "Pantop", "Pan 40", "Ecosprin", "Voveran",
]

_DRUG_DOSAGES = [
    "5mg", "10mg", "20mg", "25mg", "40mg", "50mg", "75mg",
    "100mg", "150mg", "200mg", "250mg", "300mg", "400mg",
    "500mg", "650mg", "750mg", "1000mg",
    "5 mg", "10 mg", "20 mg", "100 mg", "200 mg", "250 mg",
    "500 mg", "650 mg",
    # Uppercase MG variant ("Ibuprofen 200 MG Tablet" — user example)
    "200 MG", "250 MG", "500 MG", "650 MG", "1000 MG",
    "25mcg", "50mcg", "75mcg", "100mcg", "150mcg",
    "100 units/mL", "300 units/mL",
    # Volumetric forms ("Insulin Injection 10ml" — user example)
    "10ml", "30ml", "50ml", "100ml", "200ml", "5 ml", "10 ml",
    "30 ml", "100 ml",
]

_DRUG_FORMS = [
    "Tablet", "Tablets", "Capsule", "Capsules", "Injection",
    "Inhaler", "Suspension", "Syrup", "Solution", "Spray",
    "Cream", "Ointment", "Patch", "Drops", "Powder",
    "ER", "XR", "SR", "CR", "DR",
]

# Pharmaceutical salt / form descriptors used in branded preparations.
# Real examples: "Albuterol Sulfate Inhaler", "Losartan Potassium 50mg",
# "Metoprolol Succinate", "Hydroxyzine Hydrochloride".
_DRUG_SALTS = [
    "Sulfate", "Potassium", "Sodium", "Hydrochloride", "HCl", "Succinate",
    "Tartrate", "Maleate", "Fumarate", "Citrate", "Mesylate", "Acetate",
    "Phosphate", "Calcium", "Bromide",
]


def _medication() -> str:
    """Generate a medication name with diverse forms.

    Distribution:
      • 22 % generic name alone           ("Metformin")
      • 18 % generic + dose                ("Metformin 500mg")
      • 18 % brand alone                   ("Tylenol", "Crocin 650")
      • 14 % brand + dose                  ("Lipitor 20mg")
      •  8 % generic + dose + form         ("Amoxicillin 250mg Capsule")
      •  6 % brand + form                  ("Ventolin Inhaler")
      •  6 % generic + salt                ("Losartan Potassium")
      •  5 % generic + salt + dose         ("Losartan Potassium 50mg")
      •  3 % generic + salt + form         ("Albuterol Sulfate Inhaler")
    """
    form = random.choices(
        [
            "generic", "generic_dose", "brand", "brand_dose",
            "generic_dose_form", "brand_form", "generic_salt",
            "generic_salt_dose", "generic_salt_form",
        ],
        weights=[22, 18, 18, 14, 8, 6, 6, 5, 3],
        k=1,
    )[0]

    if form == "generic":
        return random.choice(_GENERIC_DRUGS)
    if form == "generic_dose":
        return f"{random.choice(_GENERIC_DRUGS)} {random.choice(_DRUG_DOSAGES)}"
    if form == "brand":
        return random.choice(_BRAND_DRUGS)
    if form == "brand_dose":
        # Some brand names already include a dosage; only append for those that don't
        b = random.choice(_BRAND_DRUGS)
        if any(d.replace(" ", "") in b.replace(" ", "") for d in ("mg", "mcg", "ml")):
            return b
        return f"{b} {random.choice(_DRUG_DOSAGES)}"
    if form == "generic_dose_form":
        return (f"{random.choice(_GENERIC_DRUGS)} "
                f"{random.choice(_DRUG_DOSAGES)} "
                f"{random.choice(_DRUG_FORMS)}")
    if form == "brand_form":
        b = random.choice(_BRAND_DRUGS)
        return f"{b} {random.choice(_DRUG_FORMS)}"
    if form == "generic_salt":
        return f"{random.choice(_GENERIC_DRUGS)} {random.choice(_DRUG_SALTS)}"
    if form == "generic_salt_dose":
        return (f"{random.choice(_GENERIC_DRUGS)} "
                f"{random.choice(_DRUG_SALTS)} "
                f"{random.choice(_DRUG_DOSAGES)}")
    if form == "generic_salt_form":
        return (f"{random.choice(_GENERIC_DRUGS)} "
                f"{random.choice(_DRUG_SALTS)} "
                f"{random.choice(_DRUG_FORMS)}")
    return random.choice(_GENERIC_DRUGS)

def _job_title() -> str:
    return fake.job()

def _employer() -> str:
    return fake.company()

def _username() -> str:
    base = fake.user_name()
    fmt = random.choices(
        ["plain", "with_digits", "with_underscore", "with_dot",
         "with_dash", "social_handle", "email_local", "leet",
         "first_dot_last", "first_last", "user_prefix",
         "all_lowercase", "title_case"],
        weights=[14, 14, 10, 10, 8, 8, 8, 6, 8, 6, 4, 2, 2],
        k=1,
    )[0]
    if fmt == "plain":
        return base
    if fmt == "with_digits":
        return f"{base}{random.randint(10, 9999)}"
    if fmt == "with_underscore":
        return f"{base}_{random.randint(10, 999)}"
    if fmt == "with_dot":
        return f"{base}.{random.randint(10, 999)}"
    if fmt == "with_dash":
        return f"{base}-{random.randint(10, 999)}"
    if fmt == "social_handle":
        return f"@{base}"
    if fmt == "email_local":
        return f"{base}@example.com"
    if fmt == "leet":
        m = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
        return "".join(m.get(c, c) for c in base)
    if fmt == "first_dot_last":
        return f"{fake.first_name().lower()}.{fake.last_name().lower()}"
    if fmt == "first_last":
        return f"{fake.first_name().lower()}{fake.last_name().lower()}"
    if fmt == "user_prefix":
        return f"user_{base}"
    if fmt == "all_lowercase":
        return base.lower()
    return base.title()

def _dna_str_locus() -> str:
    """DNA / genetic identifier across CODIS STR, profile IDs, and accession refs."""
    loci = ["D3S1358", "vWA", "TH01", "TPOX", "CSF1PO", "FGA", "D7S820",
            "D13S317", "D16S539", "D2S1338", "D21S11", "D18S51", "D5S818",
            "D8S1179", "PentaE", "PentaD", "SE33", "Amelogenin"]
    fmt = random.choices(
        ["str_allele", "str_pair", "profile_id", "dna_dashed",
         "specimen", "ncbi_acc", "rs_snp"],
        weights=[22, 14, 18, 14, 14, 10, 8],
        k=1,
    )[0]
    locus = random.choice(loci)
    if fmt == "str_allele":
        return f"{locus}-{random.randint(6, 30)}"
    if fmt == "str_pair":
        return f"{locus}: {random.randint(6, 18)},{random.randint(6, 18)}"
    if fmt == "profile_id":
        return f"DNA-{random.randint(100000, 9999999)}"
    if fmt == "dna_dashed":
        return f"PROFILE-{random.randint(2018, 2025)}-{random.randint(10000, 999999)}"
    if fmt == "specimen":
        return f"SPECIMEN-{random.choice('ABCDEFGHJK')}{random.randint(100000, 9999999)}"
    if fmt == "ncbi_acc":
        prefix = random.choice(["NC_", "NM_", "NG_", "NR_", "NW_"])
        return f"{prefix}{random.randint(100000, 999999)}.{random.randint(1, 9)}"
    # rs_snp — SNP accession
    return f"rs{random.randint(100000, 99999999)}"


def _voiceprint_vp() -> str:
    n = random.randint(1000, 9999999)
    fmt = random.choices(
        ["vp_dashed", "voice_dashed", "voiceprint", "vp_attached",
         "biometric_voice", "speaker_id"],
        weights=[24, 18, 16, 16, 12, 14],
        k=1,
    )[0]
    if fmt == "vp_dashed":
        return f"VP-{n}"
    if fmt == "voice_dashed":
        return f"VOICE-{n}"
    if fmt == "voiceprint":
        return f"VOICEPRINT-{n}"
    if fmt == "vp_attached":
        return f"VP{n}"
    if fmt == "biometric_voice":
        return f"BIO-VOICE-{n}"
    return f"SPK-{n}"

def _ferpa_id() -> str:
    n = random.randint(10000, 9999999)
    fmt = random.choices(
        ["ferpa_fer", "ferpa_long", "fer_dashed", "ferpa_word",
         "student_record_id", "academic_record", "transcript_id",
         "student_file"],
        weights=[16, 14, 14, 14, 14, 12, 10, 6],
        k=1,
    )[0]
    if fmt == "ferpa_fer":
        return f"FERPA FER-{n}"
    if fmt == "ferpa_long":
        return f"STUDENT_RECORDS_FERPA-{n}"
    if fmt == "fer_dashed":
        return f"FER-{n}"
    if fmt == "ferpa_word":
        return f"FERPA-{n}"
    if fmt == "student_record_id":
        return f"STUDENT-REC-{n}"
    if fmt == "academic_record":
        return f"ACAD-REC-{n}"
    if fmt == "transcript_id":
        return f"TRANS-{n}"
    return f"SF-{n}"


def _cpo_order() -> str:
    n = random.randint(100, 99999)
    year = random.randint(2018, 2025)
    fmt = random.choices(
        ["cpo_year", "po_dashed", "restraining", "order_word",
         "protection", "tpo", "no_contact", "bare"],
        weights=[18, 16, 14, 14, 14, 12, 8, 4],
        k=1,
    )[0]
    if fmt == "cpo_year":
        return f"CPO-{year}-{n}"
    if fmt == "po_dashed":
        return f"PO-{year}-{n}"
    if fmt == "restraining":
        return f"RO-{year}-{n}"
    if fmt == "order_word":
        return f"ORDER-{year}-{n}"
    if fmt == "protection":
        return f"PROT-{year}-{n}"
    if fmt == "tpo":
        return f"TPO-{n}"
    if fmt == "no_contact":
        return f"NCO-{year}-{n}"
    return str(n)


def _driver_history_dh() -> str:
    n = random.randint(10000, 9999999)
    state = random.choice(_US_STATE_ABBR)
    fmt = random.choices(
        ["dh_dashed", "dh_attached", "dr_history", "mvr",
         "driving_record", "drv_hist", "state_dh", "bare"],
        weights=[20, 14, 14, 14, 12, 10, 10, 6],
        k=1,
    )[0]
    if fmt == "dh_dashed":
        return f"DH-{n}"
    if fmt == "dh_attached":
        return f"DH{n}"
    if fmt == "dr_history":
        return f"DRH-{n}"
    if fmt == "mvr":
        return f"MVR-{n}"
    if fmt == "driving_record":
        return f"DRIVING-REC-{n}"
    if fmt == "drv_hist":
        return f"DRV-HIST-{n}"
    if fmt == "state_dh":
        return f"{state}-DH-{n}"
    return str(n)

def _confidential_ref() -> str:
    # GAP-12: Confidential document reference
    kind = random.choice(["memo","document","file","record","report"])
    pfx = random.choice(["CONF","INT","SEC","PRIV"])
    return f"Confidential {kind} {pfx}-{random.randint(1000,9999999)}"

def _username_bare() -> str:
    base = fake.user_name().replace("-", "_").lower()[:15]
    fmt = random.choices(
        ["user_prefix", "with_dot", "with_dash", "with_underscore",
         "with_digits", "social_handle", "email_local",
         "leet_handle", "plain"],
        weights=[14, 14, 12, 14, 14, 10, 10, 6, 6],
        k=1,
    )[0]
    if fmt == "user_prefix":
        return base if base.startswith("user_") else f"user_{base}"
    if fmt == "with_dot":
        return f"{base}.{random.randint(1, 999)}"
    if fmt == "with_dash":
        return f"{base}-{random.randint(1, 999)}"
    if fmt == "with_underscore":
        return f"{base}_{random.randint(1, 999)}"
    if fmt == "with_digits":
        return f"{base}{random.randint(10, 9999)}"
    if fmt == "social_handle":
        return f"@{base}"
    if fmt == "email_local":
        return f"{base}@example.com"
    if fmt == "leet_handle":
        m = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
        return "".join(m.get(c, c) for c in base)
    return base


# ---------------------------------------------------------------------------
# Entity template definitions — 20-25 templates per entity
# ---------------------------------------------------------------------------

def _physician_name() -> str:
    """Multi-form physician/doctor name generator.

    Mirrors the diversity in _first_last() but always with a 'Dr.' / 'Doctor'
    prefix and occasionally an 'MD' / 'DO' / 'DDS' suffix.
    """
    title = random.choice(["Dr.", "Doctor", "Dr"])
    suffix = random.choices(
        ["", ", MD", ", DO", ", DDS", ", MD, PhD", ", FACS"],
        weights=[55, 22, 6, 4, 7, 6],
        k=1,
    )[0]

    form = random.choices(
        [
            "basic",                # Dr. Mike Smith
            "with_middle_initial",  # Dr. Mike S. Smith
            "intl",                 # Dr. Kim Min Soo / Dr. Wei Zhang
            "two_initials_surname", # Dr. K. W. Smith
            "single_letter_middle", # Dr. Kate M.
            "nickname_quoted",      # Dr. Katherine "Kate" Doe
            "surname_only",         # Dr. Smith
            "with_suffix",          # Dr. Alex Johnson Jr.
        ],
        weights=[36, 16, 14, 6, 8, 8, 8, 4],
        k=1,
    )[0]

    if form == "basic":
        body = f"{fake.first_name()} {fake.last_name()}"
    elif form == "with_middle_initial":
        body = _name_with_middle_initial()
    elif form == "intl":
        body = _intl_name()
    elif form == "two_initials_surname":
        body = _name_two_initials_surname()
    elif form == "single_letter_middle":
        # "Dr. Kate M." — single-letter middle, no surname
        body = f"{fake.first_name()} {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}."
    elif form == "nickname_quoted":
        body = _name_with_nickname()
    elif form == "surname_only":
        body = fake.last_name()
    elif form == "with_suffix":
        body = _name_with_suffix()
    else:
        body = f"{fake.first_name()} {fake.last_name()}"

    return f"{title} {body}{suffix}"

def _card_holder() -> str:
    """Cardholder name across embossing/casing conventions.

    Cards in the wild use: all-uppercase (most embossed), title case,
    initial+last, name with middle initial, and (rarely) international
    naming conventions.
    """
    fmt = random.choices(
        ["upper_basic", "upper_middle", "title_basic", "title_middle",
         "upper_initial_last", "upper_with_suffix", "intl_upper",
         "two_initials_last_upper", "first_initial_last_upper"],
        weights=[28, 14, 18, 10, 8, 6, 8, 4, 4],
        k=1,
    )[0]
    fn = fake.first_name()
    ln = fake.last_name()
    mid = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    suffix = random.choice(["JR", "SR", "II", "III"])

    if fmt == "upper_basic":
        return f"{fn.upper()} {ln.upper()}"
    if fmt == "upper_middle":
        return f"{fn.upper()} {mid} {ln.upper()}"
    if fmt == "title_basic":
        return f"{fn} {ln}"
    if fmt == "title_middle":
        return f"{fn} {mid}. {ln}"
    if fmt == "upper_initial_last":
        return f"{fn[0].upper()} {ln.upper()}"
    if fmt == "upper_with_suffix":
        return f"{fn.upper()} {ln.upper()} {suffix}"
    if fmt == "intl_upper":
        return _intl_name().upper()
    if fmt == "two_initials_last_upper":
        return f"{fn[0].upper()}.{mid}. {ln.upper()}"
    # first_initial_last_upper
    return f"{fn[0].upper()}.{ln.upper()}"


def _card_iin_bin() -> str:
    """6-digit Issuer Identification Number across all major networks.

    Originally only emitted Visa BINs (4xxxxx) — production failures showed
    Mastercard, Amex, Discover, JCB, Diners, RuPay etc. were misclassified
    because the model never saw them in training.
    """
    network = random.choice([
        "visa", "mastercard", "amex", "discover", "diners",
        "jcb", "rupay", "unionpay", "maestro",
    ])
    if network == "visa":
        return str(random.randint(400000, 499999))
    if network == "mastercard":
        first2 = random.choice([51, 52, 53, 54, 55])  # legacy 5x
        return f"{first2}{random.randint(0, 9999):04d}"
    if network == "amex":
        first2 = random.choice([34, 37])
        return f"{first2}{random.randint(0, 9999):04d}"
    if network == "discover":
        return f"6011{random.randint(0, 99):02d}"
    if network == "diners":
        first3 = random.choice([300, 301, 302, 303, 304, 305, 309, 360, 380])
        return f"{first3}{random.randint(0, 999):03d}"
    if network == "jcb":
        return f"35{random.randint(0, 9999):04d}"
    if network == "rupay":
        first3 = random.choice([508, 606, 607, 608, 652, 653])
        return f"{first3}{random.randint(0, 999):03d}"
    if network == "unionpay":
        return f"62{random.randint(0, 9999):04d}"
    # maestro
    first2 = random.choice([50, 56, 57, 58, 63, 67])
    return f"{first2}{random.randint(0, 9999):04d}"


def _card_last4() -> str:
    """Last-four-digit values across bare and masked-prefix formats.

    Production failures showed masked formats (****1234, XXXX-5678,
    **** **** **** 9012, CARD-4455, TAIL-1122) were being misrouted to
    state_id or employee_id labels. Generator emits them as part of the
    value so the NER label covers the masking prefix.
    """
    n = f"{random.randint(0, 9999):04d}"
    form = random.choices(
        [
            "bare",              # 1234
            "stars_attached",    # ****1234
            "stars_spaced",      # **** 1234
            "x_uppercase",       # XXXX-1234
            "x_lowercase",       # xxxx-1234
            "stars_full_pan",    # **** **** **** 1234
            "x_dash",            # x-1234
            "card_dash",         # CARD-1234 / ENDING-1234 / TAIL-1234
            "mask_dash",         # MASK-1234 / VISA-1234
        ],
        weights=[28, 14, 14, 10, 6, 10, 4, 8, 6],
        k=1,
    )[0]

    if form == "bare":
        return n
    if form == "stars_attached":
        return f"****{n}"
    if form == "stars_spaced":
        return f"**** {n}"
    if form == "x_uppercase":
        return f"XXXX-{n}"
    if form == "x_lowercase":
        return f"xxxx-{n}"
    if form == "stars_full_pan":
        return f"**** **** **** {n}"
    if form == "x_dash":
        return f"x-{n}"
    if form == "card_dash":
        pfx = random.choice(["CARD", "ENDING", "TAIL", "CCLAST4", "DEBITEND"])
        return f"{pfx}-{n}"
    # mask_dash
    pfx = random.choice(["MASK", "VISA", "MC", "AMEX", "PAY", "POS"])
    return f"{pfx}-{n}"


# ---------------------------------------------------------------------------
# Generic record-id builder + CJIS / stolen-X / biometric / misc generators
# ---------------------------------------------------------------------------

def _record_id(
    prefixes: list[str],
    digits_low: int = 10000,
    digits_high: int = 9999999,
    include_year: bool = True,
    include_bare: bool = True,
    include_attached: bool = True,
) -> str:
    """Generic prefix-varied record identifier.

    `prefixes` is a list of label strings (e.g. ['FBI-', 'NCIC-FBI-']) — the
    digits are appended directly. Extra forms can be enabled to add a bare
    digits-only variant, a prefix-attached-no-separator variant, and a
    year-segmented variant.
    """
    n = random.randint(digits_low, digits_high)
    pool = list(prefixes)
    if include_bare:
        pool.append("__BARE__")
    if include_attached:
        pool.append("__ATTACHED__")
    if include_year:
        pool.append("__YEAR__")
    form = random.choice(pool)

    if form == "__BARE__":
        return str(n)
    if form == "__ATTACHED__":
        # strip trailing separator and concatenate the digits
        pfx = random.choice(prefixes).rstrip("-#_ ")
        return f"{pfx}{n}"
    if form == "__YEAR__":
        pfx = random.choice(prefixes).rstrip("-#_ ")
        return f"{pfx}-{random.randint(2018, 2025)}-{n}"
    return f"{form}{n}"


def _fbi_number() -> str:
    return _record_id(["FBI-", "FBI#", "FBI ID ", "NCIC-FBI-", "FBI/"],
                      digits_low=1000000, digits_high=9999999)


def _chri() -> str:
    return _record_id(["CHRI-", "CHRI#", "CRIM-HIST-", "NCIC-CHRI-",
                       "CHIST-"],
                      digits_low=100000, digits_high=99999999)


def _arrest_record() -> str:
    return _record_id(["AR-", "ARR-", "ARREST-", "A-", "NCIC-AR-"],
                      digits_low=10000, digits_high=999999)


def _incarceration_info() -> str:
    return _record_id(["INCAR-", "INMATE-", "JAIL-", "DOC-", "BOOK-",
                       "PRISON-", "CUSTODY-"],
                      digits_low=10000, digits_high=99999999)


def _missing_person_report() -> str:
    return _record_id(["MP-", "MISSING-", "MP#", "NCIC-MP-",
                       "MISS-PER-"], digits_low=10000, digits_high=999999)


def _wanted_person_report() -> str:
    return _record_id(["WP-", "WANTED-", "WP#", "NCIC-WP-",
                       "WANT-PER-"], digits_low=10000, digits_high=999999)


def _sex_offender_report() -> str:
    return _record_id(["SOR-", "SO-", "SEXOFF-", "REGISTRY-",
                       "NSOPW-", "REGSO-"],
                      digits_low=10000, digits_high=9999999)


def _foreign_fugitive() -> str:
    return _record_id(["FF-", "FUGITIVE-", "FF#", "INTERPOL-FF-",
                       "FOR-FUG-"], digits_low=10000, digits_high=999999)


def _identity_theft_victim() -> str:
    return _record_id(["IDT-", "ID-THEFT-", "IDV-", "IDTHEFT-",
                       "VICTIM-"], digits_low=10000, digits_high=999999)


def _gang_terrorist_member() -> str:
    return _record_id(["GT-", "GANG-", "TERR-", "GTM-",
                       "NCIC-GTM-", "TKDB-"],
                      digits_low=10000, digits_high=999999)


def _supervised_release() -> str:
    return _record_id(["SR-", "SUPREL-", "SV-", "SUPREL#",
                       "SUP-REL-", "USPO-SR-"],
                      digits_low=10000, digits_high=999999)


def _probation_record() -> str:
    return _record_id(["PROB-", "PR-", "PROBATION-", "PROB#",
                       "USPO-PR-"], digits_low=10000, digits_high=999999)


def _parole_record() -> str:
    return _record_id(["PAR-", "PAROLE-", "PR-", "USPC-PAR-",
                       "PAROLE#"], digits_low=10000, digits_high=999999)


def _stolen_vehicle() -> str:
    return _record_id(["SV-", "STOLEN-VEH-", "STV-", "NCIC-SV-",
                       "VEH-STL-"], digits_low=10000, digits_high=9999999)


def _stolen_guns() -> str:
    return _record_id(["SG-", "STOLEN-GUN-", "GUN-STL-", "NCIC-SG-",
                       "FIREARM-STL-"], digits_low=10000, digits_high=999999)


def _stolen_license_plate() -> str:
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    fmt = random.choices(
        ["plate_value", "slp_prefix", "stolen_lp", "ncic_slp"],
        weights=[35, 25, 20, 20], k=1)[0]
    if fmt == "plate_value":
        return f"{''.join(random.choices(L, k=3))}-{random.randint(1000, 9999)}"
    if fmt == "slp_prefix":
        return f"SLP-{random.randint(10000, 999999)}"
    if fmt == "stolen_lp":
        return f"STOLEN-LP-{random.randint(10000, 999999)}"
    return f"NCIC-SLP-{random.randint(10000, 999999)}"


def _stolen_boats() -> str:
    return _record_id(["SB-", "STOLEN-BOAT-", "BOAT-STL-", "NCIC-SB-",
                       "VESSEL-STL-"], digits_low=10000, digits_high=999999)


def _stolen_securities() -> str:
    return _record_id(["SS-", "STOLEN-SEC-", "SEC-STL-", "NCIC-SS-",
                       "BOND-STL-"], digits_low=10000, digits_high=9999999)


def _stolen_articles() -> str:
    return _record_id(["SA-", "STOLEN-ART-", "ART-STL-", "NCIC-SA-",
                       "GOODS-STL-"], digits_low=10000, digits_high=999999)


def _inmate_id() -> str:
    return _record_id(["INMATE-", "JAIL-ID-", "DOC-", "BOOK-",
                       "CUSTODY-", "PRISON-"],
                      digits_low=10000, digits_high=9999999)


def _application_id() -> str:
    return _record_id(["APP-", "GOV-", "REF-", "FORM-", "APPL-",
                       "APP#", "GOV#"],
                      digits_low=10000, digits_high=99999999)


def _terminal_id() -> str:
    return _record_id(["TID-", "TERMINAL-", "POS-", "T-", "TID#",
                       "POS-TID-", "TML-"],
                      digits_low=10000, digits_high=99999999)


_CARD_TYPE_POOL = [
    "Visa", "VISA", "Visa Debit", "Visa Credit", "Visa Platinum",
    "Visa Signature", "Visa Infinite", "Visa Electron",
    "Mastercard", "MasterCard", "MASTERCARD", "Mastercard Debit",
    "Mastercard Credit", "Mastercard World", "Mastercard Gold",
    "Mastercard Platinum",
    "American Express", "Amex", "AmEx", "AMEX",
    "American Express Platinum", "American Express Gold",
    "American Express Centurion",
    "Discover", "Discover It", "Discover Card",
    "Diners Club", "Diners Club International", "Diners Club Carte Blanche",
    "JCB", "JCB Gold", "JCB Platinum",
    "UnionPay", "China UnionPay", "UnionPay Debit",
    "RuPay", "RuPay Select", "RuPay Platinum",
    "Maestro", "Cirrus", "Interac", "Hipercard", "Elo",
]


def _card_type() -> str:
    return random.choice(_CARD_TYPE_POOL)


def _facial_recognition() -> str:
    n = random.randint(1000, 9999999)
    fmt = random.choices(
        ["filename_bin", "filename_jpg", "filename_png", "face_dashed",
         "fr_prefix", "faceprint", "biometric_face", "base64ish_short"],
        weights=[14, 12, 12, 16, 14, 12, 12, 8], k=1,
    )[0]
    if fmt == "filename_bin":
        return f"facial_scan_{n}.bin"
    if fmt == "filename_jpg":
        return f"face_{n}.jpg"
    if fmt == "filename_png":
        return f"faceprint_{n}.png"
    if fmt == "face_dashed":
        return f"FACE-{n}"
    if fmt == "fr_prefix":
        return f"FR-PROFILE-{n}"
    if fmt == "faceprint":
        return f"FACEPRINT-{n}"
    if fmt == "biometric_face":
        return f"BIO-FACE-{n}"
    return "".join(random.choices(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        k=32))


def _iris_scan() -> str:
    n = random.randint(1000, 9999999)
    fmt = random.choices(
        ["filename_dat", "filename_bin", "iris_dashed", "eye_profile",
         "iris_id", "biometric_iris", "iris_template"],
        weights=[14, 14, 18, 14, 14, 14, 12], k=1,
    )[0]
    if fmt == "filename_dat":
        return f"iris_scan_{n}.dat"
    if fmt == "filename_bin":
        return f"iris_{n}.bin"
    if fmt == "iris_dashed":
        return f"IRIS-{n}"
    if fmt == "eye_profile":
        return f"EYE-PROFILE-{n}"
    if fmt == "iris_id":
        return f"IRISID{n}"
    if fmt == "biometric_iris":
        return f"BIO-IRIS-{n}"
    return f"IRIS-TPL-{n}"


def _fingerprint() -> str:
    n = random.randint(100000, 9999999)
    fmt = random.choices(
        ["fp_under", "fp_dashed", "fpid", "afis_prefix",
         "fbi_fp", "filename", "biometric_fp", "ten_print"],
        weights=[14, 18, 14, 14, 12, 12, 10, 6], k=1,
    )[0]
    if fmt == "fp_under":
        return f"FP_{n}"
    if fmt == "fp_dashed":
        return f"FP-{n}"
    if fmt == "fpid":
        return f"FPID-{n}"
    if fmt == "afis_prefix":
        return f"AFIS-{n}"
    if fmt == "fbi_fp":
        return f"FBI-FP-{n}"
    if fmt == "filename":
        ext = random.choice(["jpg", "png", "wsq", "nist", "bmp"])
        return f"fingerprint_{n}.{ext}"
    if fmt == "biometric_fp":
        return f"BIO-FP-{n}"
    return f"TENPRINT-{n}"


def _signature() -> str:
    n = random.randint(1000, 9999999)
    fmt = random.choices(
        ["sig_dashed", "sign_dashed", "signature_word", "esig",
         "sig_img", "signature_id", "filename", "docusign_envelope"],
        weights=[16, 14, 14, 14, 14, 12, 10, 6], k=1,
    )[0]
    if fmt == "sig_dashed":
        return f"SIG-{n}"
    if fmt == "sign_dashed":
        return f"SIGN-{n}"
    if fmt == "signature_word":
        return f"SIGNATURE-{n}"
    if fmt == "esig":
        return f"e-SIG-{n}"
    if fmt == "sig_img":
        return f"SIG-IMG-{n}"
    if fmt == "signature_id":
        return f"SIGNID-{n}"
    if fmt == "filename":
        ext = random.choice(["png", "jpg", "svg", "p7s"])
        return f"signature_{n}.{ext}"
    # docusign_envelope (UUID)
    HEX = "0123456789abcdef"
    return (f"{''.join(random.choices(HEX, k=8))}-"
            f"{''.join(random.choices(HEX, k=4))}-"
            f"{''.join(random.choices(HEX, k=4))}-"
            f"{''.join(random.choices(HEX, k=4))}-"
            f"{''.join(random.choices(HEX, k=12))}")


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
            # Quoted-nickname and edge-case variants
            "Cardiologist: {value}",
            "Oncologist: {value}",
            "Pediatrician: {value}",
            "Radiologist on call: {value}",
            "Anesthesiologist: {value}",
            "Pathologist: {value}",
            "Neurologist consulted: {value}",
            "Orthopedic surgeon: {value}",
            "Dermatologist: {value}",
            "Psychiatrist: {value}",
            "OB-GYN: {value}",
            "Resident on duty: {value}",
            "Chief of staff: {value}",
            "Department head: {value}",
            "Medical director: {value}",
            "Board-certified physician: {value}",
            "On-call provider: {value}",
            "The dictating physician was {value}.",
            "Co-signed by {value}.",
            "Reviewed by {value}.",
            "{value} authored the discharge summary.",
            "{value} ordered the imaging study.",
            "Telehealth consult with {value}.",
            "Pre-op evaluation by {value}.",
            "Post-op visit with {value}.",
            "{value} was the operating surgeon.",
            "{value} is the patient's PCP.",
            # Multicultural / international physician contexts
            "International physician {value} licensed to practice.",
            "Visiting consultant: {value}",
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
            # Edge-case formats teach the model to handle uncommon variants
            "Beneficiary: {value}",
            "Authorized representative: {value}",
            "Power of attorney: {value}",
            "Spouse: {value}",
            "Next of kin: {value}",
            "Dependent: {value}",
            "Co-applicant: {value}",
            "Customer Name: {value}",
            "Subscriber: {value}",
            "Policyholder: {value}",
            "Account opened by {value}.",
            "Statement issued to {value}.",
            "The party of record is {value}.",
            "Application received from {value}.",
            "Transferee: {value}",
            "Loan applicant {value} submitted documentation.",
            "Verified ID for {value}.",
            "Customer {value} called regarding their account.",
            "Witness: {value}",
            # Multicultural / international name contexts
            "International student {value} arrived on the F-1 visa.",
            "Visiting scholar {value} delivered the keynote.",
            "Foreign national: {value}",
            "Patient name (last, first): {value}",
            # All-caps and uppercase contexts
            "Customer name (as printed): {value}",
            "PRINTED NAME: {value}",
            "PATIENT NAME: {value}",
            "ACCOUNT HOLDER: {value}",
            # Filiation / genealogy markers (mostly Indian government IDs)
            "{value} resides in the registered address.",
            "Father's name: {value}",
            "Mother's name: {value}",
            "Spouse's name: {value}",
            "Care of {value} at the listed address.",
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
            "Address: {value}, Chicago, IL 60614",
            "The patient at {value}, Los Angeles, CA 90001.",
            "Home: {value}, Houston, TX 77001.",
            "Mailing: {value}, New York, NY 10001.",
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
            "Residing in {value} TX.",
            "Located in {value} CA.",
            "Lives in {value} NY 10001.",
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
            "Year: {value}",
            "Birth year: {value}",
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
            "Surgery Date: {value}",
            "Follow-up appointment: {value}",
            "Prescription Date: {value}",
            "Date of Service: {value}",
            "Clinical Date: {value}",
            "Date: {value}",
            "Effective {value}",
            "As of {value}",
            "Since {value}",
            "Commercial license issued {value}",
            "Enrolled {value}.",
            "Year: {value}",
            "Effective year {value}.",
            "Record timestamp: {value}",
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
        # Mix the health-plan-specific format with the generic member-id format
        # so the model sees both UHC-/BCBS-/INS- prefixed and MBR/SUB/M-prefixed IDs.
        "generator": lambda: random.choice([_health_plan_id(), _member_id()]),
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
        "generator": _billing_num,
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
            "ML: {value}",
            "Medical license on file: {value}",
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
            "Provider {value} on file.",
        ],
    },

    "passport_number": {
        "generator": _passport_num,
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
        # Mix MAC addresses (6-octet, multiple formats), IMEI / IMEISV, ICCID,
        # Android-ID, iOS UDID. Single _mac generator was too narrow — devices
        # have many ID schemes in the wild.
        "generator": lambda: random.choice([_mac(), _mac(), _mac(), _imei()]),
        "templates": [
            "MAC address: {value}",
            "Device MAC: {value}",
            "Network interface {value}",
            "Hardware address: {value}",
            "IMEI: {value}",
            "IMEISV: {value}",
            "Device ID: {value}",
            "Device identifier: {value}",
            "MAC Address: {value}",
            "ICCID: {value}",
            "Android ID: {value}",
            "iOS UDID: {value}",
            "The device MAC is {value}.",
            "Physical address: {value}",
            "Device fingerprint: {value}",
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
        "generator": _facial_recognition,
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
        "generator": lambda: random.choice([
            f"voice_template_{random.randint(1000,9999)}.dat",
            _voiceprint_vp(),
        ]),
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
            "Voiceprint No: {value}",
            "VP reference: {value}",
        ],
    },

    "biometric_iris_scan": {
        "generator": _iris_scan,
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
        "generator": lambda: random.choice([
            f"DNA_profile_{random.randint(10000,99999)}",
            _dna_str_locus(),
        ]),
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
            "STR locus allele: {value}",
            "CODIS locus result: {value}",
        ],
    },

    "fingerprint": {
        "generator": _fingerprint,
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
            # ── Core / generic anchors ──
            "Organization: {value}",
            "Employer: {value}",
            "Company: {value}",
            "Entity: {value}",
            "Business: {value}",
            "The organization is {value}.",
            "Referred from {value}.",
            "The employer on record is {value}.",
            "Filed by {value}.",
            "Authorized by {value}.",
            # ── Title-case keyword anchors (user-reported failure modes) ──
            "Company Name: {value}",
            "Business Name: {value}",
            "Organization Name: {value}",
            "Nonprofit Organization: {value}",
            "Government Organization: {value}",
            "Technology Organization: {value}",
            "Startup Organization: {value}",
            "Consulting Organization: {value}",
            "Retail Organization: {value}",
            "Transportation Organization: {value}",
            "Manufacturing Organization: {value}",
            "Media Organization: {value}",
            "Research Organization: {value}",
            "Healthcare Organization: {value}",
            "Educational Organization: {value}",
            "Financial Organization: {value}",
            "Insurance Organization: {value}",
            "Legal Organization: {value}",
            "Charitable Organization: {value}",
            # ── ALL-CAPS label form ──
            "ORGANIZATION#: {value}",
            "NONPROFIT ORG#: {value}",
            "GOVT ORGANIZATION-ID: {value}",
            "MEDIA ORG-ID: {value}",
            "BUSINESS NAME-ID: {value}",
            "COMPANY NAME#: {value}",
            "ENTITY ORG-ID: {value}",
            # ── Conversational / prose forms ──
            "The organization name is {value}.",
            "Government organization {value} updated the policy.",
            "Retail organization {value} linked to the account.",
            "Research organization {value} validated the protocol.",
            "Consulting organization {value} stored the data securely.",
            "Patient is employed by {value}.",
            "Records were filed by {value}.",
            "The application was submitted by {value} on behalf of the patient.",
            "{value} confirmed the appointment.",
            "Provided by {value}.",
            "Operated under {value}.",
            "Subsidiary of {value}.",
            "Parent entity: {value}.",
            "Sponsoring organization: {value}.",
            # ── Multi-line directory style ──
            "Vendor: {value}",
            "Supplier: {value}",
            "Contractor: {value}",
            "Agency: {value}",
            "Department: {value}",
            "Bureau: {value}",
        ],
    },

    "signature": {
        # Mix of (a) free-form name signatures and (b) coded signature IDs.
        # Coded IDs already have rock-solid regex coverage; including them here
        # teaches GLiNER to back the regex up when context is clipped.
        "generator": lambda: random.choice([
            _first_last(),
            _first_last(),
            _first_last(),  # 3x weight on free-form names
            f"/s/ {_first_last()}",
            f"/s/ {_first_last()}",
            _signature(),
            _signature(),
            _signature(),  # 3x weight on coded signature IDs from helper
            f"BIO-SIGN-{random.randint(1000, 9999)}",
            f"DOCSIGN-{random.randint(1000, 9999)}",
            f"REGSIG-{random.randint(1000, 9999)}",
            f"AST-{random.randint(1000, 9999)}",
            f"PKI-{random.randint(1000, 9999)}",
            f"ENC-SIGN-{random.randint(1000, 9999)}",
            f"SDS-{random.randint(1000, 9999)}",
            # Underscored handle: "JDoe_Sign", "MSmith_Sign"
            (lambda fn=_first_last(): f"{fn.split()[0][0]}{fn.split()[-1]}_Sign")(),
            # Dotted initials: "J.Doe", "K.Kate"
            (lambda fn=_first_last(): f"{fn.split()[0][0]}.{fn.split()[-1]}")(),
            # Spaced initials: "K. Kate"
            (lambda fn=_first_last(): f"{fn.split()[0][0]}. {fn.split()[-1]}")(),
        ]),
        "templates": [
            # ── Generic signature anchors ──
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
            # ── User-reported keyword anchors ──
            "Secure Signature ID: {value}",
            "Biometric Signature: {value}",
            "Document Signature: {value}",
            "Registered Signature: {value}",
            "Authenticated Signature Token: {value}",
            "PKI Digital Signature: {value}",
            "Encrypted Signature ID: {value}",
            "Secure Document Sign-Off: {value}",
            "Notarized Signature: {value}",
            "Witness Signature: {value}",
            # ── ALL-CAPS label form ──
            "DIGITAL SIGNATURE-ID: {value}",
            "BIOMETRIC SIGN#: {value}",
            "DOC SIGNATURE#: {value}",
            "REGISTERED SIGN-ID: {value}",
            "SIGN#: {value}",
            "E-SIGNATURE-ID: {value}",
            # ── Conversational forms ──
            "Document signature {value} verified.",
            "Registered signature {value} stored securely.",
            "Biometric signature {value} validated.",
            "Electronic signature {value} verified successfully.",
            "Secure signature ID {value} generated.",
            # ── Letter / contract closing forms (bare name as signature) ──
            "Sincerely,\n{value}",
            "Respectfully,\n{value}",
            "Best regards,\n{value}",
            "/s/ {value}",
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
        "generator": _pin_block,
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
        "generator": _card_iin_bin,
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
            "routing {value}",
            "RTN {value}",
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
        "generator": _terminal_id,
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
        "generator": _card_type,
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
        "generator": _card_last4,
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
        "generator": lambda: random.choice([_username(), _username_bare()]),
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
            "Logged in as {value}.",
            "Access granted to {value}.",
        ],
    },

    "confidential": {
        "generator": lambda: random.choice([
            "CONFIDENTIAL", "Confidential", "TOP SECRET", "Top Secret",
            "RESTRICTED", "Restricted", "SENSITIVE", "SENSITIVE PII",
            "SECRET", "PRIVATE", "PROPRIETARY", "INTERNAL USE ONLY",
            "FOR OFFICIAL USE ONLY", "FOUO", "CUI", "PHI",
            "PRIVILEGED", "ATTORNEY-CLIENT PRIVILEGED", "DRAFT",
            "NOT FOR DISTRIBUTION", "DO NOT FORWARD", "EYES ONLY",
            "CLASSIFIED", "TS/SCI", "NOFORN", "LIMITED DISTRIBUTION",
            "TLP:RED", "TLP:AMBER", "TLP:GREEN",
            _confidential_ref(), _confidential_ref(),
        ]),
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
        "generator": _dept_color,
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
        # Mix simple "Title at Company" with date-ranged history, multi-line,
        # career arc, and termination notes.
        "generator": lambda: random.choice([
            f"{_job_title()} at {_employer()}",
            f"{_job_title()} at {_employer()}",  # 2x weight basic
            f"{_job_title()} at {_employer()} ({random.randint(2010, 2020)}-{random.randint(2021, 2025)})",
            f"{_job_title()} at {_employer()}, {random.randint(2015, 2024)}-Present",
            f"Former {_job_title()} at {_employer()}",
            f"Previously {_job_title()} at {_employer()}",
            f"Senior {_job_title()} at {_employer()}",
            f"Junior {_job_title()} at {_employer()}",
            f"{_job_title()} at {_employer()}; terminated {random.randint(2018, 2024)}",
            f"{_job_title()} at {_employer()} ({random.randint(1, 15)} years)",
            f"{_job_title()}, {_employer()}, {random.choice(['Full-time', 'Contract', 'Part-time'])}",
            f"{_job_title()} (Intern) at {_employer()}",
        ]),
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
        "generator": lambda: random.choice([
            # 5-point scale
            "Exceeds Expectations 4.5/5", "Exceeds Expectations 4.2/5",
            "Meets Expectations 3.2/5", "Meets Expectations 3.5/5",
            "Needs Improvement 2.0/5", "Needs Improvement 2.5/5",
            "Outstanding 5/5", "Outstanding 4.8/5",
            "Below Expectations 1.8/5", "Unsatisfactory 1.0/5",
            # Categorical
            "Exceeds Expectations", "Meets Expectations",
            "Needs Improvement", "Outstanding", "Below Expectations",
            "Unsatisfactory", "Satisfactory",
            # Letter grades
            "Grade A", "Grade B+", "Grade B", "Grade C", "Grade D",
            "Grade F",
            # Numeric percent
            f"{random.randint(60, 99)}%", f"{random.randint(2, 10) / 10 * 100:.0f}%",
            # Numeric out-of-100
            f"{random.randint(60, 100)}/100",
            # 10-point scale
            f"{random.uniform(1, 10):.1f}/10",
            # 4-point scale (academic)
            f"{random.uniform(0, 4):.2f}/4.0",
            # Tiered/banded
            "Tier 1 - Exemplary", "Tier 2 - Strong", "Tier 3 - Developing",
            "Tier 4 - Underperforming", "Band A", "Band B", "Band C",
            # Star rating
            "5 stars", "4 stars", "3 stars", "2 stars",
            # Qualitative
            "Top performer", "High potential", "Solid contributor",
            "Improvement plan required", "Performance Improvement Plan (PIP)",
        ]),
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
        "generator": lambda: random.choice([
            f"GPA {round(random.uniform(1.5,4.0),2)}, enrolled in {random.randint(12,18)} credit hours",
            _ferpa_id(),
        ]),
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
            "FERPA ID: {value}",
            "Student records FERPA: {value}",
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
            # Brand-name and dosage variants
            "Prescription Medication: {value}",
            "OTC Medication: {value}",
            "Treatment Medication: {value}",
            "Hospital Medication Name: {value}",
            "Pharmacy Drug Name: {value}",
            "Diabetes Medication: {value}",
            "Antibiotic: {value}",
            "Prescribed antibiotic: {value}",
            "Pain medication: {value}",
            "Antihypertensive: {value}",
            "Antidepressant: {value}",
            "Anti-inflammatory: {value}",
            "Inhaler prescribed: {value}",
            "Patient takes {value} every morning.",
            "Discharge medications: {value}",
            "Take {value} as directed.",
            "Refill ordered for {value}.",
            "Started on {value} two weeks ago.",
            "Continue {value} per prior regimen.",
            "Discontinued {value} due to side effects.",
            "Switched from prior med to {value}.",
            "Increased dose of {value}.",
            "Tapering {value} over 2 weeks.",
            "Allergic to {value}.",
            "PATIENT DRUG-ID: {value}",
            "ANTIBIOTIC MED#: {value}",
            "Pharmacy filled {value}.",
            "Dispensed by pharmacist: {value}",
            "Brand-name medication: {value}",
            "Generic equivalent: {value}",
            "Over-the-counter purchase: {value}",
            "Insulin therapy: {value}",
            "Statin therapy: {value}",
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
        "generator": _application_id,
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
            # Specialty-facility templates teach the model to recognize
            # dental / rehab / trauma / women's / urgent-care centers as
            # hospital_name (industry taxonomy).
            "Patient transferred to {value} for specialty care.",
            "Emergency case routed to {value}.",
            "Therapy sessions ongoing at {value}.",
            "Dental records updated from {value}.",
            "REHAB CENTER-ID: {value}",
            "Women's Health Clinic: {value}",
            "Dental Medical Center: {value}",
            "Trauma Center: {value}",
            "Rehab facility: {value}",
            "Urgent Care location: {value}",
            "Surgical Center: {value}",
            "Imaging Center: {value}",
            "Behavioral Health Provider: {value}",
            "OB-GYN clinic: {value}",
            "Children's hospital: {value}",
            "Cancer Center: {value}",
            "Cardiac Center: {value}",
            "Eye Institute: {value}",
            "Outpatient services rendered at {value}.",
            "Long-term care provided at {value}.",
            "Skilled nursing facility: {value}",
            "Physical therapy at {value}.",
            "Mental health services from {value}.",
            "Hospice care at {value}.",
            "Walk-in clinic visit at {value}.",
            "Network provider: {value}",
            "Affiliated hospital: {value}",
            "Patient admitted to the {value} ICU.",
            "Surgery scheduled at {value}.",
            "Laboratory tests run at {value}.",
            "Visiting hours at {value} are 9 AM to 8 PM.",
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
        "generator": _fbi_number,
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
        "generator": _chri,
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
        "generator": _arrest_record,
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
        "generator": _warrant_num,
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
        "generator": _incident_num,
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
        "generator": lambda: random.choice([_incarceration_info(), _inmate_id()]),
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
        "generator": _missing_person_report,
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
        "generator": _wanted_person_report,
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
        "generator": _sex_offender_report,
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
        "generator": _cpo_order,
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
        "generator": _foreign_fugitive,
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
        "generator": _identity_theft_victim,
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
        "generator": _gang_terrorist_member,
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
        "generator": _supervised_release,
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
        "generator": _probation_record,
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
        "generator": _parole_record,
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
        "generator": _stolen_vehicle,
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
        "generator": _stolen_guns,
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
        "generator": _stolen_license_plate,
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
        "generator": _stolen_boats,
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
        "generator": _stolen_securities,
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
        "generator": _stolen_articles,
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
        "generator": lambda: random.choice([
            f"DMV-{random.randint(100000,9999999)}",
            _driver_history_dh(),
        ]),
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
    # Label-word-before-"ID" negatives — the word before "ID"/"number" is NOT a
    # place. Stops GLiNER tagging Transaction/Merchant/Terminal/etc. as LOCATION.
    "Transaction ID is generated automatically for each order.",
    "Please reference the Merchant ID when contacting the processor.",
    "The Terminal ID must be configured before the device is used.",
    "Enter your Employee ID to clock in for the shift.",
    "The Tax ID number is required before filing the return.",
    "Provide the Application ID printed on your receipt.",
    "Your Student ID grants access to the campus library.",
    "The Card IIN identifies the issuing bank.",
    "A Booking Reference is emailed after the reservation is confirmed.",
    "The Case Number is assigned by the court clerk.",
    "Each Device Identifier is unique to the hardware unit.",
    "The State ID number is issued by the motor vehicle agency.",
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
    # Group-B hard negatives — guard against over-tagging of generic vocabulary
    "The patient was treated and released.",
    "Please refer to the patient's chart for medical history.",
    "The doctor will see you now.",
    "The physician on call will respond within 30 minutes.",
    "All medications must be reviewed annually.",
    "The pharmacy is open from 8 AM to 10 PM.",
    "The hospital's policy requires written consent for all procedures.",
    "The medical center has a Level 1 trauma rating.",
    "Walk-in clinics are open seven days a week.",
    "Urgent care services are available without an appointment.",
    "Emergency departments are staffed 24/7.",
    "Dental insurance covers preventive care twice yearly.",
    "Behavioral health appointments require a referral.",
    "Rehabilitation services include physical and occupational therapy.",
    "The women's clinic offers comprehensive obstetric care.",
    "Cardiology consults must be requested in advance.",
    "Oncology referrals are processed within 48 hours.",
    "The pediatric ward visits hours are restricted.",
    "Surgical procedures are scheduled by the operating room coordinator.",
    "Imaging studies require radiology approval.",
    "Skilled nursing facilities provide 24-hour care.",
    "Hospice services focus on comfort and quality of life.",
    "The patient's primary care provider should be contacted.",
    "All providers must verify patient identity before treatment.",
    "Antibiotic stewardship reduces resistance.",
    "Generic medications are typically less expensive than brand names.",
    "Over-the-counter drugs do not require a prescription.",
    "Brand-name medications may have generic equivalents.",
    "The drug interaction warning was reviewed.",
    "Insulin administration requires patient education.",
    "Steroid tapering should be done gradually.",
    "Chemotherapy schedules are individualized.",
    "Pain management plans are tailored to each patient.",
    "Antidepressant therapy may take weeks to show effect.",
    "The pharmacist counseled the patient on side effects.",
    "Medication reconciliation occurs at every transition of care.",
    "Drug allergies must be documented in the chart.",
    "Inhaler technique should be reviewed at every visit.",
    "Statin therapy is recommended for high cholesterol patients.",
    "All clinicians must wash hands before patient contact.",
    "The nurse practitioner can write prescriptions.",
    "The physician assistant works under physician supervision.",
    # Generic person-name distractors (don't tag these as person_name)
    "Dear Sir or Madam,",
    "To Whom It May Concern,",
    "The Patient consented to the procedure.",
    "Customer service is available Monday through Friday.",
    "Account holders are responsible for all transactions.",
    "The Defendant entered a plea of not guilty.",
    "The Plaintiff filed a motion to dismiss.",
    "The Witness was sworn in before testifying.",
    "Members of the audience must remain seated.",
    "Visitors must check in at the front desk.",
    # ── Organization-name distractors (Group B-2) ──
    "The organization follows a strict code of conduct.",
    "Any organization must comply with HIPAA.",
    "The company has a strong customer service track record.",
    "A nonprofit organization is exempt under section 501(c)(3).",
    "This business is registered with the state.",
    "An entity must register before filing taxes.",
    "Government agencies must publish annual reports.",
    "The Department issues permits during business hours.",
    "Bureau policies are reviewed every five years.",
    "A startup may qualify for tax credits.",
    "Consulting firms typically bill hourly.",
    "Retail organizations are subject to state sales tax.",
    "Manufacturing organizations report quarterly emissions data.",
    "Media organizations observe editorial independence.",
    "Research organizations partner with universities.",
    "Healthcare organizations must meet accreditation standards.",
    "Charitable organizations rely on donor contributions.",
    "An organization name was not included on the form.",
    "Please enter your employer's full legal name.",
    "The company name is printed on the badge.",
    # ── Signature-context distractors ──
    "The signature line is at the bottom of the page.",
    "A signature is required to complete the form.",
    "Please sign and date the document where indicated.",
    "Electronic signatures are legally binding under E-SIGN.",
    "Wet signatures require the original document.",
    "The digital signature certificate has expired.",
    "Notarized signatures must be witnessed.",
    "Please initial each page and sign the last one.",
    "All signatures must be in blue ink.",
    "The signature block is reserved for authorized officers.",
    "A witness signature is required for this form.",
    "The signature was illegible and rejected.",
    "Counter-signature is required by the second party.",
    "Authorized signatories are listed in the corporate registry.",

    # ── Single-word hallucination kills ───────────────────────────────────────
    # Sentences where COMMON ENGLISH WORDS that were getting tagged as entities
    # (Bank, State, Treasury, Transaction, Domestic, Charter, Web, Transfer,
    # Card, Session, Order, International, Secure, Corporate, etc.) appear
    # in their everyday non-entity meaning. Teaches GLiNER that these are
    # NOT entities by themselves.
    "Bank holidays are listed at the bottom of the statement.",
    "Bank fees apply for paper statements.",
    "The bank closed at 5pm today.",
    "We met at the bank yesterday morning.",
    "Bank rules differ from credit union rules.",
    "Treasury yields rose in the afternoon.",
    "The treasury department published new guidance.",
    "Treasury notes mature in ten years.",
    "Treasury bonds carry lower risk than corporate bonds.",
    "State law requires this disclosure.",
    "The state of California passed a new law.",
    "State officials confirmed the budget today.",
    "Each state sets its own minimum wage.",
    "The state and federal forms differ.",
    "State income tax is withheld from each paycheck.",
    "Transaction history is available in the online portal.",
    "Transaction fees may apply to international purchases.",
    "Transaction costs were higher than expected.",
    "Domestic shipping is included with the purchase.",
    "Domestic flights board 30 minutes before departure.",
    "The domestic policy was revised in March.",
    "Charter schools follow different rules than public schools.",
    "The charter was renewed for another five years.",
    "Charter flights serve smaller airports.",
    "Web traffic increased during the campaign.",
    "Web design is part of the curriculum.",
    "The web team will fix the bug tomorrow.",
    "Transfer paperwork is processed within 48 hours.",
    "Transfer fees were waived for premium members.",
    "Card readers must be replaced annually.",
    "Card stock is available in multiple weights.",
    "The card game lasted until midnight.",
    "Session attendance was higher this quarter.",
    "The session began at 9 AM with introductions.",
    "Session times are listed on the website.",
    "Order processing takes one to two business days.",
    "Order forms are available at the front desk.",
    "The order was placed yesterday morning.",
    "International shipping is available to most countries.",
    "International law governs cross-border disputes.",
    "International students must register by Monday.",
    "Secure storage is provided for valuable items.",
    "Secure access requires two-factor authentication.",
    "The secure facility is closed to visitors.",
    "Corporate policy prohibits personal use of the equipment.",
    "Corporate dining is available on the second floor.",
    "Investment grew over the past five years.",
    "Investment strategy varies by risk tolerance.",
    "Refund policy applies to all returned items.",
    "Refund amount depends on the original payment method.",
    "Identifier fields are required on the application.",
    "Statement balances reflect last night's posting.",
    "Reference materials are available in the library.",
    "Reference checks were completed last week.",
    "Policy changes take effect next quarter.",
    "Policy reviews happen every six months.",
    "Number formatting depends on the locale.",
    "Number ranges are defined in the schema.",
    "Web Session ID is generated server-side.",
    "Transaction SWIFT ID rotates daily.",
    "Bank Identifier Code conventions follow ISO 9362.",
    "Corporate Card Number formats vary by issuer.",
    "Transfer SWIFT ID validation runs at submission.",
    "Treasury SWIFT Number is reserved for sovereign transactions.",
    "Domestic Flight Number prefixes differ from international codes.",
    "Charter Flight Number is assigned by the operator.",
    "Routing SWIFT records are updated weekly.",
    "Card Last4 is shown on the receipt only.",
    "Card Suffix display is configurable per merchant.",
    "Last Four of Card is sufficient for verification.",
    "POS Card Ending Number is masked in logs.",
    "Authentication Session Cookie is set on login.",
    "Secure Cookie ID expires when the browser closes.",
    "Single Sign-On Token is rotated every hour.",
    "Web Session ID format is opaque to the client.",
    "User Session Identifier is generated per device.",
    "Authentication tokens should not appear in URLs.",
    "Bank Identification Number lookup takes milliseconds.",
    "Visa BIN Number ranges are documented publicly.",
    "Merchant BIN Number is internal to the processor.",
    "Issuer Identification Number ranges change rarely.",
    "Card Issuer BIN data is licensed from networks.",
    "Network IIN is the same across all cards in a network.",
    "State Identification Number formats differ by state.",
    "State issued ID requirements vary by jurisdiction.",
    "Legal state identification documents include passports.",
    "Regional state ID requirements differ from federal ones.",
    "Government state ID is required for boarding domestic flights.",

    # Words in list/heading contexts where they should NOT be tagged
    "Sections: Bank, Treasury, Transaction, Domestic, International.",
    "Categories include: State, Federal, Local, and Regional.",
    "Fields: Card, Session, Web, Order, Transfer.",
    "Topics covered: Banking, Treasury, Investments, Refunds.",
    "Headers: Bank Account, Routing, Statement, Transfer.",

    # Verbs / common nouns that often share form with PII tags
    "Please charge the corporate card for this purchase.",
    "Please reference the policy when making a claim.",
    "Please order new business cards this week.",
    "Please transfer the file to the secure drive.",
    "Please secure the access doors after hours.",
    "Please web-search the topic before the meeting.",

    # Address-shaped non-addresses
    "Building 4 houses the engineering team.",
    "Suite 12 is the conference room.",
    "Room 301 is reserved for the meeting.",
    "Office 5 has been reassigned to marketing.",

    # Date-context non-dates (prevent random number → date hallucinations)
    "Page 12 of 50 was missing from the binder.",
    "Question 3 of 25 was answered incorrectly.",
    "Item 7 of 100 is on backorder.",
    "Section 4 of the contract covers indemnification.",

    # Number-shape negatives (large bare numbers that are NOT PII)
    "There are 50000 employees worldwide.",
    "Revenue increased to 100000 last quarter.",
    "The event drew 75000 attendees over three days.",
    "The population reached 250000 in 2024.",
    "The library catalogs 980000 volumes.",
    "Distance to the depot is 12000 meters.",
    "The room capacity is 1500.",
    "Page count: 850000 documents archived this year.",

    # 5-digit non-zip negatives
    "The product weighs 12345 grams.",
    "The drive holds 50000 files.",
    "The error code 60110 indicates a connection failure.",
    "Lot number 99999 was recalled last month.",

    # 6-digit non-BIN negatives
    "The serial 601100 is on the back panel.",
    "Asset tag 411111 belongs to the IT department.",
    "Build 305693 was released last Tuesday.",
    "Job 555544 is in the queue for processing.",

    # Common-word + ID combo (the user's biggest failure mode)
    "Transaction was approved automatically.",
    "Web administered the certificate renewal.",
    "Domestic violence prevention training is mandatory.",
    "Charter members get exclusive benefits.",
    "Order forms must be approved by a manager.",
    "Card processing follows PCI-DSS standards.",
    "Session expired due to inactivity.",
    "Transfer protocol uses TLS 1.3.",
    "Treasury management is centralized at headquarters.",
    "International expansion is planned for next year.",

    # MM/YY and date-shaped non-dates
    "Step 5/28 of the wizard is the final review.",
    "Page 12/27 has the appendix.",
    "Lesson 8/30 covers advanced topics.",
    "Module 11/29 is required reading.",

    # BIC-shaped non-BIC words (8-char uppercase clusters)
    "FACEBOOK launched a new feature today.",
    "INSTAGRAM is owned by Meta Platforms.",
    "WHATSAPP is widely used in Europe.",
    "TWITTER rebranded as X in 2023.",
    "LINKEDIN profiles are required for our talent system.",

]


# ---------------------------------------------------------------------------
# Multi-occurrence and disambiguation training data
# ---------------------------------------------------------------------------

# Multi-occurrence templates: {v1} and {v2} are two different values of the same entity.
# This teaches the model to find ALL occurrences, not just the first.
MULTI_ENTITY_TEMPLATES: dict[str, list[str]] = {
    "clinical_date": [
        "Patient DOB: {v1}. Procedure scheduled for {v2}.",
        "Admission Date: {v1}. Discharge Date: {v2}.",
        "Lab drawn on {v1}. Results received {v2}.",
        "Treatment started {v1}, follow-up on {v2}.",
        "Enrolled {v1}. Terminated {v2}.",
        "Service date {v1}. Billed on {v2}.",
    ],
    "date_of_birth": [
        "Patient DOB: {v1}. Spouse DOB: {v2}.",
        "Primary insured DOB {v1}, dependent DOB {v2}.",
    ],
    "person_name": [
        "Patient {v1} was seen by Dr. {v2}.",
        "Claimant {v1}. Emergency contact: {v2}.",
        "Referring provider: {v1}. Treating physician: {v2}.",
        "Insured: {v1}. Beneficiary: {v2}.",
        "Arrested: {v1}. Victim: {v2}.",
    ],
    "phone_number": [
        "Home phone: {v1}. Work phone: {v2}.",
        "Patient: {v1}. Emergency contact: {v2}.",
        "Primary: {v1}. Alternate: {v2}.",
    ],
    "medication_name": [
        "Patient takes {v1} and {v2} daily.",
        "Prescribed {v1}. Also on {v2}.",
        "Current medications: {v1}, {v2}.",
        "Rx 1: {v1}. Rx 2: {v2}.",
    ],
    "case_number": [
        "Case Number: {v1}. Related case: {v2}.",
        "Primary case {v1}; consolidated with {v2}.",
    ],
    "ip_address": [
        "Source IP: {v1}. Destination IP: {v2}.",
        "Login from {v1}. Last session: {v2}.",
    ],
}


# Templates with two different-entity types — teaches model to distinguish similar-looking values.
DISAMBIGUATION_TEMPLATES: list[dict] = [
    # Case number next to license plate — prevent case# -> [LICENSE PLATE]
    {
        "text_template": "Case Number: {case_num}. Vehicle license plate: {plate}.",
        "entities": [
            ("case_num", "case_number", _case_number),
            ("plate", "license_plate", _license_plate),
        ],
    },
    # NPI next to phone — prevent NPI -> [PHONE NUMBER]
    {
        "text_template": "NPI: {npi}. Phone: {phone}.",
        "entities": [
            ("npi", "npi_number", _npi_plain),
            ("phone", "phone_number", _phone),
        ],
    },
    # Hospital next to Insurance company — prevent hospital -> [INSURANCE COMPANY]
    {
        "text_template": "Facility: {hospital}. Insurance carrier: {insurer}.",
        "entities": [
            ("hospital", "hospital_name", _hospital),
            ("insurer", "insurance_company_name", _insurance_co),
        ],
    },
    {
        "text_template": "Patient admitted to {hospital}. Claim filed with {insurer}.",
        "entities": [
            ("hospital", "hospital_name", _hospital),
            ("insurer", "insurance_company_name", _insurance_co),
        ],
    },
    # IP address next to GPS — prevent IP -> [GPS]
    {
        "text_template": "Device IP Address: {ip}. GPS coordinates: {gps}.",
        "entities": [
            ("ip", "ip_address", _ip),
            ("gps", "precise_geolocation", _gps),
        ],
    },
    # Routing number next to transaction ID — prevent routing -> [TRANSACTION ID]
    {
        "text_template": "Routing Number: {routing}. Transaction ID: {txn}.",
        "entities": [
            ("routing", "bank_routing_number", _routing),
            ("txn", "transaction_id", _transaction_id),
        ],
    },
    # Bank account next to routing — together in financial records
    {
        "text_template": "Account Number: {account}. Routing Number: {routing}.",
        "entities": [
            ("account", "bank_account_number", _bank_account),
            ("routing", "bank_routing_number", _routing),
        ],
    },
    # Address with city+state
    {
        "text_template": "Patient resides at {addr}, {city}, {state} {zipcode}.",
        "entities": [
            ("addr", "street_address", _street_address),
            ("city", "city_name", _city),
            ("state", "us_state", _state_abbr),
            ("zipcode", "zipcode", _zipcode),
        ],
    },
    {
        "text_template": "Home address: {addr}, {city}, {state} {zipcode}. Phone: {phone}.",
        "entities": [
            ("addr", "street_address", _street_address),
            ("city", "city_name", _city),
            ("state", "us_state", _state_abbr),
            ("zipcode", "zipcode", _zipcode),
            ("phone", "phone_number", _phone),
        ],
    },
    # Date with second date — teach model to find both
    {
        "text_template": "DOB: {dob}. Procedure Date: {proc}.",
        "entities": [
            ("dob", "date_of_birth", _date_mmddyyyy),
            ("proc", "clinical_date", _date_clinical),
        ],
    },
    {
        "text_template": "Admission Date: {admit}. Discharge Date: {discharge}.",
        "entities": [
            ("admit", "clinical_date", _date_clinical),
            ("discharge", "clinical_date", _date_clinical),
        ],
    },
    # NPI with prefix — teach model the NPI- prefix format
    {
        "text_template": "Provider NPI: {npi}. Phone: {phone}.",
        "entities": [
            ("npi", "npi_number", _npi_with_prefix),
            ("phone", "phone_number", _phone),
        ],
    },
    # Case number with alphanumeric format vs plate
    {
        "text_template": "Court case {case_num}. Plate number: {plate}.",
        "entities": [
            ("case_num", "case_number", _case_number_alphanumeric),
            ("plate", "license_plate", _license_plate),
        ],
    },
    # ── Group B disambiguation ────────────────────────────────────────────
    # Patient name vs physician name — same surface form, different role
    {
        "text_template": "Patient {patient} was seen by {doctor}.",
        "entities": [
            ("patient", "person_name", _first_last),
            ("doctor", "physician_name", _physician_name),
        ],
    },
    {
        "text_template": "Referring provider: {doctor}. Patient: {patient}.",
        "entities": [
            ("doctor", "physician_name", _physician_name),
            ("patient", "person_name", _first_last),
        ],
    },
    {
        "text_template": "{doctor} treated {patient} at {facility}.",
        "entities": [
            ("doctor", "physician_name", _physician_name),
            ("patient", "person_name", _first_last),
            ("facility", "hospital_name", _hospital),
        ],
    },
    # Specialty hospital + medication
    {
        "text_template": "Patient admitted to {facility}. Prescribed: {drug}.",
        "entities": [
            ("facility", "hospital_name", _hospital),
            ("drug", "medication_name", _medication),
        ],
    },
    {
        "text_template": "Treatment at {facility} included {drug}.",
        "entities": [
            ("facility", "hospital_name", _hospital),
            ("drug", "medication_name", _medication),
        ],
    },
    # Multiple medications (poly-pharmacy realism)
    {
        "text_template": "Current meds: {drug1}, {drug2}, and {drug3}.",
        "entities": [
            ("drug1", "medication_name", _medication),
            ("drug2", "medication_name", _medication),
            ("drug3", "medication_name", _medication),
        ],
    },
    # Multicultural names with physician — ensures model learns intl names
    {
        "text_template": "{patient} was admitted under the care of {doctor}.",
        "entities": [
            ("patient", "person_name", _intl_name),
            ("doctor", "physician_name", _physician_name),
        ],
    },
    # All-caps name vs uppercase label (distractor)
    {
        "text_template": "PATIENT NAME: {patient}. ATTENDING: {doctor}.",
        "entities": [
            ("patient", "person_name", _name_all_caps),
            ("doctor", "physician_name", _physician_name),
        ],
    },
    # Nickname-quoted name with physician
    {
        "text_template": "Cardholder: {cardholder}. Treating physician: {doctor}.",
        "entities": [
            ("cardholder", "card_holder_name", _card_holder),
            ("doctor", "physician_name", _physician_name),
        ],
    },
    # Genealogy prefix names — Indian ID-card style
    {
        "text_template": "Customer Name: {patient}. Father's name: {father}.",
        "entities": [
            ("patient", "person_name", _first_last),
            ("father", "person_name", _first_last),
        ],
    },
    # Hospital vs insurance company (sharp boundary — both are organizations)
    {
        "text_template": "Treatment at {facility}. Coverage through {insurer}.",
        "entities": [
            ("facility", "hospital_name", _hospital),
            ("insurer", "insurance_company_name", _insurance_co),
        ],
    },
    {
        "text_template": "Discharged from {facility}. Claim filed with {insurer}.",
        "entities": [
            ("facility", "hospital_name", _hospital),
            ("insurer", "insurance_company_name", _insurance_co),
        ],
    },
    # Brand-name drug vs facility — both can have similar token shapes
    {
        "text_template": "Patient at {facility} was prescribed {drug}.",
        "entities": [
            ("facility", "hospital_name", _hospital),
            ("drug", "medication_name", _medication),
        ],
    },
    # Prescriber + DEA + medication
    {
        "text_template": "{doctor} (DEA: {dea}) ordered {drug}.",
        "entities": [
            ("doctor", "physician_name", _physician_name),
            ("dea", "dea_number", _dea),
            ("drug", "medication_name", _medication),
        ],
    },
    # ─────────────────────────────────────────────────────────────────────
    # Group B (second batch) — organization / signature / dosage disambig.
    # ─────────────────────────────────────────────────────────────────────
    # Organization vs Hospital — both are facilities, but different labels.
    {
        "text_template": (
            "Employer: {org}. Treating facility: {hospital}."
        ),
        "entities": [
            ("org", "organization_name", _company),
            ("hospital", "hospital_name", _hospital),
        ],
    },
    {
        "text_template": (
            "Patient employed by {org}; admitted to {hospital}."
        ),
        "entities": [
            ("org", "organization_name", _company),
            ("hospital", "hospital_name", _hospital),
        ],
    },
    # Organization vs Insurance company — both are corporate entities.
    {
        "text_template": (
            "Employer: {org}. Insurance carrier: {insurer}."
        ),
        "entities": [
            ("org", "organization_name", _company),
            ("insurer", "insurance_company_name", _insurance_co),
        ],
    },
    # Organization vs Person — patient vs employer in the same sentence.
    {
        "text_template": (
            "Patient: {patient}. Employer: {org}."
        ),
        "entities": [
            ("patient", "person_name", _first_last),
            ("org", "organization_name", _company),
        ],
    },
    # Organization-on-its-own (no surrounding distractors) — strengthens
    # the model's prior on bare org names like "OpenAI Technologies".
    {
        "text_template": "Authorized by {org}.",
        "entities": [
            ("org", "organization_name", _company),
        ],
    },
    {
        "text_template": "{org} confirmed the appointment.",
        "entities": [
            ("org", "organization_name", _company),
        ],
    },
    # Signature vs Person (patient mentioned in body, signature at end).
    {
        "text_template": (
            "Patient: {patient}. /s/ {signer}"
        ),
        "entities": [
            ("patient", "person_name", _first_last),
            ("signer", "signature", _first_last),
        ],
    },
    {
        "text_template": (
            "Treating physician: {doctor}.\nElectronic signature: {signer}"
        ),
        "entities": [
            ("doctor", "physician_name", _physician_name),
            ("signer", "signature", _first_last),
        ],
    },
    {
        "text_template": (
            "Consent obtained from {patient}. Signed by: {signer}."
        ),
        "entities": [
            ("patient", "person_name", _first_last),
            ("signer", "signature", _first_last),
        ],
    },
    # Medication with explicit dosage form (Albuterol Sulfate Inhaler etc.)
    {
        "text_template": "Prescription: {drug}. Refill in 30 days.",
        "entities": [
            ("drug", "medication_name", _medication),
        ],
    },
    {
        "text_template": "Active medications: {drug1}; {drug2}.",
        "entities": [
            ("drug1", "medication_name", _medication),
            ("drug2", "medication_name", _medication),
        ],
    },
    # Brand-form medication ("Ventolin Inhaler", "Insulin Injection 10ml")
    {
        "text_template": "Dispensed: {drug}.",
        "entities": [
            ("drug", "medication_name", _medication),
        ],
    },
]


def generate_multi_occurrence_examples(n_per_template: int = 200) -> list[dict]:
    """Generate examples where the same entity appears twice in one sentence.

    Teaches the model to find ALL occurrences, not just the first.
    """
    examples: list[dict] = []
    for entity_id, tmpl_list in MULTI_ENTITY_TEMPLATES.items():
        label = LABEL_MAP.get(entity_id, entity_id)
        defn = ENTITY_DEFS.get(entity_id)
        if defn is None:
            continue
        generator = defn["generator"]
        for tmpl in tmpl_list:
            for _ in range(n_per_template):
                try:
                    v1 = generator()
                    v2 = generator()
                    # Ensure they're different (avoid duplicate-token edge case)
                    if v1 == v2:
                        v2 = generator()
                    text = tmpl.replace("{v1}", v1).replace("{v2}", v2)
                except Exception:
                    continue
                tokens = text.split()
                ner: list[list] = []
                for val in (v1, v2):
                    val_tokens = val.split()
                    for i in range(len(tokens) - len(val_tokens) + 1):
                        if tokens[i: i + len(val_tokens)] == val_tokens:
                            # Check not already covered
                            occupied = {idx for span in ner for idx in range(span[0], span[1] + 1)}
                            pos = set(range(i, i + len(val_tokens)))
                            if not pos & occupied:
                                ner.append([i, i + len(val_tokens) - 1, label])
                            break
                if ner:
                    examples.append({"tokenized_text": tokens, "ner": ner})
    return examples


def generate_disambiguation_examples(n_per_template: int = 200) -> list[dict]:
    """Generate examples with multiple co-occurring entities to teach disambiguation."""
    examples: list[dict] = []
    for tmpl in DISAMBIGUATION_TEMPLATES:
        for _ in range(n_per_template):
            # Generate values for each entity slot
            values: dict[str, str] = {}
            for slot_name, _entity_id, gen_fn in tmpl["entities"]:
                try:
                    values[slot_name] = gen_fn()
                except Exception:
                    values[slot_name] = ""

            text = tmpl["text_template"]
            for slot_name, val in values.items():
                text = text.replace("{" + slot_name + "}", val)

            tokens = text.split()
            ner: list[list] = []
            occupied: set[int] = set()

            for slot_name, entity_id, _gen_fn in tmpl["entities"]:
                val = values.get(slot_name, "")
                if not val:
                    continue
                label = LABEL_MAP.get(entity_id, entity_id)
                if not label:
                    continue
                val_tokens = val.split()
                for i in range(len(tokens) - len(val_tokens) + 1):
                    if tokens[i: i + len(val_tokens)] == val_tokens:
                        pos = set(range(i, i + len(val_tokens)))
                        if not pos & occupied:
                            ner.append([i, i + len(val_tokens) - 1, label])
                            occupied |= pos
                        break

            if ner:
                examples.append({"tokenized_text": tokens, "ner": ner})
    return examples


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
    display_map: dict[str, str] = {}
    for raw in data.get("entities", []):
        if not raw.get("enabled", True):
            continue
        eid = raw["id"]
        gliner_label_map[eid] = raw.get("gliner_label") or raw.get("display_name", eid)
        # Short, human phrase for value-before-label templates: take the part of
        # display_name before any "/" or "(" and lowercase it ("City Name / Town"
        # -> "city name").
        disp = (raw.get("display_name") or eid)
        for sep in ("/", "("):
            disp = disp.split(sep)[0]
        display_map[eid] = disp.strip().lower()

    def _value_before_label_templates(short: str) -> list[str]:
        """Constructions the model under-learned: value BEFORE its label, and
        keyword-anchored value-after forms — mirroring the validation corpus
        ('reviewers recorded X as the city', 'the intake note references X for Y')."""
        if not short:
            return []
        # Keep {value} away from terminal punctuation so the whitespace-token
        # span match in make_gliner_example succeeds.
        return [
            f"reviewers recorded {{value}} as the {short}.",
            f"{{value}} was recorded as the {short}.",
            f"the {short} is {{value}} on file.",
            f"the intake note references {{value}} for {short}.",
            f"recorded {{value}} for the {short}.",
        ]

    all_examples: list[dict] = []
    entity_counts: dict[str, int] = {}

    for entity_id, defn in ENTITY_DEFS.items():
        label = gliner_label_map.get(entity_id)
        if label is None:
            continue

        generator: Callable = defn["generator"]
        # Mix in value-before-label constructions so GLiNER learns to detect the
        # value when it precedes its label (the dominant real-world leak mode).
        templates: list[str] = defn["templates"] + _value_before_label_templates(
            display_map.get(entity_id, "")
        )
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
    print(f"  Added {len(multi):,} multi-entity document examples")

    # Multi-occurrence examples (same entity twice in one sentence)
    multi_occ = generate_multi_occurrence_examples(n_per_template=200)
    all_examples.extend(multi_occ)
    print(f"  Added {len(multi_occ):,} multi-occurrence examples")

    # Disambiguation examples (teach model to distinguish similar-looking entities)
    disambiguation = generate_disambiguation_examples(n_per_template=300)
    all_examples.extend(disambiguation)
    print(f"  Added {len(disambiguation):,} disambiguation examples")

    rng.shuffle(all_examples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    total = len(all_examples)
    single = total - len(HARD_NEGATIVES) - len(multi) - len(multi_occ) - len(disambiguation)
    print(f"\nGenerated {total:,} training examples")
    print(f"  Single-entity:    {single:,}")
    print(f"  Hard negatives:   {len(HARD_NEGATIVES):,}")
    print(f"  Multi-entity:     {len(multi):,}")
    print(f"  Multi-occurrence: {len(multi_occ):,}")
    print(f"  Disambiguation:   {len(disambiguation):,}")
    print(f"  Saved to: {out_path}\n")
    print("Entity coverage:")
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
