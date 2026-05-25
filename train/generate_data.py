
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
    fmt = random.choice(["parens", "dash", "dot", "intl", "ext", "bare10"])
    if fmt == "parens":
        return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"
    elif fmt == "dash":
        return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    elif fmt == "dot":
        return f"{random.randint(200,999)}.{random.randint(200,999)}.{random.randint(1000,9999)}"
    elif fmt == "intl":
        return f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    elif fmt == "bare10":
        # GAP-04: bare 10-digit number no separators
        return f"{random.randint(2000000000, 9999999999)}"
    else:
        return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}x{random.randint(100,999)}"

def _phone2() -> str:
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def _fax() -> str:
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"

def _ssn() -> str:
    return f"{random.randint(100,799):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"

def _credit_card() -> str:
    if random.random() < 0.25:
        # GAP-03: 8+4+4 format (Discover-style)
        return f"{random.randint(60110000,60119999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
    return f"{random.randint(4000,4999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def _cvv() -> str:
    return str(random.randint(100, 999))

def _card_exp() -> str:
    return f"{random.randint(1,12):02d}/{random.randint(25,30):02d}"

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
    return fake.ipv4_private()

def _mac() -> str:
    return ":".join(f"{random.randint(0,255):02X}" for _ in range(6))

def _url() -> str:
    return f"https://portal.hospital.org/patients/{random.randint(10000,99999)}"

def _mrn_value() -> str:
    return str(random.randint(1000000, 9999999))

def _npi() -> str:
    digits = str(random.randint(1000000000, 9999999999))
    # 30% chance of NPI- prefix format (since this is a real test failure)
    if random.random() < 0.3:
        return f"NPI-{digits}"
    return digits

def _npi_plain() -> str:
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
    lon = round(random.uniform(66.0, 124.0), 6)
    if random.random() < 0.3:
        # GAP-11: degree-symbol format
        return f"{lat}°N, {lon}°W"
    return f"{lat}, -{lon}"

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

def _city() -> str:
    return fake.city()

def _email() -> str:
    return fake.email()

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

def _case_number_alphanumeric() -> str:
    prefixes = ["CASE", "CR", "CV", "CF", "DK"]
    return f"{random.choice(prefixes)}-{random.randint(2020,2025)}-{random.randint(1000,99999):05d}"

def _phone_dotted() -> str:
    return f"{random.randint(200,999)}.{random.randint(200,999)}.{random.randint(1000,9999)}"

def _phone_with_ext() -> str:
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}x{random.randint(100,9999)}"

def _npi_with_prefix() -> str:
    return f"NPI-{random.randint(1000000000, 9999999999)}"

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
    if random.random() < 0.4:
        # GAP-15: WRT- prefix format
        return f"WRT-{random.randint(10000,9999999)}"
    return f"W{random.randint(2020,2024)}{random.randint(10000,99999)}"

def _incident_num() -> str:
    if random.random() < 0.4:
        # GAP-20: INC-YYYY-NNN format
        return f"INC-{random.randint(2020,2024)}-{random.randint(100,99999)}"
    return f"IR#{random.randint(2020,2024)}{random.randint(10000,99999)}"

def _bart_emp_id() -> str:
    return str(random.randint(10000,999999))

def _passport_num() -> str:
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10000000,99999999)}"

def _drivers_license() -> str:
    return f"D{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def _state_id_num() -> str:
    if random.random() < 0.4:
        # GAP-09: SID-STATE-YYYYNNN format
        return f"SID-{_state_abbr()}-{random.randint(10000000, 999999999)}"
    return f"ID{random.randint(100000,9999999)}"

def _med_license() -> str:
    r = random.random()
    if r < 0.35:
        # GAP-07: ML-STATE-NNNNNN state-infix format
        return f"ML-{_state_abbr()}-{random.randint(100000,9999999)}"
    if r < 0.65:
        # ML-NNNNNN prefix format
        return f"ML-{random.randint(100000,9999999)}"
    return f"ML{random.randint(100000,9999999)}"

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
    return fake.user_name()

def _dna_str_locus() -> str:
    # GAP-10: CODIS STR locus-allele format
    loci = ["D3S1358","vWA","TH01","TPOX","CSF1PO","FGA","D7S820","D13S317",
            "D16S539","D2S1338","D21S11","D18S51","D5S818","D8S1179","PentaE","PentaD","SE33"]
    return f"{random.choice(loci)}-{random.randint(6,30)}"

def _voiceprint_vp() -> str:
    # GAP-13: VP-NNNNNN format
    return f"VP-{random.randint(1000,9999999)}"

def _ferpa_id() -> str:
    # GAP-16: FERPA FER- and STUDENT_RECORDS_FERPA- formats
    if random.random() < 0.5:
        return f"FERPA FER-{random.randint(10000,9999999)}"
    return f"STUDENT_RECORDS_FERPA-{random.randint(1000,9999999)}"

def _cpo_order() -> str:
    # GAP-24: CPO-YYYY-NNN format
    return f"CPO-{random.randint(2018,2024)}-{random.randint(100,99999)}"

def _driver_history_dh() -> str:
    # GAP-19: DH-NNNNNN format
    return f"DH-{random.randint(10000,9999999)}"

def _confidential_ref() -> str:
    # GAP-12: Confidential document reference
    kind = random.choice(["memo","document","file","record","report"])
    pfx = random.choice(["CONF","INT","SEC","PRIV"])
    return f"Confidential {kind} {pfx}-{random.randint(1000,9999999)}"

def _username_bare() -> str:
    # GAP-21: user_xxx format
    base = fake.user_name().replace("-","_").lower()[:15]
    if not base.startswith("user_"):
        base = "user_" + base
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
            f"SIG-{random.randint(100000, 999999)}",
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
            "CONFIDENTIAL", "TOP SECRET", "RESTRICTED", "SENSITIVE PII",
            _confidential_ref(),
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
        "generator": lambda: f"SOR-{random.randint(10000,999999)}",
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
        "generator": lambda: f"PR-{random.randint(10000,999999)}",
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
