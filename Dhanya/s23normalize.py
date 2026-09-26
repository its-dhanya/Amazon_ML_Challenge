"""
Step 2 + 3: Cleaning and Normalization
=======================================
Produces cleaned columns alongside the raw ones (raw columns are preserved).
The single entry point is `clean_record()` — call it identically at index time
AND at query time. This is the most important invariant in the whole pipeline.

WATCH OUT FOR:
- NFKD + strip accents must happen BEFORE lowercasing (order matters)
- Legal suffix normalization must use whole-word matching (re.sub with \\b)
  otherwise "corporation" inside "incorporation" gets mangled
- Indic→Latin transliteration is expensive; cache results or batch with transliterate lib
- Address component splitting produces NaN for missing parts — fill with ""
- empty → "missing" flag must be applied AFTER all cleaning, not before
- Do NOT strip digits from addresses — house numbers are signal
- The "&" → "and" swap must also handle " & " (spaces both sides) and "& " etc.
- Run this on BOTH train and test with the exact same function — any branch on
  split name inside this file is a bug
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("data/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── legal suffix table ────────────────────────────────────────────────────────
# Map variant → canonical token.  Add more from profiling findings.
LEGAL_SUFFIXES: list[tuple[str, str]] = [
    (r"\bprivate\b",      "pvt"),
    (r"\bpvt\.?\b",       "pvt"),
    (r"\blimited\b",      "ltd"),
    (r"\bltd\.?\b",       "ltd"),
    (r"\bcorporation\b",  "corp"),
    (r"\bcorp\.?\b",      "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\binc\.?\b",       "inc"),
    (r"\bllc\.?\b",       "llc"),
    (r"\bllp\.?\b",       "llp"),
    (r"\bcompany\b",      "co"),
    (r"\bco\.?\b",        "co"),
    (r"\benterprises?\b", "ent"),
    (r"\bindustries?\b",  "ind"),
    (r"\bservices?\b",    "svc"),
    (r"\bgroup\b",        "grp"),
    (r"\binternational\b","intl"),
    (r"\bassociates?\b",  "assoc"),
    (r"\bbrothers?\b",    "bros"),
    (r"\bsolutions?\b",   "sol"),
    (r"\btechnologies?\b","tech"),
]

# Pre-compile legal suffix patterns (case-insensitive)
_LEGAL_RE = [(re.compile(pat, re.IGNORECASE), canon) for pat, canon in LEGAL_SUFFIXES]

# Address abbreviations
ADDR_ABBREV: list[tuple[str, str]] = [
    (r"\broad\b",    "rd"),
    (r"\bstreet\b",  "st"),
    (r"\bavenue\b",  "ave"),
    (r"\bboulevard\b","blvd"),
    (r"\bnagar\b",   "ngr"),
    (r"\bcolony\b",  "col"),
    (r"\bdistrict\b","dist"),
    (r"\bphase\b",   "ph"),
    (r"\bsector\b",  "sec"),
    (r"\bfloor\b",   "fl"),
    (r"\bbuilding\b","bldg"),
    (r"\bappartement\b","apt"),  # French
    (r"\brue\b",     "rue"),     # French — keep as-is (already short)
]
_ADDR_RE = [(re.compile(pat, re.IGNORECASE), canon) for pat, canon in ADDR_ABBREV]

# Stray symbols to strip (keep digits, letters, spaces, hyphens, commas, dots, slashes)
_STRAY_RE = re.compile(r"[<>#@$%^*=~`|\\+]")

# Collapse multiple whitespace
_WS_RE = re.compile(r"\s+")


# ── core cleaning ─────────────────────────────────────────────────────────────

def nfkd_clean(text: str) -> str:
    """
    NFKD decomposition with Latin accent folding while preserving non-Latin script.

    Do not ASCII-encode here: that previously erased Indic strings before the
    semantic/UCL branch could represent or transliterate them.
    """
    normalized = unicodedata.normalize("NFKD", text)
    kept: list[str] = []
    for char in normalized:
        # Drop the accent in e.g. "é", but keep vowel signs in Indic scripts.
        if unicodedata.combining(char) and kept and "LATIN" in unicodedata.name(kept[-1], ""):
            continue
        kept.append(char)
    return "".join(kept)


def clean_name(raw: str) -> str:
    """Clean a business name field."""
    if not raw.strip():
        return "missing"

    text = nfkd_clean(raw)
    text = text.lower()

    # & → and (handle all spacing variants)
    text = re.sub(r"\s*&\s*", " and ", text)

    # strip stray symbols
    text = _STRAY_RE.sub(" ", text)

    # collapse whitespace
    text = _WS_RE.sub(" ", text).strip()

    # normalize legal suffixes (whole-word only)
    for pattern, canon in _LEGAL_RE:
        text = pattern.sub(canon, text)

    # remove trailing punctuation (., ;)
    text = re.sub(r"[.,;]+$", "", text).strip()

    return text if text else "missing"


def clean_address(raw: str) -> str:
    """Clean a business address field."""
    if not raw.strip():
        return "missing"

    text = nfkd_clean(raw)
    text = text.lower()

    # & → and
    text = re.sub(r"\s*&\s*", " and ", text)

    # strip stray symbols (but keep / for address notation like "12/3")
    text = re.sub(r"[<>#@$%^*=~`|\\+]", " ", text)

    # normalize address abbreviations
    for pattern, canon in _ADDR_RE:
        text = pattern.sub(canon, text)

    # collapse whitespace
    text = _WS_RE.sub(" ", text).strip()

    return text if text else "missing"


def split_address_components(address_clean: str) -> dict[str, str]:
    """
    Heuristic component split: try to extract PIN/ZIP and state code.

    WATCH OUT:
    - Indian PINs are 6 digits; US ZIPs are 5 (or 5+4); French codes are 5
    - State codes are 2-char uppercase abbreviations in US; not reliable in India
    - Return empty strings for components not found — never None
    - This is a best-effort split; missing components are fine
    """
    pin_match = re.search(r"\b(\d{5,6})\b", address_clean)
    pin = pin_match.group(1) if pin_match else ""

    # Use a conservative US-style state feature only.  It is evidence, never a filter.
    state_match = re.search(r"\b(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|ia|id|il|in|ks|ky|la|ma|md|me|mi|mn|mo|ms|mt|nc|nd|ne|nh|nj|nm|nv|ny|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|va|vt|wa|wi|wv|wy)\b", address_clean)
    state = state_match.group(1) if state_match else ""

    return {"pin_zip": pin, "state_code": state}


def clean_record(row: pd.Series) -> pd.Series:
    """
    Master cleaning function. Returns a Series with new columns appended.

    THIS FUNCTION MUST BE CALLED IDENTICALLY AT INDEX TIME AND QUERY TIME.
    Never add a split-conditional branch here.
    """
    name_clean = clean_name(row.get("business_name", ""))
    addr_clean = clean_address(row.get("business_address", ""))
    components = split_address_components(addr_clean)

    return pd.Series({
        # preserve raw
        "entity_id":        row["entity_id"],
        "business_name":    row["business_name"],
        "business_address": row["business_address"],
        "country":          row["country"],
        # cleaned
        "name_clean":  name_clean,
        "addr_clean":  addr_clean,
        "pin_zip":     components["pin_zip"],
        "state_code":  components["state_code"],
        # combined field for BM25 indexing
        "text_for_index": f"{name_clean} {addr_clean}".strip(),
    })


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_record to a whole dataframe without memory bloat."""
    from tqdm import tqdm
    results = []
    
    b_names = df.get("business_name", pd.Series([""] * len(df))).fillna("").astype(str)
    b_addrs = df.get("business_address", pd.Series([""] * len(df))).fillna("").astype(str)
    e_ids = df["entity_id"]
    countries = df["country"]
    
    for eid, name, addr, country in tqdm(zip(e_ids, b_names, b_addrs, countries), total=len(df), desc="Normalizing"):
        name_clean = clean_name(name)
        addr_clean = clean_address(addr)
        components = split_address_components(addr_clean)
        
        results.append({
            "entity_id": eid,
            "business_name": name,
            "business_address": addr,
            "country": country,
            "name_clean": name_clean,
            "addr_clean": addr_clean,
            "pin_zip": components["pin_zip"],
            "state_code": components["state_code"],
            "text_for_index": f"{name_clean} {addr_clean}".strip(),
        })
        
    return pd.DataFrame(results)


def run():
    for fname in DATA_DIR.glob("*.parquet"):
        df = pd.read_parquet(fname)
        if "entity_id" not in df.columns:
            continue
        cleaned = process_df(df)
        out = OUT_DIR / fname.name
        cleaned.to_parquet(out, index=False)
        print(f"Cleaned {fname.name} → {out.name}  ({len(cleaned):,} rows)")

    print("\n=== THINGS TO CHECK AFTER CLEANING ===")
    print("1. Sample 20 name_clean rows — do legal suffixes look right?")
    print("   → Watch for 'incorporation' → 'incorp' (should stay 'inc' only on whole word)")
    print("2. Check 'missing' rate in name_clean vs original empty rate")
    print("   → Should be similar; large difference means over-aggressive stripping")
    print("3. Spot-check Indian addresses — Indic scripts should be gone (NFKD'd away)")
    print("   → If Devanagari remains, transliteration step needed before NFKD")
    print("4. Verify text_for_index is non-empty for >95% of rows")
    print("5. Check pin_zip extraction on US vs India — 5-digit vs 6-digit")


if __name__ == "__main__":
    run()
