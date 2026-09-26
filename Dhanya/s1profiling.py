"""Step 1: streamable profiling for raw challenge TSVs (read-only)."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

DATASET, OUT = Path("student_resource/dataset"), Path("data/profiles/profile.json")

def profile(path: Path) -> dict:
    countries, empty_name, empty_address, non_ascii, devanagari = Counter(), 0, 0, 0, 0
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows += 1; countries[row["country"]] += 1
            text = f"{row['business_name']} {row['business_address']}"
            empty_name += not row["business_name"].strip(); empty_address += not row["business_address"].strip()
            non_ascii += any(ord(c) > 127 for c in text)
            devanagari += any("\u0900" <= c <= "\u097f" for c in text)
    return {"rows": rows, "countries": dict(countries), "empty_name": empty_name, "empty_address": empty_address, "non_ascii": non_ascii, "devanagari": devanagari}

def run() -> None:
    results = {str(p): profile(p) for p in sorted(DATASET.glob("*/*_source*.tsv"))}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run()
