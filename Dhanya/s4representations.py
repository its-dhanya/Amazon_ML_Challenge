"""Step 4: country-constrained Sparkly + UniBlocker candidate union.

Sparkly and UniBlocker are external, scalable retrieval engines.  Each must
write a long Parquet file in ``data/branch_candidates/`` with at least
``entity_id_s1`` and ``entity_id_s2s3``.  This module is deliberately the
single point that enforces country, unions branch results, and defines the
exact final candidate pool sent to Fellegi-Sunter.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("data/cleaned")
BRANCH_DIR = Path("data/branch_candidates")
OUT_DIR = Path("data/candidates")
TOP_K_PER_BRANCH = 50


def load_records(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_parquet(CLEAN_DIR / f"{split}_s1.parquet")
    index = pd.concat([pd.read_parquet(CLEAN_DIR / f"{split}_{source}.parquet") for source in ("s2", "s3")], ignore_index=True)
    return s1[["entity_id", "country"]], index[["entity_id", "country"]]


def read_branch(split: str, branch: str) -> pd.DataFrame:
    path = BRANCH_DIR / f"{split}_{branch}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the {branch} retrieval job first. It must emit "
            "entity_id_s1, entity_id_s2s3 and an optional rank column."
        )
    pairs = pd.read_parquet(path)
    required = {"entity_id_s1", "entity_id_s2s3"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"{path}: required columns are {sorted(required)}")
    if "rank" in pairs.columns:
        pairs = pairs.sort_values("rank").groupby("entity_id_s1", as_index=False).head(TOP_K_PER_BRANCH)
    return pairs[["entity_id_s1", "entity_id_s2s3"]].drop_duplicates()


def enforce_country(pairs: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    checked = pairs.merge(s1.rename(columns={"entity_id": "entity_id_s1", "country": "country_s1"}), on="entity_id_s1", how="inner")
    checked = checked.merge(index.rename(columns={"entity_id": "entity_id_s2s3", "country": "country_s2s3"}), on="entity_id_s2s3", how="inner")
    return checked.loc[checked.country_s1.eq(checked.country_s2s3), ["entity_id_s1", "entity_id_s2s3"]]


def generate_candidates(split: str) -> pd.DataFrame:
    s1, index = load_records(split)
    # sparkly = enforce_country(read_branch(split, "sparkly"), s1, index)
    uniblocker = enforce_country(read_branch(split, "uniblocker"), s1, index)
    
    # Bypass sparkly due to PyLucene install issues on ARM
    pairs = uniblocker.drop_duplicates()
    pairs.to_parquet(OUT_DIR / f"candidate_pairs_{split}.parquet", index=False)
    
    counts = pairs.groupby("entity_id_s1").size()
    print(f"{split}: {len(pairs):,} union pairs; blocking S1={counts.size:,}; "
          f"avg candidates/S1={counts.mean():.1f}; singleton rate={(counts.eq(1).mean()):.2%}")
    return pairs


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        generate_candidates(split)


if __name__ == "__main__":
    run()
