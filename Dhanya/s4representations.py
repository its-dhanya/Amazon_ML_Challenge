"""Step 4: country-constrained lexical (+ optional UniBlocker) candidate union.

Each branch writes a long Parquet file in ``data/branch_candidates/`` with
``entity_id_s1``, ``entity_id_s2s3`` and a per-query ``rank`` (1 = best); the
lexical branch also writes ``score``. This module enforces country, keeps the
top ``TOP_K_PER_BRANCH`` per query, unions the branches, and writes the exact
candidate pool sent to the final model, carrying ``lex_rank``, ``lex_score`` and
``dense_rank`` along so later steps need no further joins.

    python s4representations.py [val] [test]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("data/cleaned")
BRANCH_DIR = Path("data/branch_candidates")
OUT_DIR = Path("data/candidates")
TOP_K_PER_BRANCH = 20


def load_records(split: str) -> tuple[pd.Series, pd.Series]:
    s1 = pd.read_parquet(CLEAN_DIR / f"{split}_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    index = pd.concat(
        [pd.read_parquet(CLEAN_DIR / f"{split}_{source}.parquet", columns=["entity_id", "country"]) for source in ("s2", "s3")],
        ignore_index=True,
    ).set_index("entity_id").country
    return s1, index


def read_branch(split: str, branch: str) -> pd.DataFrame:
    path = BRANCH_DIR / f"{split}_{branch}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the {branch} retrieval job first.")
    pairs = pd.read_parquet(path)
    required = {"entity_id_s1", "entity_id_s2s3", "rank"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"{path}: required columns are {sorted(required)}")
    pairs = pairs[pairs["rank"] <= TOP_K_PER_BRANCH]  # rank is per query, 1 = best
    return pairs.drop_duplicates(["entity_id_s1", "entity_id_s2s3"])


def enforce_country(pairs: pd.DataFrame, s1: pd.Series, index: pd.Series) -> pd.DataFrame:
    c1 = pairs.entity_id_s1.map(s1).to_numpy()
    c2 = pairs.entity_id_s2s3.map(index).to_numpy()
    ok = (c1 == c2) & pd.notna(c1)
    if not ok.all():
        print(f"  dropped {(~ok).sum():,} cross-country/unknown pairs")
    return pairs[ok]


def generate_candidates(split: str) -> pd.DataFrame:
    s1, index = load_records(split)
    pairs = enforce_country(read_branch(split, "sparkly"), s1, index).rename(columns={"rank": "lex_rank", "score": "lex_score"})
    pairs["dense_rank"] = np.nan
    if (BRANCH_DIR / f"{split}_uniblocker.parquet").exists():
        dense = enforce_country(read_branch(split, "uniblocker"), s1, index)[["entity_id_s1", "entity_id_s2s3", "rank"]]
        pairs = pairs.merge(dense.rename(columns={"rank": "dense_rank_new"}), how="outer", on=["entity_id_s1", "entity_id_s2s3"])
        pairs["dense_rank"] = pairs.pop("dense_rank_new")
    else:
        print(f"{split}: no uniblocker branch found, using lexical only")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT_DIR / f"candidate_pairs_{split}.parquet", index=False)

    counts = pairs.groupby("entity_id_s1").size()
    recall_str = "N/A"
    gt_path = Path("data/processed") / f"{split}_positive_pairs.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path)[["entity_id_s1", "entity_id_s2s3"]]
        found = gt.merge(pairs[["entity_id_s1", "entity_id_s2s3"]], how="inner", on=["entity_id_s1", "entity_id_s2s3"])
        recall_str = f"{len(found) / len(gt):.2%}"
    print(f"{split}: {len(pairs):,} union pairs; blocking S1={counts.size:,}/{len(s1):,}; "
          f"avg cands/S1={counts.mean():.1f}; singletons={counts.eq(1).mean():.2%}; recall={recall_str}")
    return pairs


def run() -> None:
    for split in (sys.argv[1:] or ["val"]):
        generate_candidates(split)


if __name__ == "__main__":
    run()
