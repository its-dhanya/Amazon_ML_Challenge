"""
Step 5b: Evaluation Harness
============================
Compute blocking recall, F0.5, and diagnostics on the val split.
Run this BEFORE step5 to verify blocking quality, and AFTER to check
final scores. This is the "Check + iterate" node in the pipeline.

Metrics
-------
  blocking_recall   = true_matches_in_candidates / total_true_matches
  avg_cands_per_s1  = mean candidate count per S1 entity
  singleton_rate    = fraction of S1 entities with exactly 1 candidate
  F0.5, P, R        = classification performance after threshold

WATCH OUT FOR:
- Blocking recall MUST be checked before training the classifier.
  A low blocking recall (< 90%) means the classifier can never recover
  those missed pairs, no matter how good the features are.
- Use the val split for all tuning — NEVER peek at test labels.
- The "France proxy" check: hold out all India rows from train,
  evaluate recall on them as a proxy for unseen France distribution.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix

CLEAN_DIR = Path("data/cleaned")
CAND_DIR  = Path("data/candidates")
GT_DIR    = Path("data/processed")
OUT_DIR   = Path("data/results")


def load_gt(split: str) -> pd.DataFrame:
    gt_path = GT_DIR / f"{split}_positive_pairs.parquet"
    if not gt_path.exists():
        print(f"  WARNING: No GT found at {gt_path}")
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_s2s3", "label"])
    return pd.read_parquet(gt_path)


def blocking_recall(
    candidates: pd.DataFrame,
    gt_positives: pd.DataFrame,
) -> dict:
    """
    What fraction of true matches appear in the candidate set?

    A true match is 'recalled' if the pair (s1_id, s2s3_id) appears
    anywhere in the candidate set, regardless of rank.
    """
    cand_set = set(
        zip(candidates["entity_id_s1"], candidates["entity_id_s2s3"])
    )
    gt_set = set(
        zip(gt_positives["entity_id_s1"], gt_positives["entity_id_s2s3"])
    )

    recalled = cand_set & gt_set
    n_gt     = len(gt_set)
    n_recalled = len(recalled)

    recall = n_recalled / n_gt if n_gt > 0 else 0.0
    missed = gt_set - recalled

    return {
        "total_true_matches":     n_gt,
        "recalled_in_candidates": n_recalled,
        "blocking_recall":        round(recall, 4),
        "missed_pairs":           len(missed),
        "missed_examples":        list(missed)[:5],   # first 5 for inspection
    }


def candidate_stats(candidates: pd.DataFrame) -> dict:
    per_s1 = candidates.groupby("entity_id_s1").size()
    return {
        "total_pairs":        len(candidates),
        "unique_s1":          candidates["entity_id_s1"].nunique(),
        "avg_cands_per_s1":   round(per_s1.mean(), 2),
        "median_cands":       int(per_s1.median()),
        "max_cands":          int(per_s1.max()),
        "singleton_rate":     round((per_s1 == 1).mean(), 4),
    }


def classification_metrics(
    candidates: pd.DataFrame,
    gt_positives: pd.DataFrame,
    score_col: str = "match_score",
    threshold: float = 0.5,
) -> dict:
    if score_col not in candidates.columns:
        return {"note": f"No {score_col!r} column found — run step5 first."}

    gt_set = set(
        zip(gt_positives["entity_id_s1"], gt_positives["entity_id_s2s3"])
    )

    candidates = candidates.copy()
    candidates["label"] = candidates.apply(
        lambda r: 1 if (r["entity_id_s1"], r["entity_id_s2s3"]) in gt_set else 0,
        axis=1,
    )
    candidates["pred"] = (candidates[score_col] >= threshold).astype(int)

    y_true = candidates["label"].values
    y_pred = candidates["pred"].values

    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": threshold,
        "F0.5":      round(f05, 4),
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "confusion_matrix": cm,   # [[TN, FP], [FN, TP]]
    }


def country_breakdown(
    candidates: pd.DataFrame,
    gt_positives: pd.DataFrame,
    cleaned_s1: pd.DataFrame,
) -> list[dict]:
    """Blocking recall broken down by country (for the France-proxy check)."""
    candidates = candidates.merge(
        cleaned_s1[["entity_id", "country"]].rename(
            columns={"entity_id": "entity_id_s1", "country": "country"}
        ),
        on="entity_id_s1", how="left",
    )
    gt_positives = gt_positives.merge(
        cleaned_s1[["entity_id", "country"]].rename(
            columns={"entity_id": "entity_id_s1", "country": "country"}
        ),
        on="entity_id_s1", how="left",
    )

    rows = []
    for country, cand_c in candidates.groupby("country"):
        gt_c = gt_positives[gt_positives["country"] == country]
        stats = blocking_recall(cand_c, gt_c)
        rows.append({"country": country, **stats})
    return rows


def run():
    split = "val"
    print(f"\n=== Evaluation on {split} split ===")

    cand_path = CAND_DIR / f"candidate_pairs_{split}.parquet"
    if not cand_path.exists():
        print(f"  No candidate file found at {cand_path}. Run step4 first.")
        return

    candidates  = pd.read_parquet(cand_path)
    gt_positive = load_gt(split)

    # ── Blocking recall ───────────────────────────────────────────────────────
    print("\n-- Blocking recall --")
    br = blocking_recall(candidates, gt_positive)
    for k, v in br.items():
        if k != "missed_examples":
            print(f"  {k}: {v}")
    if br["missed_examples"]:
        print(f"  missed_examples (first 5): {br['missed_examples']}")

    # ── Candidate stats ───────────────────────────────────────────────────────
    print("\n-- Candidate stats --")
    cs = candidate_stats(candidates)
    for k, v in cs.items():
        print(f"  {k}: {v}")

    # ── Classification metrics (if scores present) ────────────────────────────
    if "match_score" in candidates.columns:
        print("\n-- Classification metrics (at stored threshold) --")
        # Try to load saved threshold
        model_path = Path("data/models/fs_model.pkl")
        threshold = 0.5
        if model_path.exists():
            import joblib
            bundle = joblib.load(model_path)
            threshold = bundle.get("threshold", 0.5)

        cm_stats = classification_metrics(candidates, gt_positive, threshold=threshold)
        for k, v in cm_stats.items():
            print(f"  {k}: {v}")

    # ── Country breakdown ─────────────────────────────────────────────────────
    s1_path = Path("data/cleaned") / f"{split}_s1.parquet"
    if s1_path.exists():
        cleaned_s1 = pd.read_parquet(s1_path)
        print("\n-- Blocking recall by country --")
        rows = country_breakdown(candidates, gt_positive, cleaned_s1)
        for row in rows:
            print(f"  {row['country']}: recall={row['blocking_recall']:.4f} "
                  f"({row['recalled_in_candidates']}/{row['total_true_matches']})")

    # ── France-proxy: hold-out India rows from train ──────────────────────────
    print("\n-- France-proxy check (India hold-out) --")
    train_cand_path = CAND_DIR / "candidate_pairs_train.parquet"
    train_gt_path   = GT_DIR / "train_positive_pairs.parquet"
    train_s1_path   = Path("data/cleaned") / "train_s1.parquet"

    if train_cand_path.exists() and train_s1_path.exists():
        train_cands  = pd.read_parquet(train_cand_path)
        train_gt_pos = load_gt("train")
        train_s1     = pd.read_parquet(train_s1_path)

        india_s1_ids = set(train_s1[train_s1["country"] == "India"]["entity_id"])
        india_cands  = train_cands[train_cands["entity_id_s1"].isin(india_s1_ids)]
        india_gt     = train_gt_pos[train_gt_pos["entity_id_s1"].isin(india_s1_ids)]

        if len(india_gt) > 0:
            br_india = blocking_recall(india_cands, india_gt)
            print(f"  India blocking recall: {br_india['blocking_recall']:.4f} "
                  f"({br_india['recalled_in_candidates']}/{br_india['total_true_matches']})")
            print("  (This is your proxy for France generalisation)")
        else:
            print("  No India GT rows found in train.")
    else:
        print("  Train data not found; skipping France-proxy check.")

    print("\n=== ACTION THRESHOLDS ===")
    br_val = br["blocking_recall"]
    if br_val < 0.90:
        print(f"  ⚠ Blocking recall {br_val:.2%} < 90% — MUST improve before classifier.")
        print("    → Raise TOP_K_BM25/TOP_K_DENSE in step4")
        print("    → Add character n-gram features to BM25 corpus")
        print("    → Check if Indic scripts are reaching the index un-transliterated")
    elif br_val < 0.95:
        print(f"  ⚡ Blocking recall {br_val:.2%} — acceptable but room to improve.")
    else:
        print(f"  ✓ Blocking recall {br_val:.2%} — good. Focus on classifier.")


if __name__ == "__main__":
    run()
