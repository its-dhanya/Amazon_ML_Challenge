"""
Step 5: Fellegi–Sunter Pairwise Scoring + F0.5-Optimised Threshold
===================================================================
For each candidate pair from Step 4, compute a rich feature vector and
train a probabilistic match classifier (logistic regression as a lightweight
FS proxy). Then tune threshold on val to maximise F0.5.

Feature set
-----------
  Names:    Jaro-Winkler, token-sort ratio, token-set ratio, exact match
  Address:  Jaro-Winkler on full address, PIN/ZIP exact, state_code exact,
            Jaccard on address tokens
  Country:  Binary exact match (should always be 1 after country filter,
            but keep as a sanity feature)
  Combined: BM25 rank, dense rank, RRF score (from Step 4)

Pipeline
--------
  1. Load candidate pairs + cleaned records
  2. Join to get both sides of each pair
  3. Compute features (vectorised where possible, row-wise for string metrics)
  4. Train LR on train pairs with GT labels
  5. Predict on val; sweep threshold for best F0.5
  6. Apply threshold to test; write matching_results.tsv + candidate_pairs.tsv

WATCH OUT FOR:
- F0.5 weights precision 2×; threshold will be higher than for F1.
  Start sweep at 0.3 and go up to 0.95.
- jellyfish (Jaro-Winkler) and rapidfuzz (token ratios) are fast C extensions;
  don't use pure-Python difflib — it's 20× slower.
- If a cleaned field is "missing", treat string similarity as 0.0, NOT as a
  high match (two "missing" names are NOT evidence of a match).
- Class imbalance is severe (many more non-matches). Pass class_weight="balanced"
  to LogisticRegression, or undersample non-matches during training.
- Preserve candidate_pairs.tsv (all pairs before threshold) for leaderboard
  submission alongside matching_results.tsv.
- Never leak val/test labels into feature engineering or threshold selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, precision_score, recall_score
import jellyfish
from rapidfuzz import fuzz
import joblib

# ── paths ─────────────────────────────────────────────────────────────────────
CLEAN_DIR     = Path("data/cleaned")
CAND_DIR      = Path("data/candidates")
GT_DIR        = Path("data/processed")        # ground-truth lives here
OUT_DIR       = Path("student_resource/output")
MODEL_DIR     = Path("data/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def safe_jw(a: str, b: str) -> float:
    """Jaro-Winkler; returns 0 if either side is missing sentinel."""
    if a == "missing" or b == "missing" or not a or not b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(a, b)


def safe_token_sort(a: str, b: str) -> float:
    if a == "missing" or b == "missing" or not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def safe_token_set(a: str, b: str) -> float:
    if a == "missing" or b == "missing" or not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def jaccard_tokens(a: str, b: str) -> float:
    if a == "missing" or b == "missing" or not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def exact_match(a: str, b: str) -> float:
    return 1.0 if a == b and a != "missing" and a != "" else 0.0


def pin_match(a: str, b: str) -> float:
    """Exact PIN/ZIP match; 0 if either is empty."""
    if not a or not b:
        return 0.0
    return 1.0 if a == b else 0.0


# ── feature computation ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    # name features
    "name_jw",
    "name_token_sort",
    "name_token_set",
    "name_exact",
    "name_jaccard",
    # address features
    "addr_jw",
    "addr_token_sort",
    "addr_jaccard",
    "pin_exact",
    "state_exact",
    # country feature
    "country_exact",
    # ranking signals from step 4
    "bm25_rank_inv",       # 1/(rank) so higher = better
    "dense_rank_inv",
    "rrf_score",
]


def compute_features(row: pd.Series) -> np.ndarray:
    """
    Compute feature vector for one candidate pair row.

    row must have: name_clean_s1, name_clean_s2s3,
                   addr_clean_s1, addr_clean_s2s3,
                   pin_zip_s1, pin_zip_s2s3,
                   state_code_s1, state_code_s2s3,
                   country_s1, country_s2s3,
                   bm25_rank, dense_rank, rrf_score
    """
    n1, n2 = row["name_clean_s1"], row["name_clean_s2s3"]
    a1, a2 = row["addr_clean_s1"], row["addr_clean_s2s3"]
    p1, p2 = row["pin_zip_s1"], row["pin_zip_s2s3"]
    st1, st2 = row["state_code_s1"], row["state_code_s2s3"]
    c1, c2 = row["country_s1"], row["country_s2s3"]

    bm25_r  = row.get("bm25_rank")
    dense_r = row.get("dense_rank")
    rrf     = row.get("rrf_score")

    feats = np.array([
        safe_jw(n1, n2),
        safe_token_sort(n1, n2),
        safe_token_set(n1, n2),
        exact_match(n1, n2),
        jaccard_tokens(n1, n2),

        safe_jw(a1, a2),
        safe_token_sort(a1, a2),
        jaccard_tokens(a1, a2),
        pin_match(p1, p2),
        exact_match(st1, st2),

        exact_match(c1, c2),

        (1.0 / bm25_r  if pd.notna(bm25_r)  and bm25_r  > 0 else 0.0),
        (1.0 / dense_r if pd.notna(dense_r) and dense_r > 0 else 0.0),
        (float(rrf) if pd.notna(rrf) else 0.0),
    ], dtype=np.float32)

    return feats


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Vectorised over rows; falls back to row-wise for string metrics."""
    feats = np.vstack(df.apply(compute_features, axis=1).values)
    return feats


# ── data loading ──────────────────────────────────────────────────────────────

def load_cleaned(split: str) -> dict[str, pd.DataFrame]:
    dfs = {}
    for src in ["s1", "s2", "s3"]:
        p = CLEAN_DIR / f"{split}_{src}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = df[["entity_id", "name_clean", "addr_clean",
                      "pin_zip", "state_code", "country"]].copy()
            dfs[src] = df
    return dfs


def join_pairs(pairs: pd.DataFrame, cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join candidate pairs with cleaned records to get both sides' features.
    S1 is always the query side; S2+S3 merged form the index side.
    """
    s1 = cleaned["s1"].rename(columns={
        "entity_id":   "entity_id_s1",
        "name_clean":  "name_clean_s1",
        "addr_clean":  "addr_clean_s1",
        "pin_zip":     "pin_zip_s1",
        "state_code":  "state_code_s1",
        "country":     "country_s1",
    })

    index_parts = [cleaned[s] for s in ("s2", "s3") if s in cleaned]
    s_index = pd.concat(index_parts, ignore_index=True).rename(columns={
        "entity_id":   "entity_id_s2s3",
        "name_clean":  "name_clean_s2s3",
        "addr_clean":  "addr_clean_s2s3",
        "pin_zip":     "pin_zip_s2s3",
        "state_code":  "state_code_s2s3",
        "country":     "country_s2s3",
    })

    merged = (
        pairs
        .merge(s1,      on="entity_id_s1",   how="left")
        .merge(s_index, on="entity_id_s2s3", how="left")
    )

    # Fill NaN strings (records missing from cleaned set shouldn't crash)
    str_cols = ["name_clean_s1", "name_clean_s2s3",
                "addr_clean_s1", "addr_clean_s2s3",
                "pin_zip_s1", "pin_zip_s2s3",
                "state_code_s1", "state_code_s2s3",
                "country_s1", "country_s2s3"]
    merged[str_cols] = merged[str_cols].fillna("missing")

    return merged


def load_gt(split: str) -> pd.DataFrame:
    """
    Load expanded positive pairs produced by Step 0.  Challenge ground truth is
    list-valued, so it must not be consumed as if it were pair-labelled TSV.
    """
    gt_path = GT_DIR / f"{split}_positive_pairs.parquet"
    if not gt_path.exists():
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_s2s3", "label"])
    gt = pd.read_parquet(gt_path)
    gt["label"] = 1
    return gt


# ── F0.5 threshold sweep ──────────────────────────────────────────────────────

def macro_f05(pred_pairs: pd.DataFrame, truth_pairs: pd.DataFrame, s1_ids: pd.Series) -> float:
    """Challenge metric: average entity-level F0.5, including singletons."""
    predicted = pred_pairs.groupby("entity_id_s1")["entity_id_s2s3"].agg(set).to_dict()
    truth = truth_pairs.groupby("entity_id_s1")["entity_id_s2s3"].agg(set).to_dict()
    scores = []
    for s1 in s1_ids:
        p, y = predicted.get(s1, set()), truth.get(s1, set())
        if not p and not y:
            scores.append(1.0); continue
        tp = len(p & y)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(y) if y else 0.0
        scores.append((1.25 * precision * recall / (0.25 * precision + recall)) if precision + recall else 0.0)
    return float(np.mean(scores))


def best_f05_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    pairs: pd.DataFrame,
    truth_pairs: pd.DataFrame,
    s1_ids: pd.Series,
    steps: int = 200,
) -> tuple[float, float]:
    """
    Sweep probability threshold to maximise F0.5.
    F0.5 weights precision 2× over recall.

    WATCH OUT: F0.5 is asymmetric — push threshold HIGH to favour precision.
    """
    best_t, best_f = 0.5, 0.0
    for t in np.linspace(0.2, 0.98, steps):
        preds = (probs >= t).astype(int)
        selected = pairs.loc[preds == 1, ["entity_id_s1", "entity_id_s2s3"]]
        f = macro_f05(selected, truth_pairs, s1_ids)
        if f > best_f:
            best_f, best_t = f, t

    preds_best = (probs >= best_t).astype(int)
    p = precision_score(labels, preds_best, zero_division=0)
    r = recall_score(labels, preds_best, zero_division=0)
    print(f"  Best threshold: {best_t:.3f}  →  F0.5={best_f:.4f}  "
          f"P={p:.4f}  R={r:.4f}")
    return best_t, best_f


# ── main pipeline ─────────────────────────────────────────────────────────────

def run():
    # ── 1. Build features for train ──────────────────────────────────────────
    print("\n[train] Loading candidates + cleaned data …")
    train_pairs = pd.read_parquet(CAND_DIR / "candidate_pairs_train.parquet")
    train_clean = load_cleaned("train")
    train_joined = join_pairs(train_pairs, train_clean)

    train_gt = load_gt("train")
    train_joined = train_joined.merge(
        train_gt[["entity_id_s1", "entity_id_s2s3", "label"]],
        on=["entity_id_s1", "entity_id_s2s3"],
        how="left",
    )
    train_joined["label"] = train_joined["label"].fillna(0).astype(int)

    print(f"  Train pairs: {len(train_joined):,}  "
          f"| matches: {train_joined['label'].sum():,}  "
          f"| ratio: {train_joined['label'].mean():.4%}")

    print("  Computing train features …")
    X_train = build_feature_matrix(train_joined)
    y_train = train_joined["label"].values

    # ── 2. Build features for val ────────────────────────────────────────────
    print("\n[val] Loading candidates + cleaned data …")
    val_pairs = pd.read_parquet(CAND_DIR / "candidate_pairs_val.parquet")
    val_clean = load_cleaned("val")
    val_joined = join_pairs(val_pairs, val_clean)

    val_gt = load_gt("val")
    val_joined = val_joined.merge(
        val_gt[["entity_id_s1", "entity_id_s2s3", "label"]],
        on=["entity_id_s1", "entity_id_s2s3"],
        how="left",
    )
    val_joined["label"] = val_joined["label"].fillna(0).astype(int)

    print("  Computing val features …")
    X_val = build_feature_matrix(val_joined)
    y_val = val_joined["label"].values

    # ── 3. Train classifier ──────────────────────────────────────────────────
    print("\n[model] Training Logistic Regression (Fellegi-Sunter proxy) …")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # handles severe class imbalance
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_train_s, y_train)

    # Feature importance (log-odds coefficients)
    coef_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coef":    clf.coef_[0],
    }).sort_values("coef", ascending=False)
    print("\n  Feature log-odds (top features first):")
    print(coef_df.to_string(index=False))

    # ── 4. Tune threshold on val ─────────────────────────────────────────────
    print("\n[val] Sweeping F0.5 threshold …")
    val_probs = clf.predict_proba(X_val_s)[:, 1]
    threshold, best_f05 = best_f05_threshold(
        val_probs, y_val, val_joined, val_gt, val_clean["s1"]["entity_id"]
    )

    # Save model + scaler + threshold
    joblib.dump({"clf": clf, "scaler": scaler, "threshold": threshold},
                MODEL_DIR / "fs_model.pkl")
    print(f"\n  Saved model → {MODEL_DIR / 'fs_model.pkl'}")

    # ── 5. Apply to test ─────────────────────────────────────────────────────
    print("\n[test] Generating predictions …")
    test_pairs = pd.read_parquet(CAND_DIR / "candidate_pairs_test.parquet")
    test_clean = load_cleaned("test")
    test_joined = join_pairs(test_pairs, test_clean)

    X_test   = build_feature_matrix(test_joined)
    X_test_s = scaler.transform(X_test)
    test_probs = clf.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    # ── 6. Write outputs ──────────────────────────────────────────────────────
    # Challenge TSVs are one row per S1, with comma-separated IDs.  The long
    # candidate table remains internal only; this export is exactly model input.
    test_joined["match_score"] = test_probs
    test_s1_ids = test_clean["s1"]["entity_id"]
    candidate_map = test_joined.groupby("entity_id_s1")["entity_id_s2s3"].agg(
        lambda ids: ",".join(sorted(set(ids)))
    )
    candidate_out = pd.DataFrame({"source1_entity_id": test_s1_ids})
    candidate_out["candidate_entity_ids"] = candidate_out["source1_entity_id"].map(candidate_map).fillna("")
    candidate_out.to_csv(OUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
    print(f"  Saved candidate_pairs.tsv ({len(candidate_out):,} S1 rows)")

    # matching_results.tsv — only pairs predicted as matches
    match_map = test_joined.loc[test_preds == 1].groupby("entity_id_s1")["entity_id_s2s3"].agg(
        lambda ids: ",".join(sorted(set(ids)))
    )
    matches = pd.DataFrame({"source1_entity_id": test_s1_ids})
    matches["matched_entity_ids"] = matches["source1_entity_id"].map(match_map).fillna("")
    matches.to_csv(OUT_DIR / "matching_results.tsv", sep="\t", index=False)
    print(f"  Saved matching_results.tsv ({len(matches):,} S1 rows)")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n=== FINAL CHECKS ===")
    print(f"  Val F0.5 = {best_f05:.4f}  (threshold = {threshold:.3f})")
    match_rate = test_preds.mean()
    print(f"  Test match rate: {match_rate:.2%}  (sanity: should be ~5-30%)")
    print("\n=== NEXT STEPS IF F0.5 IS LOW ===")
    print("  1. Blocking recall too low? → Raise TOP_K in step4.")
    print("  2. Precision too low?       → Raise threshold; add more name features.")
    print("  3. Recall too low?          → Lower threshold; add transliteration.")
    print("  4. Imbalance issue?         → Check class_weight or try XGBoost.")
    print("  5. France/OOD?             → Validate on US↔India hold-out as proxy.")


if __name__ == "__main__":
    run()
