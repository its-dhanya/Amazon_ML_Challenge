"""Step 5: LightGBM pair classifier + one-to-one assignment + F0.5 threshold.

    python s5lgbm.py fit    # fit on 80% of val (split by S1 entity), tune on the other 20%
    python s5lgbm.py test   # score test candidates, write output/*.tsv

Country is deliberately not a feature (test has an unseen country). A missing
field is NaN, never a match. Each S2/S3 record matches at most one S1 entity
(true in train/val ground truth), so after scoring every record is kept only
for the S1 entity that scores it highest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from runs import RUNS, run_dir, run_id, safe_path

CLEAN = Path("data/cleaned")
CAND = Path("data/candidates")
BRANCH = Path("data/branch_candidates")
GT = Path("data/processed")
CACHE = Path("data/cache")  # feature caches are keyed by pool run + a hash of FEATS and store the pair ids (checked on load)
POOL_RUN = os.environ.get("POOL_RUN")  # run whose candidates/ holds the pool; unset = legacy data/candidates
FEATSET = os.environ.get("FEATSET", "v1")
V2 = FEATSET in ("v2", "v3")  # v2 adds multi-branch and cross-script/glued-name features
V3 = FEATSET == "v3"          # v3 adds the dense (bi-encoder) branch score features
REC = ["entity_id", "name_clean", "addr_clean", "pin_zip", "state_code"]
CHUNK = 1_000_000
UNSEEN_THRESHOLD = 0.75
NUM = re.compile(r"\d+")

FEATS_V1 = [
    "name_jw", "name_tsort", "name_tset", "name_ratio", "name_exact", "name_len_ratio",
    "addr_jw", "addr_tsort", "addr_tset", "addr_num_jacc", "pin_eq", "state_eq",
    "lex_rank", "lex_score", "lex_rel", "dense_rank",
    "n_cands", "name_jw_gap", "name_jw_rank", "rec_n_s1", "rec_lex_rank", "rec_name_gap",
]
FEATS_V2_EXTRA = [
    "wa_rank", "wa_score", "wa_rel", "cn_rank", "cn_score", "cn_rel", "n_branches",
    "name_desp_ratio", "name_desp_partial", "name_skel_ratio", "addr_i_empty", "is_s3",
]
FEATS_V3_EXTRA = ["dense_score", "dense_rel"]   # dense_rank is already in FEATS_V1 (it was the UniBlocker slot)
FEATS = FEATS_V1 + (FEATS_V2_EXTRA if V2 else []) + (FEATS_V3_EXTRA if V3 else [])
BRANCH_COLS = ["lex_rank", "lex_score", "dense_rank", "dense_score", "wa_rank", "wa_score", "cn_rank", "cn_score"]


# ── pair table ────────────────────────────────────────────────────────────────

def build_pairs(split: str) -> pd.DataFrame:
    pool = (RUNS / POOL_RUN / "candidates" if POOL_RUN else CAND) / f"candidate_pairs_{split}.parquet"
    pairs = pd.read_parquet(pool)
    print(f"pool: {pool} ({len(pairs):,} pairs)")
    if "lex_rank" not in pairs.columns:  # older union file without the branch columns: join them in
        lex = pd.read_parquet(BRANCH / f"{split}_sparkly.parquet", columns=["entity_id_s1", "entity_id_s2s3", "rank", "score"])
        pairs = pairs.merge(lex.rename(columns={"rank": "lex_rank", "score": "lex_score"}), how="left", on=["entity_id_s1", "entity_id_s2s3"])
        dense = BRANCH / f"{split}_uniblocker.parquet"
        if dense.exists():
            d = pd.read_parquet(dense, columns=["entity_id_s1", "entity_id_s2s3", "rank"]).rename(columns={"rank": "dense_rank"})
            pairs = pairs.merge(d, how="left", on=["entity_id_s1", "entity_id_s2s3"])
        else:
            pairs["dense_rank"] = np.nan
    for c in BRANCH_COLS:
        if c not in pairs.columns:
            pairs[c] = np.nan
    if "n_branches" not in pairs.columns:
        pairs["n_branches"] = pairs[["lex_rank", "wa_rank", "cn_rank", "dense_rank"]].notna().sum(axis=1).astype(np.float32)
    for b in ("lex", "wa", "cn", "dense"):
        pairs[f"{b}_rel"] = pairs[f"{b}_score"] / pairs.groupby("entity_id_s1")[f"{b}_score"].transform("max")
    return pairs


def load_records(pairs: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=REC)
    s1 = s1[s1.entity_id.isin(pairs.entity_id_s1.unique())].set_index("entity_id")
    ids = set(pairs.entity_id_s2s3.unique())
    idx = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet", columns=REC) for s in ("s2", "s3")], ignore_index=True)
    idx = idx[idx.entity_id.isin(ids)].set_index("entity_id")
    if V2:  # romanized/despaced and phonetic-skeleton names (same normalization as the s4ngram branches)
        from s4ngram import prep_despaced, prep_skeleton  # only the v2 feature set needs anyascii/s4ngram
        for df in (s1, idx):
            nm = df.name_clean.fillna("")
            df["name_desp"], df["name_skel"] = nm.map(prep_despaced), nm.map(prep_skeleton)
    return s1, idx


def attach_records(chunk: pd.DataFrame, s1: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    for c in REC[1:] + (["name_desp", "name_skel"] if V2 else []):
        chunk[c + "_s"] = s1[c].reindex(chunk.entity_id_s1).to_numpy()
        chunk[c + "_i"] = idx[c].reindex(chunk.entity_id_s2s3).to_numpy()
    return chunk


# ── features ──────────────────────────────────────────────────────────────────

def _missing(x: pd.Series) -> np.ndarray:
    return (x.isna() | x.isin(["", "missing"])).to_numpy()


def _sim(a: pd.Series, b: pd.Series, scorer, scale: float = 1.0) -> np.ndarray:
    miss = _missing(a) | _missing(b)
    out = process.cpdist(a.fillna("").tolist(), b.fillna("").tolist(), scorer=scorer, dtype=np.float32, workers=-1)
    out = out.astype(np.float32) / scale
    out[miss] = np.nan
    return out


def _nums(x: str) -> set:
    n = NUM.findall(x)
    return {t.lstrip("0") or "0" for t in n} if V2 else set(n)  # v2: "0250" == "250"


def _num_jacc(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = np.full(len(a), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(a.fillna("").tolist(), b.fillna("").tolist())):
        sx, sy = _nums(x), _nums(y)
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out


def _eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = (a.to_numpy() == b.to_numpy()).astype(np.float32)
    out[_missing(a) | _missing(b)] = np.nan
    return out


def base_features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    ns, ni, as_, ai = p.name_clean_s, p.name_clean_i, p.addr_clean_s, p.addr_clean_i
    f["name_jw"] = _sim(ns, ni, JaroWinkler.similarity)
    f["name_tsort"] = _sim(ns, ni, fuzz.token_sort_ratio, 100)
    f["name_tset"] = _sim(ns, ni, fuzz.token_set_ratio, 100)
    f["name_ratio"] = _sim(ns, ni, fuzz.ratio, 100)
    f["name_exact"] = _eq(ns, ni)
    ls, li = ns.fillna("").str.len().to_numpy(), ni.fillna("").str.len().to_numpy()
    f["name_len_ratio"] = np.minimum(ls, li) / np.maximum(np.maximum(ls, li), 1)
    f["addr_jw"] = _sim(as_, ai, JaroWinkler.similarity)
    f["addr_tsort"] = _sim(as_, ai, fuzz.token_sort_ratio, 100)
    f["addr_tset"] = _sim(as_, ai, fuzz.token_set_ratio, 100)
    f["addr_num_jacc"] = _num_jacc(as_, ai)
    f["pin_eq"] = _eq(p.pin_zip_s, p.pin_zip_i)
    f["state_eq"] = _eq(p.state_code_s, p.state_code_i)
    if V2:
        f["name_desp_ratio"] = _sim(p.name_desp_s, p.name_desp_i, fuzz.ratio, 100)
        f["name_desp_partial"] = _sim(p.name_desp_s, p.name_desp_i, fuzz.partial_ratio, 100)
        f["name_skel_ratio"] = _sim(p.name_skel_s, p.name_skel_i, fuzz.ratio, 100)
        f["addr_i_empty"] = _missing(ai).astype(np.float32)
        f["is_s3"] = p.entity_id_s2s3.str.startswith("S3").to_numpy().astype(np.float32)
    return f


def build_features(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per pair, columns FEATS, index aligned with pairs."""
    s1, idx = load_records(pairs, split)
    parts = []
    for i in range(0, len(pairs), CHUNK):
        parts.append(base_features(attach_records(pairs.iloc[i:i + CHUNK], s1, idx)))
        print(f"  features {min(i + CHUNK, len(pairs)):,}/{len(pairs):,}", flush=True)
    f = pd.concat(parts)
    for c in ("lex_rank", "lex_score", "lex_rel", "dense_rank") + (("wa_rank", "wa_score", "wa_rel", "cn_rank", "cn_score", "cn_rel", "n_branches") if V2 else ()):
        f[c] = pairs[c].to_numpy(np.float32)
    if V3:
        for c in ("dense_score", "dense_rel"):
            f[c] = pairs[c].to_numpy(np.float32)
    g_s1 = pairs.entity_id_s1.to_numpy()
    g_rec = pairs.entity_id_s2s3.to_numpy()
    f["_s1"], f["_rec"] = pd.factorize(g_s1)[0], pd.factorize(g_rec)[0]
    by_s1, by_rec = f.groupby("_s1"), f.groupby("_rec")
    f["n_cands"] = by_s1.name_jw.transform("size").astype(np.float32)
    f["name_jw_gap"] = f.name_jw - by_s1.name_jw.transform("max")
    f["name_jw_rank"] = by_s1.name_jw.rank(ascending=False, method="min")
    f["rec_n_s1"] = by_rec.name_jw.transform("size").astype(np.float32)
    f["rec_lex_rank"] = by_rec.lex_score.rank(ascending=False, method="min")
    f["rec_name_gap"] = f.name_jw - by_rec.name_jw.transform("max")
    return f


# ── scoring / selection ───────────────────────────────────────────────────────

def one_to_one(pairs: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    """Boolean mask keeping, per S2/S3 record, only its highest-probability S1 entity."""
    order = np.argsort(-prob, kind="stable")
    first = pd.Series(pairs.entity_id_s2s3.to_numpy()[order]).duplicated().to_numpy()
    keep = np.zeros(len(pairs), bool)
    keep[order[~first]] = True
    return keep


def macro_f05(sel: np.ndarray, s1_code: np.ndarray, label: np.ndarray, n_truth: np.ndarray) -> float:
    n = len(n_truth)
    tp = np.bincount(s1_code[sel & (label == 1)], minlength=n)
    npred = np.bincount(s1_code[sel], minlength=n)
    prec = np.divide(tp, npred, out=np.zeros(n), where=npred > 0)
    rec = np.divide(tp, n_truth, out=np.zeros(n), where=n_truth > 0)
    f = np.divide(1.25 * prec * rec, 0.25 * prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    f[(npred == 0) & (n_truth == 0)] = 1.0
    return float(f.mean())


def sweep(prob, pairs, label, s1_code, n_truth, use_o2o):
    keep = one_to_one(pairs, prob) if use_o2o else np.ones(len(pairs), bool)
    best = (0.0, 0.5)
    for t in np.linspace(0.2, 0.95, 76):
        f = macro_f05((prob >= t) & keep, s1_code, label, n_truth)
        if f > best[0]:
            best = (f, t)
    return best


def split_roles(entity_id_s1: pd.Series) -> np.ndarray:
    """0 = fit, 1 = early-stop / blend fitting, 2 = holdout (half B), by a stable hash of the S1 id."""
    h = pd.util.hash_pandas_object(entity_id_s1, index=False).to_numpy()
    parity, u = h % 2, ((h // 2) % 1000) / 1000
    return np.where((parity == 1) & (u < 0.4), 2, np.where((parity == 0) & (u < 0.16), 1, 0))


# ── commands ──────────────────────────────────────────────────────────────────

def fit() -> None:
    rd = run_dir()  # runs/<RUN_ID>/: model, metrics and logs of this run only
    model_path = safe_path(rd / "model.pkl")  # fail early instead of overwriting a finished run
    pairs = build_pairs("val")
    gt = pd.read_parquet(GT / "val_positive_pairs.parquet")[["entity_id_s1", "entity_id_s2s3"]].assign(label=1)
    pairs = pairs.merge(gt, how="left", on=["entity_id_s1", "entity_id_s2s3"])
    pairs["label"] = pairs.label.fillna(0).astype(int)
    print(f"pairs {len(pairs):,}, positives {pairs.label.mean():.1%}")

    cache = CACHE / f"val_features_{POOL_RUN + '_' if POOL_RUN else ''}{hashlib.md5(','.join(FEATS).encode()).hexdigest()[:8]}.parquet"
    f = None
    if cache.exists():
        f = pd.read_parquet(cache)
        aligned = ({"entity_id_s1", "entity_id_s2s3"} <= set(f.columns) and len(f) == len(pairs)
                   and (f.entity_id_s1.to_numpy() == pairs.entity_id_s1.to_numpy()).all()
                   and (f.entity_id_s2s3.to_numpy() == pairs.entity_id_s2s3.to_numpy()).all())
        if not aligned:  # never trust a cache whose rows are not the same pairs in the same order
            print(f"cache {cache.name} does not match the pool; recomputing")
            f = None
    if f is None:
        f = build_features(pairs, "val")
        f["entity_id_s1"], f["entity_id_s2s3"] = pairs.entity_id_s1.to_numpy(), pairs.entity_id_s2s3.to_numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(cache)
    X, y = f[FEATS], pairs.label.to_numpy()
    s1_code = f["_s1"].to_numpy()

    # split by a stable hash of the S1 id (independent of row order): the 20% holdout lies entirely in half B
    # (odd hash parity), the half never inspected during the error analysis; 8% of entities early-stop, 72% fit.
    role = split_roles(pairs.entity_id_s1)
    print(f"split (pairs): fit {(role == 0).sum():,}, early-stop {(role == 1).sum():,}, holdout {(role == 2).sum():,}")
    m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.1, num_leaves=127, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, n_jobs=-1, verbose=-1)
    m.fit(X[role == 0], y[role == 0], eval_set=[(X[role == 1], y[role == 1])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"best iteration {m.best_iteration_}")
    print(pd.Series(m.booster_.feature_importance("gain"), FEATS).sort_values(ascending=False).round(0).to_string())

    # holdout evaluation: truth counts over ALL ground-truth pairs of the holdout entities
    ho = role == 2
    ho_pairs = pairs[ho].reset_index(drop=True)
    ho_ids = ho_pairs.entity_id_s1.unique()
    code = pd.Series(np.arange(len(ho_ids)), ho_ids)
    n_truth = gt[gt.entity_id_s1.isin(ho_ids)].groupby("entity_id_s1").size().reindex(ho_ids).fillna(0).to_numpy()
    s1c = code.reindex(ho_pairs.entity_id_s1).to_numpy().astype(int)
    prob = m.predict_proba(X[ho])[:, 1]
    lbl = y[ho]
    print(f"\nholdout: {len(ho_ids):,} S1 entities, {len(ho_pairs):,} pairs")
    for name, o2o in (("threshold only", False), ("threshold + one-to-one", True)):
        f05, t = sweep(prob, ho_pairs, lbl, s1c, n_truth, o2o)
        keep = one_to_one(ho_pairs, prob) if o2o else np.ones(len(ho_pairs), bool)
        sel = (prob >= t) & keep
        tp = (sel & (lbl == 1)).sum()
        print(f"  {name:24s} F0.5={f05:.4f} @ t={t:.2f}   precision={tp / max(sel.sum(), 1):.3f}  "
              f"recall={tp / n_truth.sum():.3f}  predicted/entity={sel.sum() / len(ho_ids):.2f}")
    f05, t = sweep(prob, ho_pairs, lbl, s1c, n_truth, True)
    joblib.dump({"model": m, "threshold": t, "feats": FEATS}, model_path)
    (rd / "metrics.json").write_text(json.dumps({
        "holdout_f05_one_to_one": round(f05, 4), "threshold": round(float(t), 3), "best_iteration": int(m.best_iteration_),
        "pool_run": POOL_RUN, "featset": "v2" if V2 else "v1", "n_pairs": int(len(pairs)), "positive_rate": round(float(pairs.label.mean()), 4), "features": FEATS}, indent=2))
    print(f"\nsaved {model_path} (threshold {t:.2f}); metrics in {rd / 'metrics.json'}")


def write_tsv(s1_ids: pd.Series, pairs: pd.DataFrame, col: str, path: Path) -> None:
    ids = pairs.groupby("entity_id_s1").entity_id_s2s3.agg(lambda x: ",".join(sorted(set(x))))
    out = pd.DataFrame({"source1_entity_id": s1_ids.to_numpy()})
    out[col] = out.source1_entity_id.map(ids).fillna("")
    out.to_csv(path, sep="\t", index=False)
    print(f"wrote {path} ({len(out):,} rows, {(out[col] != '').mean():.1%} non-empty)")


def predict_test() -> None:
    out_dir = run_dir() / "output"  # this run's own submission files; never touches other runs' output
    targets = [safe_path(out_dir / n) for n in ("candidate_pairs.tsv", "matching_results.tsv")]
    saved = joblib.load(run_dir(os.environ.get("MODEL_RUN") or run_id(), create=False) / "model.pkl")  # MODEL_RUN = run that trained the model
    if saved["feats"] != FEATS:
        raise SystemExit(f"feature set mismatch: model has {len(saved['feats'])} features, FEATSET gives {len(FEATS)}")
    pairs = build_pairs("test")
    # Score one country at a time (halves peak memory). Every group feature (per S1 entity and per S2/S3 record)
    # stays inside one country, so the probabilities are identical to scoring all pairs at once.
    ctry_of = pd.read_parquet(CLEAN / "test_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    pair_country = ctry_of.reindex(pairs.entity_id_s1).to_numpy()
    prob = np.empty(len(pairs), np.float32)
    for c in pd.unique(pair_country):
        m = pair_country == c
        print(f"[{c}] {int(m.sum()):,} pairs", flush=True)
        f = build_features(pairs[m].reset_index(drop=True), "test")
        prob[m] = saved["model"].predict_proba(f[saved["feats"]])[:, 1]
        del f
    keep = one_to_one(pairs, prob)
    scores_path = safe_path(run_dir() / "scores_test.parquet")  # per-pair probability, input of the XLM-R second stage
    pd.DataFrame({"entity_id_s1": pairs.entity_id_s1.to_numpy(), "entity_id_s2s3": pairs.entity_id_s2s3.to_numpy(),
                  "prob": prob.astype(np.float32)}).to_parquet(scores_path, index=False)
    print(f"saved {scores_path}")
    s1 = pd.read_parquet(CLEAN / "test_s1.parquet", columns=["entity_id", "country"])
    # Countries never seen in training get a stricter threshold: the country-held-out check showed
    # the best cutoff on an unseen country was higher (0.74-0.75) than the in-country one.
    seen = set(pd.read_parquet(CLEAN / "val_s1.parquet", columns=["country"]).country.unique())
    country = s1.set_index("entity_id").country.reindex(pairs.entity_id_s1).to_numpy()
    thr = np.where(np.isin(country, list(seen)), saved["threshold"], max(saved["threshold"], UNSEEN_THRESHOLD))
    print(f"seen countries {sorted(seen)}; unseen in test: {sorted(set(s1.country) - seen)}; thresholds {sorted(set(thr.round(2)))}")
    sel = (prob >= thr) & keep
    print(f"test pairs {len(pairs):,}; predicted matches {sel.sum():,} ({sel.sum() / pairs.entity_id_s1.nunique():.2f} per S1)")
    write_tsv(s1.entity_id, pairs, "candidate_entity_ids", targets[0])
    write_tsv(s1.entity_id, pairs[sel], "matched_entity_ids", targets[1])
    c = s1.set_index("entity_id").country.reindex(pairs.entity_id_s1[sel]).value_counts()
    print("matches by country:\n" + c.to_string())


if __name__ == "__main__":
    {"fit": fit, "test": predict_test}[sys.argv[1] if len(sys.argv) > 1 else "fit"]()
