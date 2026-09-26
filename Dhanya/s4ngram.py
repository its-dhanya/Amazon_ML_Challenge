"""Independent BM25 representations for candidate generation (extra branches next to the joint word BM25).

Each representation indexes ONE field of the S2/S3 records of a country with BM25 weights and scores S1
queries with a sparse product, exactly like s4sparse.py, keeping the top-K above a relative score
threshold. Representations:
  Wn  name-only word BM25                    (an empty address cannot sink a name match)
  Wa  address-only word BM25                 (leading zeros stripped from numbers)
  Cn  name-only despaced character 4-grams   (glued/domain names, hyphens, typos, extra/dropped words)
  Pn  name-only phonetic-skeleton 3-grams    (romanize any script, drop vowels: cross-script + typos)

    python s4ngram.py gate India Wn Wa Cn Pn     # half-A sample (~20K queries) -> data/gate/
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anyascii import anyascii
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from runs import run_dir, safe_path

CLEAN = Path("data/cleaned")
K1, B = 1.2, 0.75
REL_THRESHOLD = 0.5
_SKEL = (("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"),
         ("sh", "s"), ("w", "v"), ("z", "j"), ("c", "k"), ("q", "k"))


# ── text preparation (generic script/spelling normalization, nothing hand-listed per name) ──────
def prep_words(x: str) -> str:
    return x


def prep_addr(x: str) -> str:
    return re.sub(r"\b0+(?=\d)", "", x)


def prep_despaced(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", anyascii(x).lower())


def prep_skeleton(x: str) -> str:
    x = anyascii(x).lower()
    for a, b in _SKEL:
        x = x.replace(a, b)
    x = re.sub(r"[aeiouy]", "", re.sub(r"[^a-z0-9]", "", x))
    return re.sub(r"(.)\1+", r"\1", x)


WORD = dict(token_pattern=r"(?u)\b\w+\b")
REPS = {
    "Wn": dict(field="name_clean", prep=prep_words, vec=WORD, max_df=0.01, top_k=10, batch=500),
    "Wa": dict(field="addr_clean", prep=prep_addr, vec=WORD, max_df=0.01, top_k=10, batch=500),
    "Cn": dict(field="name_clean", prep=prep_despaced, vec=dict(analyzer="char", ngram_range=(4, 4)), max_df=0.005, top_k=10, batch=250),
    "Pn": dict(field="name_clean", prep=prep_skeleton, vec=dict(analyzer="char", ngram_range=(3, 3)), max_df=0.005, top_k=10, batch=250),
}


# ── BM25 machinery (same weighting as s4sparse.py, parameterized) ───────────────────────────────
def build_index(texts: pd.Series, vec_kwargs: dict, max_df: float):
    vec = CountVectorizer(dtype=np.float32, **vec_kwargs)
    tf = vec.fit_transform(texts).tocsr()
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf[df > max_df * n_docs] = 0.0
    dl = np.asarray(tf.sum(axis=1)).ravel()
    norm = K1 * (1 - B + B * dl / max(dl.mean(), 1e-9))
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    tf.data = idf[tf.indices] * tf.data * (K1 + 1) / (tf.data + norm[rows].astype(np.float32))
    tf.eliminate_zeros()
    return vec, tf.T.tocsr().astype(np.float32)


def score_queries(q_ids, q_texts, d_ids, d_texts, cfg) -> pd.DataFrame:
    vec, dt = build_index(d_texts, cfg["vec"], cfg["max_df"])
    q = vec.transform(q_texts).tocsr()
    q.data[:] = 1.0
    top_k, batch = cfg["top_k"], cfg["batch"]
    out_q, out_d, out_r, out_s = [], [], [], []
    for start in tqdm(range(0, q.shape[0], batch), desc="  scoring", unit="batch", mininterval=30):
        s = (q[start:start + batch] @ dt).tocsr()
        for i in range(s.shape[0]):
            lo, hi = s.indptr[i], s.indptr[i + 1]
            if lo == hi:
                continue
            sc, idx = s.data[lo:hi], s.indices[lo:hi]
            if len(sc) > top_k:
                top = np.argpartition(-sc, top_k - 1)[:top_k]
                sc, idx = sc[top], idx[top]
            order = np.argsort(-sc)
            sc, idx = sc[order], idx[order]
            keep = sc >= REL_THRESHOLD * sc[0]
            sc, idx = sc[keep], idx[keep]
            out_q.append(np.full(len(sc), start + i)); out_d.append(idx)
            out_r.append(np.arange(1, len(sc) + 1)); out_s.append(sc)
    if not out_q:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_s2s3", "rank", "score"])
    qi, di = np.concatenate(out_q), np.concatenate(out_d)
    return pd.DataFrame({"entity_id_s1": q_ids[qi], "entity_id_s2s3": d_ids[di],
                         "rank": np.concatenate(out_r), "score": np.concatenate(out_s)})


def run_rep(rep: str, q: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    cfg = rep_config(rep)
    t0 = time.time()
    qt = q[cfg["field"]].fillna("").map(cfg["prep"])
    dt_ = d[cfg["field"]].fillna("").replace("missing", "").map(cfg["prep"])
    print(f"  [{rep}] prepared text in {time.time() - t0:.0f}s", flush=True)
    return score_queries(q.entity_id.to_numpy(), qt, d.entity_id.to_numpy(), dt_, cfg)


# ── half-A gate sample ──────────────────────────────────────────────────────────────────────────
def gate(country: str, reps: list[str], n: int = 20000) -> None:
    out = run_dir() / "gate"  # runs/<RUN_ID>/gate/: each run keeps its own results
    targets = {rep: safe_path(out / f"{country}_{rep}.parquet") for rep in reps}  # fail early, never overwrite
    cols = ["entity_id", "country", "name_clean", "addr_clean"]
    flt = [("country", "==", country)]
    s1 = pd.read_parquet(CLEAN / "val_s1.parquet", columns=cols, filters=flt)
    gt = pd.read_parquet("data/processed/val_positive_pairs.parquet")[["entity_id_s1", "entity_id_s2s3"]]
    parity = 0 if os.environ.get("HALF", "A") == "A" else 1  # HALF=B: the half that was never inspected
    in_half = pd.util.hash_pandas_object(s1.entity_id, index=False).to_numpy() % 2 == parity
    pool = s1[in_half & s1.entity_id.isin(gt.entity_id_s1)]
    q = pool.sample(min(n, len(pool)), random_state=0)
    q[["entity_id"]].to_parquet(out / f"{country}_queries.parquet", index=False)
    d = pd.concat([pd.read_parquet(CLEAN / f"val_{s}.parquet", columns=cols, filters=flt) for s in ("s2", "s3")], ignore_index=True)
    print(f"[gate] {country}: {len(q):,} half-{os.environ.get('HALF', 'A')} queries, index {len(d):,}", flush=True)
    for rep in reps:
        t0 = time.time()
        pairs = run_rep(rep, q, d)
        pairs.to_parquet(targets[rep], index=False)
        print(f"[gate] {country} {rep}: {len(pairs):,} pairs ({len(pairs) / len(q):.1f}/query) in {time.time() - t0:.0f}s", flush=True)


# ── full-split runs (parallel slices, like s4sparse.py) ────────────────────────────────────────
def run_part(split: str, country: str, part: int, nparts: int, reps: list[str]) -> None:
    """Score slice `part` (0-based) of `nparts` of one country's S1 queries with each representation."""
    out = run_dir() / "branches"
    targets = {rep: safe_path(out / f"{split}_{country}_{rep}_{part + 1}of{nparts}.parquet") for rep in reps}
    cols = ["entity_id", "country", "name_clean", "addr_clean"]
    flt = [("country", "==", country)]
    q = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=cols, filters=flt)
    q = q.iloc[np.array_split(np.arange(len(q)), nparts)[part]]
    d = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet", columns=cols, filters=flt) for s in ("s2", "s3")], ignore_index=True)
    print(f"[{split}] {country} part {part + 1}/{nparts}: {len(q):,} queries, index {len(d):,}", flush=True)
    for rep in reps:
        t0 = time.time()
        pairs = run_rep(rep, q, d)
        pairs.to_parquet(targets[rep], index=False)
        print(f"[{split}] {country} {rep} part {part + 1}/{nparts}: {len(pairs):,} pairs, {pairs.entity_id_s1.nunique():,}/{len(q):,} queries "
              f"with candidates, {time.time() - t0:.0f}s", flush=True)


def merge(split: str, reps: list[str]) -> None:
    out = run_dir() / "branches"
    n_q = len(pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=["entity_id"]))
    for rep in reps:
        files = sorted(out.glob(f"{split}_*_{rep}_*of*.parquet"))
        pairs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        pairs.to_parquet(safe_path(out / f"{split}_{rep}.parquet"), index=False)
        print(f"[{split}] {rep}: merged {len(files)} parts, {len(pairs):,} pairs for {pairs.entity_id_s1.nunique():,}/{n_q:,} queries", flush=True)


def rep_config(rep: str) -> dict:
    cfg = dict(REPS[rep])
    if os.environ.get("TOPK"):  # e.g. TOPK=5: keep more ranks so the cap (3 or 5) can be chosen later
        cfg["top_k"] = int(os.environ["TOPK"])
    return cfg


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gate":
        gate(sys.argv[2], sys.argv[3:])
    elif cmd == "run":   # run <split> <country> <part 1-based> <nparts> <rep...>
        run_part(sys.argv[2], sys.argv[3], int(sys.argv[4]) - 1, int(sys.argv[5]), sys.argv[6:])
    elif cmd == "merge":  # merge <split> <rep...>
        merge(sys.argv[2], sys.argv[3:])
