"""Lexical branch via sparse BM25 (replaces s4duckdb.py).

Per country: build a BM25-weighted sparse document-term matrix over the S2+S3
index, then score batches of S1 queries with a sparse product and keep the
top-K per query that score >= REL_THRESHOLD * top score. Very common tokens
(df > MAX_DF of the country's docs) are dropped: they carry no blocking signal
and their postings dominate cost/memory.

Writes data/branch_candidates/{split}_sparkly.parquet with the shared branch
contract (entity_id_s1, entity_id_s2s3, rank) plus score.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

CLEAN = Path("data/cleaned")
OUT = Path("data/branch_candidates")
PROCESSED = Path("data/processed")
TOP_K = 50
REL_THRESHOLD = 0.5
MAX_DF = 0.01
K1, B = 1.2, 0.75
BATCH = 500


def build_index(texts: pd.Series):
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    tf = vec.fit_transform(texts).tocsr()
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf[df > MAX_DF * n_docs] = 0.0

    dl = np.asarray(tf.sum(axis=1)).ravel()
    norm = K1 * (1 - B + B * dl / dl.mean())
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    tf.data = idf[tf.indices] * tf.data * (K1 + 1) / (tf.data + norm[rows].astype(np.float32))
    tf.eliminate_zeros()
    return vec, tf.T.tocsr().astype(np.float32)


def score_country(q_ids, q_texts, d_ids, d_texts):
    vec, dt = build_index(d_texts)
    q = vec.transform(q_texts).tocsr()
    q.data[:] = 1.0
    out_q, out_d, out_rank, out_score = [], [], [], []
    for start in tqdm(range(0, q.shape[0], BATCH), desc="  scoring", unit="batch"):
        s = (q[start:start + BATCH] @ dt).tocsr()
        for i in range(s.shape[0]):
            lo, hi = s.indptr[i], s.indptr[i + 1]
            if lo == hi:
                continue
            sc, idx = s.data[lo:hi], s.indices[lo:hi]
            if len(sc) > TOP_K:
                top = np.argpartition(-sc, TOP_K - 1)[:TOP_K]
                sc, idx = sc[top], idx[top]
            order = np.argsort(-sc)
            sc, idx = sc[order], idx[order]
            keep = sc >= REL_THRESHOLD * sc[0]
            sc, idx = sc[keep], idx[keep]
            out_q.append(np.full(len(sc), start + i))
            out_d.append(idx)
            out_rank.append(np.arange(1, len(sc) + 1))
            out_score.append(sc)
    if not out_q:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_s2s3", "rank", "score"])
    qi, di = np.concatenate(out_q), np.concatenate(out_d)
    return pd.DataFrame({
        "entity_id_s1": q_ids[qi],
        "entity_id_s2s3": d_ids[di],
        "rank": np.concatenate(out_rank),
        "score": np.concatenate(out_score),
    })


def text_of(df: pd.DataFrame) -> pd.Series:
    return df["name_clean"].fillna("") + " " + df["addr_clean"].fillna("")


def load_country(split: str, country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["entity_id", "country", "name_clean", "addr_clean"]
    flt = [("country", "==", country)]
    q = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=cols, filters=flt)
    d = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet", columns=cols, filters=flt) for s in ("s2", "s3")], ignore_index=True)
    return q, d


def run_part(split: str, country: str, part: int, nparts: int) -> None:
    """Score one slice of one country's queries; write it to parts/ for later merging."""
    t0 = time.time()
    q, d = load_country(split, country)
    q = q.iloc[np.array_split(np.arange(len(q)), nparts)[part]]
    print(f"[{split}] {country} part {part + 1}/{nparts}: index {len(d):,}, queries {len(q):,}", flush=True)
    pairs = score_country(q.entity_id.to_numpy(), text_of(q), d.entity_id.to_numpy(), text_of(d))
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT / "parts" / f"{split}_{country}_{part}of{nparts}.parquet", index=False)
    print(f"[{split}] {country} part {part + 1}/{nparts}: {len(pairs):,} pairs, {pairs.entity_id_s1.nunique():,}/{len(q):,} queries with candidates, {time.time() - t0:.0f}s", flush=True)


def merge(split: str) -> pd.DataFrame:
    files = sorted((OUT / "parts").glob(f"{split}_*.parquet"))
    pairs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pairs.to_parquet(OUT / f"{split}_sparkly.parquet", index=False)
    n_q = len(pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=["entity_id"]))
    print(f"[{split}] merged {len(files)} parts: {len(pairs):,} pairs for {pairs.entity_id_s1.nunique():,}/{n_q:,} queries")
    return pairs


def run_split(split: str) -> pd.DataFrame:
    s1 = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=["country"])
    for country in s1.country.unique():
        run_part(split, country, 0, 1)
    return merge(split)


def evaluate(split: str, pairs: pd.DataFrame) -> None:
    gt_path = PROCESSED / f"{split}_positive_pairs.parquet"
    if not gt_path.exists():
        return
    gt = pd.read_parquet(gt_path)[["entity_id_s1", "entity_id_s2s3"]]
    m = gt.merge(pairs[["entity_id_s1", "entity_id_s2s3", "rank"]], how="left", on=["entity_id_s1", "entity_id_s2s3"])
    print(f"[{split}] lexical recall (true pairs found, {len(gt):,} total)")
    for k in (5, 10, 20, 30, TOP_K):
        print(f"   rank<={k:<3} {(m['rank'] <= k).mean():.2%}")
    print(f"   avg candidates/query: {len(pairs) / pairs.entity_id_s1.nunique():.1f}")


def main() -> None:
    """  s4sparse.py val                          whole split, one process
         s4sparse.py test India 2 3               part 2 of 3 of one country (parallel runs)
         s4sparse.py test merge                   merge all parts into {split}_sparkly.parquet"""
    args = sys.argv[1:] or ["val"]
    split = args[0]
    if len(args) == 1:
        evaluate(split, run_split(split))
    elif args[1] == "merge":
        evaluate(split, merge(split))
    else:
        run_part(split, args[1], int(args[2]) - 1, int(args[3]))


if __name__ == "__main__":
    main()
