"""Run the UniBlocker dense branch and write the shared branch contract.

Requires a local, approved multilingual checkpoint and a UniBlocker Lightning
checkpoint trained only on challenge training records.  No external corpus,
lookup service, or remote inference is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vendor" / "Uniblocker"))
from src.models.uniblocker import UniBlocker  # noqa: E402
from src.utils.nnblocker import DenseVectorizer, FaissIndexer, NNBlocker  # noqa: E402

CLEAN = ROOT / "data/cleaned"
OUT = ROOT / "data/branch_candidates"
MODEL = ROOT / "data/models/uniblocker_amazon.ckpt"
BASE_MODEL = ROOT / "data/models/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 20


def load_model() -> UniBlocker:
    if MODEL.exists():
        model = UniBlocker.load_from_checkpoint(MODEL, map_location="cpu")
    elif BASE_MODEL.exists():
        # Valid baseline before challenge-only fine tuning has been run.
        model = UniBlocker(model_name_or_path=str(BASE_MODEL), max_length=128, augment_prob=0, pooler_type="average")
    else:
        raise FileNotFoundError(f"Place the local checkpoint at {BASE_MODEL} or {MODEL}")
    model.eval()
    return model


def run_split(split: str, model: UniBlocker) -> None:
    s1 = pd.read_parquet(CLEAN / f"{split}_s1.parquet")
    index = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet") for s in ("s2", "s3")], ignore_index=True)
    all_pairs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vectorizer = DenseVectorizer(model, model.collate_fn, device)
    for country, q in s1.groupby("country", sort=False):
        d = index[index.country.eq(country)]
        if d.empty:
            continue
        q = q.set_index("entity_id")[["name_clean", "addr_clean"]]
        d = d.set_index("entity_id")[["name_clean", "addr_clean"]]
        candidates = NNBlocker([q, d], vectorizer, FaissIndexer(index_factory="Flat"))(k=TOP_K)
        for rank, pairs in enumerate(candidates, start=1):
            all_pairs.extend((left, right, rank) for left, right in pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_pairs, columns=["entity_id_s1", "entity_id_s2s3", "rank"]).drop_duplicates(
        ["entity_id_s1", "entity_id_s2s3"]
    ).to_parquet(OUT / f"{split}_uniblocker.parquet", index=False)


def main() -> None:
    model = load_model()
    for split in (sys.argv[1:] or ["val"]):
        run_split(split, model)


if __name__ == "__main__":
    main()
