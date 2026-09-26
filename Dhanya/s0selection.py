"""Step 0: load challenge TSVs, split S1 groups, and expand training labels."""
from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
KEEP_COLS = ["entity_id", "business_name", "business_address", "country"]
TRAIN_DIR, TEST_DIR = Path("../dataset/student_resource/dataset/train"), Path("../dataset/student_resource/dataset/test")
OUT_DIR = Path("data/processed")


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) != KEEP_COLS:
        raise ValueError(f"{path}: expected exactly {KEEP_COLS}, found {list(df.columns)}")
    if df.entity_id.duplicated().any():
        raise ValueError(f"{path}: duplicate entity_id values are input corruption")
    expected = next((p for p in ("S1-", "S2-", "S3-") if f"source{p[1]}" in path.name), None)
    if expected and not df.entity_id.str.startswith(expected).all():
        raise ValueError(f"{path}: IDs do not all start with {expected}")
    return df.fillna("")


def stratified_split(s1: pd.DataFrame, val_frac: float = 0.10, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    val_ids: set[str] = set()
    for _, group in s1.groupby("country", sort=False):
        n = max(1, round(len(group) * val_frac))
        val_ids.update(rng.choice(group.entity_id.to_numpy(), n, replace=False))
    mask = s1.entity_id.isin(val_ids)
    return s1.loc[~mask].copy(), s1.loc[mask].copy()


def expand_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """Turn challenge list labels into positive pair labels for model training."""
    rows: list[tuple[str, str]] = []
    for s1, ids in gt[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False):
        for candidate in filter(None, ids.split(",")):
            if not candidate.startswith(("S2-", "S3-")):
                raise ValueError(f"ground truth has invalid candidate ID: {candidate}")
            rows.append((s1, candidate))
    return pd.DataFrame(rows, columns=["entity_id_s1", "entity_id_s2s3"])


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_s1 = load_source(TRAIN_DIR / "train_source1.tsv")
    train_s2 = load_source(TRAIN_DIR / "train_source2.tsv")
    train_s3 = load_source(TRAIN_DIR / "train_source3.tsv")
    test_s1 = load_source(TEST_DIR / "test_source1.tsv")
    test_s2 = load_source(TEST_DIR / "test_source2.tsv")
    test_s3 = load_source(TEST_DIR / "test_source3.tsv")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    if list(gt.columns) != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError("unexpected training ground-truth schema")

    train_s1_split, val_s1 = stratified_split(train_s1)
    positives = expand_ground_truth(gt)
    for split, s1 in (("train", train_s1_split), ("val", val_s1)):
        s1.to_parquet(OUT_DIR / f"{split}_s1.parquet", index=False)
        positives[positives.entity_id_s1.isin(s1.entity_id)].to_parquet(OUT_DIR / f"{split}_positive_pairs.parquet", index=False)

    # S2/S3 are retrieval corpora. Reuse their complete training pool for both S1 splits.
    for split in ("train", "val"):
        train_s2.to_parquet(OUT_DIR / f"{split}_s2.parquet", index=False)
        train_s3.to_parquet(OUT_DIR / f"{split}_s3.parquet", index=False)
    for src, df in (("s1", test_s1), ("s2", test_s2), ("s3", test_s3)):
        df.to_parquet(OUT_DIR / f"test_{src}.parquet", index=False)
    log.info("saved train/val/test inputs; validation S1=%d", len(val_s1))


if __name__ == "__main__":
    run()
