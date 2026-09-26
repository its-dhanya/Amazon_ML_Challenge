"""Union of candidate branches into one pool, keeping every branch's own rank and score.

    RUN_ID=cand_v2 python s4union.py <split> <name>=<path>[:<cap>] ...
    e.g. python s4union.py val lex=data/branch_candidates/val_sparkly.parquet:20 \
                              wa=runs/cand_v2/branches/val_Wa.parquet:3 cn=runs/cand_v2/branches/val_Cn.parquet:3

Writes runs/<RUN_ID>/candidates/candidate_pairs_<split>.parquet with entity_id_s1, entity_id_s2s3 and, per branch,
<name>_rank / <name>_score (NaN where the branch did not propose the pair). Pairs whose S1 and S2/S3 records are in
different countries are dropped. `cap` keeps only rank <= cap from that branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from runs import run_dir, safe_path

CLEAN = Path("data/cleaned")


def country_maps(split: str) -> tuple[pd.Series, pd.Series]:
    s1 = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    idx = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet", columns=["entity_id", "country"]) for s in ("s2", "s3")]).set_index("entity_id").country
    return s1, idx


def main() -> None:
    split, specs = sys.argv[1], sys.argv[2:]
    target = safe_path(run_dir() / "candidates" / f"candidate_pairs_{split}.parquet")
    frames = []
    for spec in specs:
        name, rest = spec.split("=", 1)
        path, _, cap = rest.partition(":")
        df = pd.read_parquet(path.replace("{split}", split))
        df = df[df["rank"] <= int(cap)] if cap else df
        frames.append(df.drop_duplicates(["entity_id_s1", "entity_id_s2s3"]).assign(_branch=name))
        print(f"  {name}: {len(frames[-1]):,} pairs (cap {cap or 'none'})", flush=True)
    allp = pd.concat(frames, ignore_index=True)

    # integer keys keep the outer join cheap on ~20M+ pairs
    c1, u1 = pd.factorize(allp.entity_id_s1)
    c2, u2 = pd.factorize(allp.entity_id_s2s3)
    allp["_key"] = (c1.astype(np.int64) << 32) | c2.astype(np.int64)
    keys = pd.Index(allp["_key"].unique())
    wide = pd.DataFrame(index=keys)
    for name in [s.split("=", 1)[0] for s in specs]:
        sub = allp[allp._branch == name].set_index("_key")
        wide[f"{name}_rank"] = sub["rank"].reindex(keys).to_numpy(np.float32)
        if "score" in sub:
            wide[f"{name}_score"] = sub["score"].reindex(keys).to_numpy(np.float32)
    k = wide.index.to_numpy()
    wide.insert(0, "entity_id_s2s3", u2[(k & 0xFFFFFFFF).astype(np.int64)])
    wide.insert(0, "entity_id_s1", u1[(k >> 32).astype(np.int64)])
    wide = wide.reset_index(drop=True)

    s1c, idxc = country_maps(split)
    ok = (wide.entity_id_s1.map(s1c).to_numpy() == wide.entity_id_s2s3.map(idxc).to_numpy())
    print(f"  dropped {(~ok).sum():,} cross-country pairs")
    wide = wide[ok].reset_index(drop=True)
    rank_cols = [c for c in wide if c.endswith("_rank")]
    wide["n_branches"] = wide[rank_cols].notna().sum(axis=1).astype(np.float32)
    wide.to_parquet(target, index=False)
    per = wide.groupby("entity_id_s1").size()
    print(f"{split}: {len(wide):,} pairs; S1 with candidates {per.size:,}/{len(s1c):,}; avg cands/S1 {per.mean():.1f}; "
          f"branches per pair: {wide.n_branches.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    gt_path = Path("data/processed") / f"{split}_positive_pairs.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path)[["entity_id_s1", "entity_id_s2s3"]]
        found = gt.merge(wide[["entity_id_s1", "entity_id_s2s3"]], how="inner", on=["entity_id_s1", "entity_id_s2s3"])
        print(f"{split}: pool recall {len(found) / len(gt):.2%} of {len(gt):,} true pairs")


if __name__ == "__main__":
    main()
