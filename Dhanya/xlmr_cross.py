"""XLM-R cross-encoder second stage for borderline pairs (val experiment).

The LightGBM classifier (v2) scores every pool pair; pairs whose probability lies in the uncertain zone
[LO, HI] are re-scored by a fine-tuned XLM-R base cross-encoder (MIT license, reads any script). The two
scores are blended with a logistic regression fitted on the early-stop entities (never on the holdout).

    RUN_ID=xlmr_v1 POOL_RUN=cand_v2_cap5 FEATSET=v2 CLF_RUN=fit_v2_cap5 python xlmr_cross.py train
    RUN_ID=xlmr_v1 POOL_RUN=cand_v2_cap5 FEATSET=v2 CLF_RUN=fit_v2_cap5 python xlmr_cross.py eval
"""
from __future__ import annotations

import hashlib
import math
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import s5lgbm as S
from runs import RUNS, run_dir, run_id

CLEAN = Path("data/cleaned")
MODEL_DIR = Path("data/models/xlm-roberta-base")
KEYS = ["entity_id_s1", "entity_id_s2s3"]
LO, HI = 0.10, 0.95            # uncertain zone of the LightGBM probability
MAXLEN, BS, N_TRAIN, LR = 128, 32, int(os.environ.get("N_TRAIN", 300_000)), 2e-5
CLF_RUN = os.environ.get("CLF_RUN", "fit_v2_cap5")


def tokenize_pairs(tok, a: list[str], b: list[str]):
    """The fast tokenizer sometimes fails to build tensors for a batch (an intermittent 'excessive nesting' ValueError):
    retry, then pad by hand, then encode pair by pair (an empty encoding becomes [CLS][SEP])."""
    for _ in range(3):
        try:
            return tok(a, b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        except ValueError:
            try:
                return tok.pad(tok(a, b, truncation=True, max_length=MAXLEN), padding=True, return_tensors="pt")
            except ValueError:
                time.sleep(1)
    ids = [tok(x, y, truncation=True, max_length=MAXLEN)["input_ids"] or [tok.cls_token_id, tok.sep_token_id] for x, y in zip(a, b)]
    return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prepare() -> pd.DataFrame:
    """Val pool pairs with label, LightGBM probability and split role (same split as s5lgbm.fit)."""
    saved = joblib.load(RUNS / CLF_RUN / "model.pkl")
    pairs = S.build_pairs("val")
    gt = pd.read_parquet(S.GT / "val_positive_pairs.parquet")[KEYS].assign(label=1)
    pairs = pairs.merge(gt, how="left", on=KEYS)
    pairs["label"] = pairs.label.fillna(0).astype(int)
    cache = S.CACHE / f"val_features_{S.POOL_RUN + '_' if S.POOL_RUN else ''}{hashlib.md5(','.join(S.FEATS).encode()).hexdigest()[:8]}.parquet"
    f = pd.read_parquet(cache)
    assert (f.entity_id_s1.to_numpy() == pairs.entity_id_s1.to_numpy()).all() and (f.entity_id_s2s3.to_numpy() == pairs.entity_id_s2s3.to_numpy()).all(), "feature cache not aligned with pool"
    assert saved["feats"] == S.FEATS
    pairs["prob"] = saved["model"].predict_proba(f[saved["feats"]])[:, 1]
    pairs["role"] = S.split_roles(pairs.entity_id_s1)
    pairs["border"] = (pairs.prob >= LO) & (pairs.prob <= HI)
    return pairs[KEYS + ["label", "prob", "role", "border"]]


def with_text(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = ["entity_id", "name_clean", "addr_clean"]
    i1, i2 = set(df.entity_id_s1), set(df.entity_id_s2s3)
    s1 = pd.read_parquet(CLEAN / "val_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(i1)].set_index("entity_id")
    idx = pd.concat([pd.read_parquet(CLEAN / f"val_{s}.parquet", columns=cols).pipe(lambda d: d[d.entity_id.isin(i2)]) for s in ("s2", "s3")]).set_index("entity_id")

    def fmt(t: pd.DataFrame, ids: pd.Series) -> list[str]:
        n = t.name_clean.reindex(ids).fillna("").to_numpy()
        a = t.addr_clean.reindex(ids).fillna("").replace("missing", "").to_numpy()
        return [f"{x} | {y}" for x, y in zip(n, a)]
    return fmt(s1, df.entity_id_s1), fmt(idx, df.entity_id_s2s3)


@torch.no_grad()
def score(model, tok, a: list[str], b: list[str], bs: int = int(os.environ.get("XLMR_BS", 256))) -> np.ndarray:
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])  # similar lengths together = less padding
    out = np.empty(len(a), np.float32)
    model.eval()
    for i in range(0, len(a), bs):
        ix = order[i:i + bs]
        enc = tokenize_pairs(tok, [a[j] for j in ix], [b[j] for j in ix]).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


def train() -> None:
    rd = run_dir()
    pairs = prepare()
    bl = pairs[pairs.border & (pairs.role == 0)]
    print(f"uncertain-zone pairs: fit {len(bl):,} (positive rate {bl.label.mean():.1%}); "
          f"early-stop {int((pairs.border & (pairs.role == 1)).sum()):,}; holdout {int((pairs.border & (pairs.role == 2)).sum()):,} "
          f"of {int((pairs.role == 2).sum()):,} holdout pairs", flush=True)
    tr = bl.sample(min(N_TRAIN, len(bl)), random_state=0).reset_index(drop=True)
    a, b = with_text(tr)
    y = torch.tensor(tr.label.to_numpy(), dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=1).cuda()
    model.roberta.embeddings.word_embeddings.weight.requires_grad = False  # 192M of 278M params: saves VRAM and time
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = math.ceil(len(tr) / BS)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    perm = np.random.RandomState(0).permutation(len(tr))
    lossf = torch.nn.BCEWithLogitsLoss()
    t0, run = time.time(), 0.0
    model.train()
    for step in range(steps):
        ix = perm[step * BS:(step + 1) * BS]
        enc = tokenize_pairs(tok, [a[j] for j in ix], [b[j] for j in ix]).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = lossf(logits.float(), y[ix].cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {run:.4f}  {(step + 1) * BS / (time.time() - t0):.0f} pairs/s  elapsed {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(rd / "xlmr")
    tok.save_pretrained(rd / "xlmr")
    print(f"saved {rd / 'xlmr'} after {time.time() - t0:.0f}s")


def eval_sets(df: pd.DataFrame):
    gt = pd.read_parquet(S.GT / "val_positive_pairs.parquet")[KEYS]
    ids = df.entity_id_s1.unique()
    code = pd.Series(np.arange(len(ids)), ids)
    n_truth = gt[gt.entity_id_s1.isin(ids)].groupby("entity_id_s1").size().reindex(ids).fillna(0).to_numpy()
    return code.reindex(df.entity_id_s1).to_numpy().astype(int), n_truth


def evaluate() -> None:
    rd = run_dir()
    pairs = prepare()
    mdir = RUNS / (os.environ.get("XLMR_MODEL_RUN") or run_id()) / "xlmr"   # the run that trained the cross-encoder
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    scored = {}
    for role in (1, 2):
        sub = pairs[pairs.border & (pairs.role == role)]
        a, b = with_text(sub)
        t0 = time.time()
        scored[role] = (sub.index.to_numpy(), score(model, tok, a, b))
        print(f"scored role {role}: {len(sub):,} pairs in {time.time() - t0:.0f}s ({len(sub) / (time.time() - t0):.0f}/s)", flush=True)
    pairs["xlmr"] = np.nan
    for role, (ix, sc) in scored.items():
        pairs.loc[ix, "xlmr"] = sc

    es, ho = pairs[pairs.border & (pairs.role == 1)], pairs[pairs.border & (pairs.role == 2)]
    print("\n=== uncertain-zone pairs: AUC (label = true match) ===")
    for name, d in (("early-stop", es), ("holdout", ho)):
        print(f"  {name:10s} n={len(d):,}  LightGBM {roc_auc_score(d.label, d.prob):.4f}   XLM-R {roc_auc_score(d.label, d.xlmr):.4f}")
    lr = LogisticRegression(C=10.0).fit(np.c_[logit(es.prob), es.xlmr], es.label)  # blend weights fitted on early-stop only
    print(f"  blend weights (logit LightGBM, XLM-R, bias): {lr.coef_[0].round(3)}, {lr.intercept_.round(3)}")
    pairs["blend"] = pairs.prob
    m = pairs.border & pairs.xlmr.notna()
    pairs.loc[m, "blend"] = lr.predict_proba(np.c_[logit(pairs.loc[m, "prob"]), pairs.loc[m, "xlmr"]])[:, 1]
    print(f"  holdout AUC blend {roc_auc_score(ho.label, pairs.loc[ho.index, 'blend']):.4f}")

    print("\n=== macro F0.5 (threshold chosen on early-stop entities, applied to the holdout) ===")
    res = {}
    for col in ("prob", "blend"):
        thr = {}
        for role in (1, 2):
            d = pairs[pairs.role == role].reset_index()  # keep original index in column 'index'
            s1c, nt = eval_sets(d)
            p, lab = d[col].to_numpy(), d.label.to_numpy()
            keep = S.one_to_one(d, p)
            if role == 1:
                fs = [(S.macro_f05((p >= t) & keep, s1c, lab, nt), t) for t in np.linspace(0.2, 0.95, 76)]
                thr[1] = max(fs)[1]
            else:
                sel = (p >= thr[1]) & keep
                tp = (sel & (lab == 1)).sum()
                res[col] = (S.macro_f05(sel, s1c, lab, nt), tp / max(sel.sum(), 1), tp / nt.sum(), thr[1],
                            max((S.macro_f05((p >= t) & keep, s1c, lab, nt), t) for t in np.linspace(0.2, 0.95, 76)))
    import json
    (rd / "blend.json").write_text(json.dumps({"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                                               "threshold": float(res["blend"][3]), "lo": LO, "hi": HI}, indent=2))
    for col, (f, pr, rc, t, (fo, to)) in res.items():
        print(f"  {'LightGBM only' if col == 'prob' else 'LightGBM + XLM-R':18s} F0.5 = {f:.4f} @ t={t:.2f}  precision {pr:.3f}  recall {rc:.3f}   (oracle threshold {to:.2f}: {fo:.4f})")
    print(f"  gain from XLM-R blend: {res['blend'][0] - res['prob'][0]:+.4f}")


def train2() -> None:
    """Continue training the cross-encoder (XLMR_INIT_RUN) on pairs from the TRAIN split (never val or test entities): each mined anchor
    gives up to 2 true records (label 1) and 2 hard negatives from the bi-encoder search (label 0), with a checkpoint every 4000 steps."""
    rd = run_dir()
    mined = pd.read_parquet(RUNS / os.environ["MINED_RUN"] / "mined_train.parquet")
    mined = mined[mined.negs.map(len) > 0].sample(frac=1, random_state=1).head(int(os.environ.get("N_ANCHORS", 300_000)))
    rng = np.random.RandomState(0)
    rows = []
    for a, pos, negs in zip(mined.entity_id_s1, mined.pos, mined.negs):
        rows += [(a, p, 1) for p in pos[:2]]
        rows += [(a, negs[j], 0) for j in rng.choice(min(len(negs), 8), size=min(2, len(negs)), replace=False)]
    tr = pd.DataFrame(rows, columns=["entity_id_s1", "entity_id_s2s3", "label"]).sample(frac=1, random_state=2).reset_index(drop=True)
    print(f"train pairs {len(tr):,} from {len(mined):,} anchors, positive rate {tr.label.mean():.1%}", flush=True)
    cols = ["entity_id", "name_clean", "addr_clean"]
    i1, i2 = set(tr.entity_id_s1), set(tr.entity_id_s2s3)
    s1 = pd.read_parquet(CLEAN / "train_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(i1)].set_index("entity_id")
    idx = pd.concat([pd.read_parquet(CLEAN / f"val_{s}.parquet", columns=cols).pipe(lambda d: d[d.entity_id.isin(i2)]) for s in ("s2", "s3")]).set_index("entity_id")   # val_s2/s3 == train pool

    def fmt(t, ids):
        n = t.name_clean.reindex(ids).fillna("").to_numpy()
        ad = t.addr_clean.reindex(ids).fillna("").replace("missing", "").to_numpy()
        return [f"{x} | {y}" for x, y in zip(n, ad)]
    a, b = fmt(s1, tr.entity_id_s1), fmt(idx, tr.entity_id_s2s3)
    y = torch.tensor(tr.label.to_numpy(), dtype=torch.float32)
    init = RUNS / os.environ["XLMR_INIT_RUN"] / "xlmr"
    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init).cuda()
    model.roberta.embeddings.word_embeddings.weight.requires_grad = False
    lr = float(os.environ.get("XLMR_LR", 1e-5))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = len(tr) // BS
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    ck, step = rd / "train_ckpt.pt", 0
    if ck.exists():
        st = torch.load(ck, map_location="cuda")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"resumed from step {step}", flush=True)
    t0, run = time.time(), 0.0
    model.train()
    while step < steps:
        ix = np.arange(step * BS, (step + 1) * BS)
        enc = tokenize_pairs(tok, [a[j] for j in ix], [b[j] for j in ix]).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = lossf(logits.float(), y[ix].cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {run:.4f}  elapsed {time.time() - t0:.0f}s", flush=True)
        if step % int(os.environ.get("CKPT_EVERY", 4000)) == 0 and step < steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
    model.save_pretrained(rd / "xlmr")
    tok.save_pretrained(rd / "xlmr")
    print(f"saved {rd / 'xlmr'} after {time.time() - t0:.0f}s")


UNSEEN_THRESHOLD = 0.80   # countries absent from training: the country-held-out check favoured a stricter cutoff


def test_stage() -> None:
    """Re-score the uncertain test pairs with XLM-R, blend, apply one-to-one + per-country threshold, write the TSVs."""
    import json
    rd = run_dir()
    out_dir = rd / "output"
    targets = [S.safe_path(out_dir / n) for n in ("candidate_pairs.tsv", "matching_results.tsv")]
    blend = json.loads((RUNS / os.environ["XLMR_RUN"] / "blend.json").read_text())
    scores = pd.read_parquet(RUNS / os.environ["LGB_RUN"] / "scores_test.parquet")
    border = ((scores.prob >= blend["lo"]) & (scores.prob <= blend["hi"])).to_numpy()
    print(f"test pairs {len(scores):,}; uncertain zone {border.sum():,} ({border.mean():.1%})", flush=True)

    sub = scores[border].reset_index(drop=True)
    # An XLM-R score depends only on the pair's text and the model, so scores already computed for another classifier's
    # uncertain zone can be reused (XLMR_CACHE = comma-separated parquet files with entity_id_s1, entity_id_s2s3, xlmr).
    KEYS2 = ["entity_id_s1", "entity_id_s2s3"]
    known = np.full(len(sub), np.nan, np.float32)
    for path in filter(None, os.environ.get("XLMR_CACHE", "").split(",")):
        c = pd.read_parquet(path).drop_duplicates(KEYS2)
        known = np.where(np.isnan(known), sub[KEYS2].merge(c, how="left", on=KEYS2).xlmr.to_numpy(np.float32), known)
    need = np.flatnonzero(np.isnan(known))
    print(f"  XLM-R scores reused from cache: {len(sub) - len(need):,}; still to score: {len(need):,}", flush=True)
    todo = sub.iloc[need]
    cols = ["entity_id", "name_clean", "addr_clean"]
    i1, i2 = set(todo.entity_id_s1), set(todo.entity_id_s2s3)
    s1 = pd.read_parquet(CLEAN / "test_s1.parquet", columns=cols + ["country"])
    country = s1.set_index("entity_id").country
    s1 = s1[s1.entity_id.isin(i1)].set_index("entity_id")
    idx = pd.concat([pd.read_parquet(CLEAN / f"test_{s}.parquet", columns=cols).pipe(lambda d: d[d.entity_id.isin(i2)]) for s in ("s2", "s3")]).set_index("entity_id")

    def fmt(t, ids):
        n = t.name_clean.reindex(ids).fillna("").to_numpy()
        a = t.addr_clean.reindex(ids).fillna("").replace("missing", "").to_numpy()
        return [f"{x} | {y}" for x, y in zip(n, a)]
    a, b = fmt(s1, todo.entity_id_s1), fmt(idx, todo.entity_id_s2s3)
    del s1, idx
    t0 = time.time()
    if len(a):
        mdir = RUNS / (os.environ.get("XLMR_MODEL_RUN") or os.environ["XLMR_RUN"]) / "xlmr"
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    ck = rd / "ckpt"
    ck.mkdir(exist_ok=True)
    xl_new, BLK = np.empty(len(a), np.float32), 200_000
    for b0 in range(0, len(a), BLK):
        f = ck / f"xl_{b0}.npy"
        if f.exists():
            xl_new[b0:b0 + BLK] = np.load(f)
            print(f"  block {b0:,}: loaded from checkpoint", flush=True)
            continue
        part = score(model, tok, a[b0:b0 + BLK], b[b0:b0 + BLK])
        np.save(str(f) + ".tmp.npy", part)
        os.replace(str(f) + ".tmp.npy", f)   # atomic: a crash can never leave a half-written checkpoint under its final name
        xl_new[b0:b0 + BLK] = part
        print(f"  scored {min(b0 + BLK, len(a)):,}/{len(a):,}  ({time.time() - t0:.0f}s)", flush=True)
    xl = known.copy()
    xl[need] = xl_new
    pd.DataFrame({"entity_id_s1": sub.entity_id_s1.to_numpy(), "entity_id_s2s3": sub.entity_id_s2s3.to_numpy(), "xlmr": xl}).to_parquet(rd / "xlmr_scores.parquet", index=False)
    print(f"XLM-R scored {len(a):,} new pairs in {time.time() - t0:.0f}s; total {len(xl):,} scores saved to {rd / 'xlmr_scores.parquet'}", flush=True)

    prob = scores.prob.to_numpy().astype(np.float64)
    z = blend["intercept"] + blend["coef"][0] * logit(prob[border]) + blend["coef"][1] * xl
    prob[border] = 1 / (1 + np.exp(-z))
    keep = S.one_to_one(scores, prob)
    seen = set(pd.read_parquet(CLEAN / "val_s1.parquet", columns=["country"]).country.unique())
    ctry = country.reindex(scores.entity_id_s1).to_numpy()
    thr = np.where(np.isin(ctry, list(seen)), blend["threshold"], max(blend["threshold"], UNSEEN_THRESHOLD))
    print(f"seen {sorted(seen)}; unseen in test {sorted(set(country) - seen)}; thresholds {sorted(set(thr.round(2)))}")
    sel = (prob >= thr) & keep
    print(f"predicted matches {sel.sum():,} ({sel.sum() / scores.entity_id_s1.nunique():.2f} per S1)")
    S.write_tsv(country.index.to_series(), scores, "candidate_entity_ids", targets[0])
    S.write_tsv(country.index.to_series(), scores[sel], "matched_entity_ids", targets[1])
    print("matches by country:\n" + pd.Series(ctry[sel]).value_counts().to_string())


if __name__ == "__main__":
    {"train": train, "train2": train2, "eval": evaluate, "test": test_stage}[sys.argv[1]]()
