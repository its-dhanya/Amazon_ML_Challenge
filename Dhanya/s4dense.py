"""Dense (semantic) retrieval branch: a multilingual bi-encoder fine-tuned on the val pairs, plus exact GPU nearest-neighbour search.

    RUN_ID=dense_v1 python s4dense.py train            # contrastive fine-tune (InfoNCE, in-batch + one hard negative per anchor)
    RUN_ID=dense_v1 python s4dense.py encode val       # embed every S1 record and every S2/S3 record of a split, per country
    RUN_ID=dense_v1 python s4dense.py search val 10    # top-10 same-country neighbours per S1 -> runs/dense_v1/branches/val_dense.parquet

Records are embedded as "name | address" (max 40 tokens). Embeddings are stored in half precision under runs/<RUN_ID>/emb/.
Only entities of the fit group (s5lgbm.split_roles == 0) are used for training; the holdout is never seen.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # set before transformers is imported (intermittent tokenizer crashes seen)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from runs import RUNS, run_dir, run_id, safe_path

CLEAN = Path("data/cleaned")
BASE = Path("data/models/paraphrase-multilingual-MiniLM-L12-v2")   # Apache-2.0, 384-d, covers the XLM-R vocabulary
MAXLEN, DIM, TAU = 40, 384, 0.05
BATCH, EPOCHS, LR = 192, int(os.environ.get("EPOCHS", 4)), 3e-5
KEYS = ["entity_id_s1", "entity_id_s2s3"]


def texts_of(df: pd.DataFrame) -> list[str]:
    addr = df.addr_clean.fillna("").replace("missing", "")
    return (df.name_clean.fillna("") + " | " + addr).tolist()


def tokenize(tok, texts: list[str]):
    """Batch tokenization with a fallback: the fast tokenizer occasionally fails to build tensors for a batch
    (an intermittent 'excessive nesting' ValueError that does not reproduce on the same texts), so retry, then pad by hand."""
    for attempt in range(3):
        try:
            return tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        except ValueError:
            try:
                return tok.pad(tok(texts, truncation=True, max_length=MAXLEN), padding=True, return_tensors="pt")
            except ValueError:
                time.sleep(1)
    ids = [tok(t, truncation=True, max_length=MAXLEN)["input_ids"] or [tok.unk_token_id] for t in texts]  # last resort: one by one
    return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")


def encode_batch(model, tok, texts: list[str]) -> torch.Tensor:
    enc = tokenize(tok, texts).to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        h = model(**enc).last_hidden_state
    m = enc.attention_mask.unsqueeze(-1).to(h.dtype)
    e = (h * m).sum(1) / m.sum(1).clamp(min=1)          # mean pooling
    return torch.nn.functional.normalize(e.float(), dim=-1)


# ── training ────────────────────────────────────────────────────────────────────────────────────
def train() -> None:
    import s5lgbm as S
    rd = run_dir()
    out = safe_path(rd / "biencoder" / "train_info.json")  # not config.json: save_pretrained writes the model config there
    gt = pd.read_parquet("data/processed/val_positive_pairs.parquet")[KEYS]
    pos = gt[S.split_roles(gt.entity_id_s1) == 0]
    pool = pd.read_parquet(RUNS / "cand_v2_cap5" / "candidates" / "candidate_pairs_val.parquet", columns=KEYS)
    pool = pool[S.split_roles(pool.entity_id_s1) == 0]
    neg = pool.merge(pos.assign(_m=1), how="left", on=KEYS)
    neg = neg[neg._m.isna()][KEYS]                       # hard negatives: pool candidates that are not matches
    pos_d = pos.groupby("entity_id_s1").entity_id_s2s3.agg(list).to_dict()
    neg_d = neg.groupby("entity_id_s1").entity_id_s2s3.agg(list).to_dict()
    ents = np.array(sorted(set(pos_d) & set(neg_d)))
    print(f"fit entities with a positive and a hard negative: {len(ents):,} (of {len(pos_d):,})", flush=True)

    cols = ["entity_id", "name_clean", "addr_clean"]
    need_idx = set(pos.entity_id_s2s3) | set(neg.entity_id_s2s3)
    s1 = pd.read_parquet(CLEAN / "val_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(set(ents))]
    t1 = dict(zip(s1.entity_id, texts_of(s1)))
    t2 = {}
    for src in ("s2", "s3"):
        d = pd.read_parquet(CLEAN / f"val_{src}.parquet", columns=cols)
        d = d[d.entity_id.isin(need_idx)]
        t2.update(zip(d.entity_id, texts_of(d)))
    del s1, d, pool, neg

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    steps_per_epoch = len(ents) // BATCH
    total = steps_per_epoch * EPOCHS
    warm = int(0.06 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(warm, 1), max(0.0, (total - s) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.RandomState(0)
    t0, step, run_loss, run_acc = time.time(), 0, 0.0, 0.0
    model.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(ents))
        for b in range(steps_per_epoch):
            e = ents[perm[b * BATCH:(b + 1) * BATCH]]
            a = [t1[x] for x in e]
            p = [t2[pos_d[x][rng.randint(len(pos_d[x]))]] for x in e]
            n = [t2[neg_d[x][rng.randint(len(neg_d[x]))]] for x in e]
            emb = encode_batch(model, tok, a + p + n)
            logits = emb[:BATCH] @ emb[BATCH:].T / TAU                       # each anchor vs all positives and hard negatives
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(BATCH, device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            acc = (logits.argmax(1) == torch.arange(BATCH, device="cuda")).float().mean().item()
            run_loss = loss.item() if step == 0 else 0.98 * run_loss + 0.02 * loss.item()
            run_acc = acc if step == 0 else 0.98 * run_acc + 0.02 * acc
            if step % 100 == 0:
                print(f"epoch {ep + 1}/{EPOCHS} step {step}/{total} loss {run_loss:.4f} in-batch acc {run_acc:.3f}  {time.time() - t0:.0f}s", flush=True)
            step += 1
    model.save_pretrained(out.parent)
    tok.save_pretrained(out.parent)
    out.write_text(f'{{"epochs": {EPOCHS}, "batch": {BATCH}, "tau": {TAU}, "maxlen": {MAXLEN}, "entities": {len(ents)}, "seconds": {time.time() - t0:.0f}}}')
    print(f"saved {out.parent} after {time.time() - t0:.0f}s")


# ── encoding ────────────────────────────────────────────────────────────────────────────────────
def load_model():
    d = RUNS / (os.environ.get("BIENC_RUN") or run_id()) / "biencoder"
    return AutoModel.from_pretrained(d).cuda().eval(), AutoTokenizer.from_pretrained(d)


@torch.inference_mode()
def embed_to(path: Path, texts: list[str], model, tok, bs: int = 1024) -> None:
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), DIM))
    order = np.argsort([len(t) for t in texts], kind="stable")     # similar lengths together = little padding
    t0 = time.time()
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        out[ix] = encode_batch(model, tok, [texts[j] for j in ix]).half().cpu().numpy()
        if (i // bs) % 500 == 0:
            print(f"    {i:,}/{len(texts):,}  {i / max(time.time() - t0, 1e-9):,.0f} records/s", flush=True)
    out.flush()


@torch.inference_mode()
def fill_missing(path: Path, texts: list[str], model, tok, bs: int = 1024) -> None:
    """Encode only the rows that a crashed run did not write (a valid normalized embedding is never all-zero)."""
    out = np.load(path, mmap_mode="r+")
    zero = np.zeros(len(texts), bool)
    for i in range(0, len(texts), 500_000):
        zero[i:i + 500_000] = np.abs(out[i:i + 500_000].astype(np.float32)).sum(1) == 0
    idx = np.flatnonzero(zero)
    print(f"    {len(idx):,} of {len(texts):,} rows missing; encoding them", flush=True)
    idx = idx[np.argsort([len(texts[j]) for j in idx], kind="stable")]
    for i in range(0, len(idx), bs):
        ix = idx[i:i + bs]
        out[ix] = encode_batch(model, tok, [texts[j] for j in ix]).half().cpu().numpy()
    out.flush()
    left = int((np.abs(out.astype(np.float32)).sum(1) == 0).sum()) if len(texts) < 6_000_000 else -1
    print(f"    done; rows still all-zero: {left}", flush=True)


def encode(split: str, only_country: str | None = None, only_part: str | None = None) -> None:
    emb_dir = run_dir() / "emb"
    emb_dir.mkdir(exist_ok=True)
    model, tok = load_model()
    cols = ["entity_id", "country", "name_clean", "addr_clean"]
    for country in pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=["country"]).country.unique():
        if only_country and country != only_country:
            continue
        flt = [("country", "==", country)]
        q = pd.read_parquet(CLEAN / f"{split}_s1.parquet", columns=cols, filters=flt)
        d = pd.concat([pd.read_parquet(CLEAN / f"{split}_{s}.parquet", columns=cols, filters=flt) for s in ("s2", "s3")], ignore_index=True)
        for name, df in (("q", q), ("d", d)):
            if only_part and name != only_part:
                continue
            print(f"[{split}] {country} {name}: {len(df):,} records", flush=True)
            if os.environ.get("FILL_MISSING") == "1":     # repair a partially written file, keep the existing ids/rows
                fill_missing(emb_dir / f"{split}_{country}_{name}.npy", texts_of(df), model, tok)
                continue
            df[["entity_id"]].to_parquet(safe_path(emb_dir / f"{split}_{country}_{name}_ids.parquet"), index=False)
            embed_to(safe_path(emb_dir / f"{split}_{country}_{name}.npy"), texts_of(df), model, tok)
        del q, d


# ── exact nearest-neighbour search on the GPU ──────────────────────────────────────────────────
@torch.inference_mode()
def search(split: str, k: int = 10) -> None:
    emb_dir = run_dir() / "emb"
    target = safe_path(run_dir() / "branches" / f"{split}_dense.parquet")
    parts = []
    for qf in sorted(emb_dir.glob(f"{split}_*_q.npy")):
        country = qf.name[len(split) + 1:-len("_q.npy")]
        q = torch.from_numpy(np.load(qf))
        d = torch.from_numpy(np.load(emb_dir / f"{split}_{country}_d.npy")).cuda()      # resident in VRAM (half precision)
        qid = pd.read_parquet(emb_dir / f"{split}_{country}_q_ids.parquet").entity_id.to_numpy()
        did = pd.read_parquet(emb_dir / f"{split}_{country}_d_ids.parquet").entity_id.to_numpy()
        print(f"[{split}] {country}: {len(q):,} queries x {len(d):,} records", flush=True)
        QC, DC, t0 = 4096, 262144, time.time()
        top_s, top_i = [], []
        for s in range(0, len(q), QC):
            qq = q[s:s + QC].cuda()
            best_s = best_i = None
            for c in range(0, len(d), DC):
                sc = qq @ d[c:c + DC].T
                v, i = sc.topk(min(k, sc.shape[1]), dim=1)
                i = i + c
                if best_s is None:
                    best_s, best_i = v, i
                else:
                    v = torch.cat([best_s, v], 1); i = torch.cat([best_i, i], 1)
                    best_s, j = v.topk(k, dim=1); best_i = i.gather(1, j)
            top_s.append(best_s.float().cpu()); top_i.append(best_i.cpu())
            if (s // QC) % 20 == 0:
                print(f"    {s:,}/{len(q):,}  {time.time() - t0:.0f}s", flush=True)
        ts, ti = torch.cat(top_s).numpy(), torch.cat(top_i).numpy()
        parts.append(pd.DataFrame({"entity_id_s1": np.repeat(qid, ts.shape[1]), "entity_id_s2s3": did[ti.ravel()],
                                   "rank": np.tile(np.arange(1, ts.shape[1] + 1), len(qid)), "score": ts.ravel()}))
        del d
        torch.cuda.empty_cache()
    pd.concat(parts, ignore_index=True).to_parquet(target, index=False)
    print(f"saved {target}")


# ── round 2: many more anchors from the train split, hard negatives mined with an existing encoder ──────────────
def _knn(q: torch.Tensor, d: torch.Tensor, k: int, QC: int = 4096, DC: int = 262144):
    """exact top-k cosine neighbours of the rows of q (half precision, CPU) among the rows of d (half precision, on the GPU)."""
    out_s, out_i = [], []
    for s in range(0, len(q), QC):
        qq = q[s:s + QC].cuda()
        best_s = best_i = None
        for c in range(0, len(d), DC):
            v, i = (qq @ d[c:c + DC].T).topk(min(k, min(DC, len(d) - c)), dim=1)
            i = i + c
            if best_s is None:
                best_s, best_i = v, i
            else:
                v = torch.cat([best_s, v], 1)
                i = torch.cat([best_i, i], 1)
                best_s, j = v.topk(k, dim=1)
                best_i = i.gather(1, j)
        out_s.append(best_s.float().cpu())
        out_i.append(best_i.cpu())
    return torch.cat(out_s).numpy(), torch.cat(out_i).numpy()


@torch.inference_mode()
def mine(n_anchors: int, k: int = 16) -> None:
    """Pick n train-split S1 anchors, retrieve their nearest records with the round-1 encoder (EMB_RUN embeddings of the shared
    S2/S3 pool, BIENC_RUN model) and keep the retrieved records that are NOT true matches as hard negatives."""
    rd = run_dir()
    target = safe_path(rd / "mined_train.parquet")
    emb_run = RUNS / (os.environ.get("EMB_RUN") or "dense_v1") / "emb"
    model, tok = load_model()
    cols = ["entity_id", "country", "name_clean", "addr_clean"]
    gt = pd.read_parquet("data/processed/train_positive_pairs.parquet")[KEYS]
    s1 = pd.read_parquet(CLEAN / "train_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(set(gt.entity_id_s1))].sample(n_anchors, random_state=0)
    pos = gt.groupby("entity_id_s1").entity_id_s2s3.agg(list)
    parts = []
    for country, q in s1.groupby("country"):
        d_ids = pd.read_parquet(emb_run / f"val_{country}_d_ids.parquet").entity_id.to_numpy()   # val pool == train pool
        d = torch.from_numpy(np.load(emb_run / f"val_{country}_d.npy")).cuda()
        print(f"[mine] {country}: {len(q):,} anchors x {len(d):,} records", flush=True)
        texts, t0 = texts_of(q), time.time()
        order = np.argsort([len(t) for t in texts], kind="stable")
        e = np.empty((len(q), DIM), np.float16)
        for i in range(0, len(q), 1024):
            ix = order[i:i + 1024]
            e[ix] = encode_batch(model, tok, [texts[j] for j in ix]).half().cpu().numpy()
        _, ni = _knn(torch.from_numpy(e), d, k)
        negs = []
        for a, row in zip(q.entity_id.to_numpy(), ni):
            p_set = set(pos.get(a, ()))
            negs.append([x for x in d_ids[row] if x not in p_set])
        parts.append(pd.DataFrame({"entity_id_s1": q.entity_id.to_numpy(), "negs": negs}))
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
        del d
        torch.cuda.empty_cache()
    out = pd.concat(parts, ignore_index=True)
    out["pos"] = out.entity_id_s1.map(pos)
    out.to_parquet(target, index=False)
    print(f"saved {target}: {len(out):,} anchors, mean hard negatives {out.negs.map(len).mean():.1f}, mean positives {out.pos.map(len).mean():.1f}")


def train2() -> None:
    """Fine-tune the encoder FROM THE BASE MODEL on the mined train-split anchors (never on val or test entities),
    with a checkpoint every CKPT_EVERY steps so that a crash resumes instead of restarting."""
    while (RUNS / "PAUSE_GPU").exists():   # lets another GPU job go first: create runs/PAUSE_GPU, delete it to continue
        print("paused (runs/PAUSE_GPU exists)", flush=True)
        time.sleep(30)
    rd = run_dir()
    mined = pd.read_parquet(RUNS / (os.environ.get("MINED_RUN") or run_id()) / "mined_train.parquet")
    mined = mined[mined.negs.map(len) > 0].reset_index(drop=True)
    print(f"anchors with a positive and a hard negative: {len(mined):,}", flush=True)
    cols = ["entity_id", "name_clean", "addr_clean"]
    s1 = pd.read_parquet(CLEAN / "train_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(set(mined.entity_id_s1))]
    t1 = dict(zip(s1.entity_id, texts_of(s1)))
    need = set(x for lst in mined.pos for x in lst[:3]) | set(x for lst in mined.negs for x in lst[:8])
    t2 = {}
    for src in ("s2", "s3"):
        d = pd.read_parquet(CLEAN / f"val_{src}.parquet", columns=cols)       # val_s2/s3 == train_s2/s3 (shared pool)
        d = d[d.entity_id.isin(need)]
        t2.update(zip(d.entity_id, texts_of(d)))
    del s1, d
    ents = mined.entity_id_s1.to_numpy()
    pos_d = {a: [x for x in lst[:3] if x in t2] for a, lst in zip(ents, mined.pos)}
    neg_d = {a: [x for x in lst[:8] if x in t2] for a, lst in zip(ents, mined.negs)}
    ents = ents[np.array([bool(pos_d[a]) and bool(neg_d[a]) for a in ents])]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    lr2, epochs2, ck_every = 2e-5, 1, int(os.environ.get("CKPT_EVERY", 1000))
    opt = torch.optim.AdamW(params, lr=lr2, weight_decay=0.01)
    spe = len(ents) // BATCH
    total, warm = spe * epochs2, int(0.03 * spe * epochs2)
    total = min(total, int(os.environ.get("MAX_STEPS", total)))   # MAX_STEPS: short smoke test
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(warm, 1), max(0.0, (total - s) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda")
    ck_path = rd / "train_ckpt.pt"
    step = 0
    if ck_path.exists():                                  # resume after a crash
        st = torch.load(ck_path, map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed from step {step}", flush=True)
    perm_all = [np.random.RandomState(ep).permutation(len(ents)) for ep in range(epochs2)]
    rng, t0, run_loss, run_acc = np.random.RandomState(step), time.time(), 0.0, 0.0
    model.train()
    while step < total:
        ep, b = divmod(step, spe)
        e = ents[perm_all[ep][b * BATCH:(b + 1) * BATCH]]
        a = [t1[x] for x in e]
        p = [t2[pos_d[x][rng.randint(len(pos_d[x]))]] for x in e]
        n = [t2[neg_d[x][rng.randint(len(neg_d[x]))]] for x in e]
        emb = encode_batch(model, tok, a + p + n)
        logits = emb[:BATCH] @ emb[BATCH:].T / TAU
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(BATCH, device="cuda"))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        acc = (logits.argmax(1) == torch.arange(BATCH, device="cuda")).float().mean().item()
        run_loss = loss.item() if step == 0 else 0.98 * run_loss + 0.02 * loss.item()
        run_acc = acc if step == 0 else 0.98 * run_acc + 0.02 * acc
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {run_loss:.4f} in-batch acc {run_acc:.3f}  {time.time() - t0:.0f}s", flush=True)
        if step % ck_every == 0 and step < total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, str(ck_path) + ".tmp")
            os.replace(str(ck_path) + ".tmp", ck_path)
    out = rd / "biencoder"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (rd / "train_info.json").write_text(json.dumps({"round": 2, "anchors": int(len(ents)), "epochs": epochs2, "batch": BATCH, "steps": total, "seconds": round(time.time() - t0)}))
    print(f"saved {out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "encode":
        encode(sys.argv[2], *(sys.argv[3:5]))   # encode <split> [country [q|d]] to redo one piece
    elif cmd == "mine":
        mine(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif cmd == "train2":
        train2()
    elif cmd == "search":
        search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
