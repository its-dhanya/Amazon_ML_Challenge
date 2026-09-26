# Amazon ML Challenge: Business Entity Resolution

Match every Source 1 (S1) business to its records in Source 2 / Source 3 (S2/S3). Scored by macro-averaged F0.5 per S1 entity.
All code is in [`Dhanya/`](Dhanya/); run everything from inside that directory.

| Submission | Approach | Val holdout F0.5 | Leaderboard F0.5 |
|---|---|---|---|
| 1 | Per-country sparse BM25 blocking + LightGBM pair classifier | 0.904 | **0.879** |
| 2 | + address-only and character n-gram blocking, cross-script features, XLM-R cross-encoder on uncertain pairs | 0.959 | **0.941** |

## Approach

1. **Clean** both sides (unicode, legal suffixes, PIN/state extraction) (`s23normalize.py`).
2. **Blocking** (candidate generation) per country, so a record is only ever compared within its country:
   - joint name+address word BM25, top 20 per S1 entity (`s4sparse.py`);
   - submission 2 adds two independent BM25 representations, top 5 each (`s4ngram.py`): address-only words (**Wa**) and
     character 4-grams of the romanized, despaced name (**Cn**; any script is transliterated with `anyascii`), which recover
     names in another script, glued/domain-style names, typos and records with an empty address;
   - the branches are unioned per pair, keeping each branch's rank and score (`s4union.py` / `s4representations.py`).
3. **Pair scoring**: a LightGBM classifier on name/address similarity features (`s5lgbm.py`, `FEATSET=v1` is submission 1,
   `FEATSET=v2` adds branch ranks/scores and cross-script, glued-name and house-number features).
4. **Second stage (submission 2)**: pairs whose LightGBM probability is in [0.10, 0.95] are re-scored by a fine-tuned
   `xlm-roberta-base` cross-encoder (MIT license) and blended with LightGBM (`xlmr_cross.py`).
5. **Decision**: each S2/S3 record goes to at most one S1 entity (its highest-scoring one, true in the training labels),
   then a probability threshold tuned for F0.5 (stricter for a country unseen in training, i.e. France).

No external data, lookup service or API is used. The only downloaded artifact is the `xlm-roberta-base` weights.

## Setup

Tested on Windows 11, Python 3.10, 32 GB RAM, one RTX 4060 (8 GB VRAM). Plan for about 20 GB of free disk. Peak memory is
4 to 10 GB per job, so with 16 GB of RAM run one job at a time. The GPU is only needed for the XLM-R steps.

```bash
git clone https://github.com/CS23I1065/Amazon_ML_Challenge.git && cd Amazon_ML_Challenge
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r Dhanya/requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA build; needed for the XLM-R steps
```

Put the challenge data at the repository root so that these files exist (the scripts read them via `../dataset/...`):

```
dataset/student_resource/dataset/train/{train_source1,train_source2,train_source3}.tsv, train_ground_truth.tsv
dataset/student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

Download the cross-encoder weights once (about 1.1 GB, MIT license; only needed for submission 2):

```bash
cd Dhanya
python -c "from huggingface_hub import snapshot_download as d; d('FacebookAI/xlm-roberta-base', local_dir='data/models/xlm-roberta-base', allow_patterns=['config.json','model.safetensors','sentencepiece.bpe.model','tokenizer.json','tokenizer_config.json'])"
```

## Running the pipeline

Everything below runs from `Dhanya/`. Environment variables are shown in bash syntax; in PowerShell use `$env:RUN_ID='x'`.
Each run writes its logs and results to its own `runs/<RUN_ID>/` and refuses to overwrite an existing result
(`RUN_OVERWRITE=1` to replace); see [`Dhanya/runs.py`](Dhanya/runs.py).

### Shared preparation

```bash
python s0selection.py        # split train S1 into train/val (10%, stratified by country), expand labels to positive pairs
python s23normalize.py       # data/processed/*.parquet -> data/cleaned/*.parquet
# optional: python s1profiling.py   (read-only profile; expects the data under student_resource/dataset)
```

### Lexical blocking (both submissions)

```bash
python s4sparse.py val                       # whole split in one process (about 20 min), writes data/branch_candidates/val_sparkly.parquet
# test is large: run country slices in parallel (each ~30 min, ~4 GB), then merge
python s4sparse.py test India 1 3 &  python s4sparse.py test India 2 3 &  python s4sparse.py test India 3 3 &
python s4sparse.py test US 1 1 &     python s4sparse.py test France 1 1 & wait
python s4sparse.py test merge
python s4representations.py val test         # country check + union -> data/candidates/candidate_pairs_{val,test}.parquet
```

### Submission 1: LightGBM only (leaderboard 0.879)

```bash
RUN_ID=fit_v1 python s5lgbm.py fit                         # FEATSET=v1 is the default; holdout F0.5 about 0.904
RUN_ID=sub1 MODEL_RUN=fit_v1 python s5lgbm.py test         # -> runs/sub1/output/{matching_results,candidate_pairs}.tsv
```

### Submission 2: extra branches, v2 features, XLM-R (leaderboard 0.941)

```bash
# 1) extra blocking branches, top 5 each (val: one process per country; test: slices as above)
export RUN_ID=cand_v2 TOPK=5
python s4ngram.py run val India 1 1 Wa Cn ;  python s4ngram.py run val US 1 1 Wa Cn ;  python s4ngram.py merge val Wa Cn
python s4ngram.py run test India 1 2 Wa Cn ; python s4ngram.py run test India 2 2 Wa Cn      # run slices in parallel
python s4ngram.py run test US 1 2 Wa Cn ;    python s4ngram.py run test US 2 2 Wa Cn
python s4ngram.py run test France 1 1 Wa Cn ; python s4ngram.py merge test Wa Cn
unset TOPK

# 2) candidate pools = lexical (rank <= 20) + Wa (<= 5) + Cn (<= 5)
B=runs/cand_v2/branches
RUN_ID=cand_v2_cap5   python s4union.py val  lex=data/branch_candidates/val_sparkly.parquet:20  wa=$B/val_Wa.parquet:5  cn=$B/val_Cn.parquet:5
RUN_ID=cand_test_cap5 python s4union.py test lex=data/branch_candidates/test_sparkly.parquet:20 wa=$B/test_Wa.parquet:5 cn=$B/test_Cn.parquet:5

# 3) LightGBM v2 on the val pool (holdout F0.5 about 0.9315)
RUN_ID=fit_v2_cap5 POOL_RUN=cand_v2_cap5 FEATSET=v2 python s5lgbm.py fit

# 4) XLM-R cross-encoder on the uncertain pairs (GPU): fine-tune (about 15 min), then evaluate and save the blend weights
X="RUN_ID=xlmr_v1 POOL_RUN=cand_v2_cap5 FEATSET=v2 CLF_RUN=fit_v2_cap5"
env $X python xlmr_cross.py train
env $X python xlmr_cross.py eval                # holdout F0.5 LightGBM 0.9310 -> blend 0.9589; writes runs/xlmr_v1/blend.json

# 5) test: LightGBM probabilities for every pair (about 10 min), then XLM-R re-scoring of the uncertain ones
RUN_ID=lgb_test_v2 MODEL_RUN=fit_v2_cap5 POOL_RUN=cand_test_cap5 FEATSET=v2 python s5lgbm.py test
RUN_ID=xlmr_test XLMR_RUN=xlmr_v1 LGB_RUN=lgb_test_v2 POOL_RUN=cand_test_cap5 FEATSET=v2 python xlmr_cross.py test
# -> runs/xlmr_test/output/matching_results.tsv (upload) and candidate_pairs.tsv
```

The last step is the slowest: 2.1 million test pairs were re-scored in about 100 minutes on the 8 GB GPU.

### Validate the output

```bash
cd ../dataset/student_resource
python utils/validate_submission.py --matching ../../Dhanya/runs/xlmr_test/output/matching_results.tsv \
    --candidate ../../Dhanya/runs/xlmr_test/output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Repository layout

| Path | Purpose |
|---|---|
| `Dhanya/s0selection.py`, `s23normalize.py`, `s1profiling.py` | data split, cleaning, profiling |
| `Dhanya/s4sparse.py` | joint word BM25 blocking (sparse matrix product, parallel country slices) |
| `Dhanya/s4ngram.py` | independent BM25 branches: address-only (Wa), romanized character n-grams (Cn) |
| `Dhanya/s4representations.py`, `s4union.py` | union of the branches (submission 1 / submission 2) |
| `Dhanya/s5lgbm.py` | LightGBM pair classifier, one-to-one assignment, thresholds, TSV writer |
| `Dhanya/xlmr_cross.py` | XLM-R cross-encoder second stage |
| `Dhanya/runs.py`, `Dhanya/runs/` | per-run folders; only small metadata (READMEs, `config.json`, `metrics.json`, `blend.json`) is tracked |
| `Dhanya/s4uniblocker.py`, `s4sparkly.py`, `s5splink.py`, `s5fellegisunter.py`, `s5beval.py` | earlier or abandoned attempts (dense retrieval, Sparkly, Splink), not used by either submission |

Data, model weights and outputs are not tracked (see `.gitignore`); the commands above regenerate them.
