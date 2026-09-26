# xlmr_test  (leaderboard F0.5 = 0.941)

Submission 2. `output/matching_results.tsv` is the uploaded file, `output/candidate_pairs.tsv` the pool the models scored
(both are git-ignored because of their size; regenerate with the commands below).

Chain of runs (each in its own `Dhanya/runs/<RUN_ID>/`; environment variables as in `runs.py`):

| Stage | Run | Command (from `Dhanya/`) |
|---|---|---|
| Lexical joint word BM25, test | (shared) `data/branch_candidates/test_sparkly.parquet` | `python s4sparse.py test India 1 3` ... then `python s4sparse.py test merge` |
| Address-only (Wa) and despaced character 4-gram (Cn) BM25, val and test | `cand_v2` | `RUN_ID=cand_v2 TOPK=5 python s4ngram.py run test India 1 2 Wa Cn` (slices per country), then `python s4ngram.py merge test Wa Cn` |
| Union, cap 5 (test) | `cand_test_cap5` | `RUN_ID=cand_test_cap5 python s4union.py test lex=...:20 wa=...:5 cn=...:5` |
| LightGBM v2 fit on val (holdout F0.5 0.9315) | `fit_v2_cap5` | `RUN_ID=fit_v2_cap5 POOL_RUN=cand_v2_cap5 FEATSET=v2 python s5lgbm.py fit` |
| XLM-R cross-encoder fine-tune on uncertain pairs, blend weights | `xlmr_v1` | `RUN_ID=xlmr_v1 POOL_RUN=cand_v2_cap5 FEATSET=v2 CLF_RUN=fit_v2_cap5 python xlmr_cross.py train` then `eval` (holdout F0.5 0.9589) |
| LightGBM v2 probabilities on the test pool | `lgb_test_v2` | `RUN_ID=lgb_test_v2 MODEL_RUN=fit_v2_cap5 POOL_RUN=cand_test_cap5 FEATSET=v2 python s5lgbm.py test` |
| XLM-R re-scoring of the uncertain test pairs, decision | `xlmr_test` | `RUN_ID=xlmr_test XLMR_RUN=xlmr_v1 LGB_RUN=lgb_test_v2 POOL_RUN=cand_test_cap5 FEATSET=v2 python xlmr_cross.py test` |

Decision rule: pairs with LightGBM probability in [0.10, 0.95] are re-scored by XLM-R (`xlm-roberta-base`, MIT) and blended
by a logistic regression fitted on the early-stop entities (`runs/xlmr_v1/blend.json`); one-to-one assignment per S2/S3 record;
threshold 0.69 for countries seen in training (US, India) and 0.80 for the unseen country (France).
Test: 24.5M candidate pairs, 2.14M re-scored, 3.07 predicted matches per S1 entity.
