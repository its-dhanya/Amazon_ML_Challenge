# xlmr_test_v3  (leaderboard F0.5 = 0.964)

Submission 3. `output/matching_results.tsv` is the uploaded file (git-ignored: size); regenerate with the commands in the root README,
section "Submission 3". Chain of runs: `dense_v1` (bi-encoder, embeddings, val/test dense branches) -> `cand_v3_cap5` / `cand_v3_test_cap5`
(pools with the dense branch, cap 5) -> `fit_v3_cap5` (LightGBM, `FEATSET=v3`, val holdout 0.9618) -> `xlmr_v3` (blend weights for the
cross-encoder trained in `xlmr_v1`, val holdout 0.9742) -> `lgb_test_v3` (test probabilities) -> `xlmr_test_v3` (re-scoring of the
1,454,419 uncertain test pairs + decision). Thresholds: 0.75 for US/India, 0.80 for France (unseen in training).
Test: 27.5M candidate pairs, 3.23 predicted matches per S1 entity. `runs/dense_v1/STATUS.md` has the full log of this run.
