# sub1_lexical  (leaderboard F0.5 = 0.879)
First submission. Lexical joint word-BM25 blocking (top-20 per S1, rel. threshold 0.5) + LightGBM fit on val
(72% fit / 8% early-stop / 20% holdout by S1 entity), one-to-one assignment, threshold 0.62 (US, India) and 0.75 (unseen countries, i.e. France).
Val holdout F0.5 0.9028 (P 0.970, R 0.814). Files: output/matching_results.tsv (uploaded), output/candidate_pairs.tsv, model.pkl.
Copied from output/ and data/models/lgbm_pairs.pkl; those top-level paths are no longer written by any script.
