# Dense (bi-encoder) branch: overnight status

Started 2026-09-27 02:56. Submission #2 (leaderboard 0.941) is untouched in `runs/xlmr_test/`.

## Plan and stop rules
1. Fine-tune bi-encoder: DONE 03:16 (in-batch accuracy 0.994; `runs/dense_v1/biencoder`).
2. Encode val, exact GPU search (top 10): IN PROGRESS (val US documents ~7,500 records/s).
3. GATE on holdout entities: dense top-3/5 must add >= 1.5 points pool recall at <= 4 extra candidates/entity.
   FAIL -> stop, leave submission #2 as the best.
4. PASS -> encode/search test, build pools (dense cap 3 and 5), LightGBM v3 (`FEATSET=v3`), refit XLM-R blend, test scoring, XLM-R stage.
5. Write runs/xlmr_test_v3/output/matching_results.tsv, validate, compare with #2.
   Only worth uploading if end-to-end val holdout F0.5 > 0.9589 + 0.003.

## Log
- 03:21 val US documents 1.0M/6.2M encoded at ~7,500 records/s.
- 03:46 val encoding crashed in the India documents (intermittent tokenizer ValueError, not reproducible on the same texts). US documents, US queries and India queries were already complete and verified.
- 03:48 made tokenization retry with a fallback (s4dense.py) and restarted only `encode val India d`, then `search val 10`.
- 04:09 val encoding + search finished. **GATE PASSED** (holdout entities, never seen in training): dense top-3 adds +2.38 pts pool recall for +0.5 candidates/entity;
  top-5 adds +4.15 pts (India +8.4, US +1.3) for +1.7 candidates/entity (pool recall 93.34% -> 97.49%); top-10 +5.24 pts for +6.2.
- 04:10 started `encode test` then `search test 10` (GPU, about 1 h). In parallel: val pools with dense cap 3 and cap 5, LightGBM v3 fits (CPU).
- 04:40 v3 LightGBM (with dense features) on val holdout: cap 5 F0.5 0.9618 (P 0.985, R 0.920), cap 3 0.9563. Val pool recall 97.59% (cap 5). Better than submission #2's whole pipeline (0.9589) without XLM-R. Using cap 5.
- 04:47 test encoding crashed hard (Windows APPCRASH in python310.dll) at 97.6% of the India documents. US, France and India-query embeddings verified complete.
- 04:49 added tokenizer-parallelism off, `FILL_MISSING=1` repair (encodes only unwritten rows) and auto-retry; running fill for India documents, then `search test 10`.
- 05:20 test dense search finished (17.3M rows, all 1,732,544 queries). India embeddings repaired (112,637 rows re-encoded). Next: test pool with dense cap 5, XLM-R evaluation on the v3 classifier, LightGBM v3 test scoring.
- 05:22 XLM-R (the old cross-encoder from xlmr_v1) on top of the v3 LightGBM: holdout F0.5 0.9616 -> 0.9742 (P 0.992, R 0.939). Blend weights refit in runs/xlmr_v3/blend.json. Training a second cross-encoder on the v3 uncertain zone (runs/xlmr_v3b); it is used only if its holdout F0.5 is better.
- 05:31 LightGBM v3 test scoring done (27.5M pairs, 3.19 matches/S1; LightGBM-only fallback in runs/lgb_test_v3/output). Second cross-encoder (xlmr_v3b) did not learn (loss stuck at 0.69), stopped and discarded.
- 05:33 started the XLM-R test stage: cross-encoder from xlmr_v1, blend weights/threshold from runs/xlmr_v3 (v3 classifier), 1,454,419 uncertain pairs (~67 min) -> runs/xlmr_test_v3/output/.
- 05:37 FALLBACK READY and validated (official validator + --check-ids PASS): runs/lgb_test_v3/output/matching_results.tsv (LightGBM v3 only; val holdout F0.5 0.9618). The full pipeline file (LightGBM v3 + XLM-R, val holdout 0.9742) is expected at runs/xlmr_test_v3/output/matching_results.tsv around 06:45.
- 05:38 the XLM-R test stage hit the same intermittent tokenizer ValueError within minutes (log kept as runs/xlmr_test_v3/logs/test_attempt0_tokenizer_error.out). Hardened xlmr_cross.py: tokenizer parallelism off, retry + manual-pad fallback, and 200K-pair block checkpoints (runs/xlmr_test_v3/ckpt) so a crash resumes. Relaunched with up to 3 automatic attempts.
- 06:03 the laptop blue-screened (Windows Kernel-Power 41, BugcheckCode 30 = KMODE_EXCEPTION_NOT_HANDLED) during the XLM-R test stage; back on at 08:00. All artifacts verified intact (checkpoints xl_0 and xl_200000, test pool, LightGBM v3 probabilities, dense files, models, blend.json, fallback and submission #2 files).
- 08:06 resumed the XLM-R test stage from the two saved blocks (batch size 128 to lower GPU memory use); 1.05M pairs left, about 1 h.

## TODO for later: machine stability (laptop crashes when left running)
Windows records (System log, checked 27 Sep 08:05):
- 25 Sep 17:11  bugcheck 0x9F DRIVER_POWER_STATE_FAILURE
- 25 Sep 22:56  sudden death with no bugcheck (hard freeze or power cut)
- 27 Sep 06:03  bugcheck 0x1E KMODE_EXCEPTION_NOT_HANDLED, exception 0xC0000096 (privileged instruction in kernel mode)
- Crash dumps: C:\Windows\Minidump\092526-24312-01.dmp and 092726-21171-01.dmp (need WinDbg to decode).
- Power plan is Balanced with sleep/hibernate/display-off all "never" on AC, so idle sleep is not the trigger.
- PCI Express link state on AC is "Maximum power savings" (2): a known trigger for 0x9F on gaming laptops.
- GPU under load: 71-75 C, 86-99 W, no throttling. CPU temperature not measured. NVIDIA driver 581.80.
Things to try (none applied yet, all reversible):
1. powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0   then   powercfg /setactive SCHEME_CURRENT
2. Run Windows Memory Diagnostic (mdsched); reset any BIOS/Legion Toolkit overclock or boost to default.
3. Update BIOS and NVIDIA driver (Lenovo Vantage). Keep vents clear.
4. Optional: a scheduled task at startup that resumes the job (xlmr_cross.py test resumes from runs/<RUN_ID>/ckpt; s4dense.py encode supports FILL_MISSING=1).
- 08:19 XLM-R test stage finished after the resume (8 blocks; 1,454,419 uncertain pairs). FINAL FILE: runs/xlmr_test_v3/output/matching_results.tsv, validated (official validator + --check-ids PASS): 1,732,544 rows, 102,701 empty, 5,597,394 predicted pairs (3.23 per S1; India 3.21, US 3.28, France 3.17).
  Checkpoint verification: all 8 blocks re-scored (2,500 pairs each) with correlation 1.0000, mean abs diff 0.001-0.003, 0 sign flips, 0 zero values; the two pre-crash blocks are intact.
  vs submission #2 (0.941): keeps 98.1% of its pairs, drops 101,495, adds 380,530 (India +252K/-40K, US +39K/-33K, France +89K/-29K). Val holdout F0.5 of this pipeline: 0.9742 (LightGBM v3 alone 0.9618). Expected leaderboard about 0.957 (range 0.95-0.965). NOT YET UPLOADED.
- 08:34 UPLOADED runs/xlmr_test_v3/output/matching_results.tsv -> **leaderboard F0.5 = 0.964** (submission #2: 0.941, #1: 0.879). Val holdout 0.9742, so the val-to-test gap is 1.0 point (was 1.8 and 2.5).
- 08:52 diagnosis of the 0.964 pipeline on the val holdout (runs/diag_v3): 80% of entities perfect; loss 2.58 pts = recall ~75% (missed some 1.37, predicted nothing 0.42, both 0.15), false pairs only 0.84% of predictions (0.47 + 0.17 singleton). True pairs missed: 2.5% never in the pool, 3.6% in the pool but rejected (1,902 of them at prob 0.5-0.75). Expected-F0.5 decoding does NOT help (0.9734 vs 0.9742): the threshold is near-optimal.
  In the fixable band (0.2-0.75): 68% have an EMPTY S2/3 address (name equal or near-equal, name unique among S1), or an identical full address with an unrelated alias name. Missing signal: how unique the S1 name/address is among S1 entities.
- 08:57 added FEATSET=v4 (s1_name_cnt, s1_addr_cnt, s1_name_idf; reference = train+val S1 for val, test S1 for test); refitting on the v3 pool (runs/fit_v4_cap5) then XLM-R blend eval (runs/xlmr_v4). GATE: only go to test if holdout F0.5 (blend) > 0.9742 + 0.002.
- 08:59 v4 (uniqueness features) on the val holdout: LightGBM alone 0.9645 (v3 0.9618); + XLM-R blend 0.9771 (v3 0.9742), P 0.993 R 0.945. GATE PASSED (+0.0029 > +0.002). Sibling propagation tested and rejected (out-of-fold F0.5 unchanged). Expected-F0.5 decoding rejected (0.9734 vs 0.9742).
- 09:01 running LightGBM v4 on the test pool (runs/lgb_test_v4), then the XLM-R stage (runs/xlmr_test_v4) reusing the 1,454,419 cached scores from runs/xlmr_test_v3/xlmr_scores.parquet and scoring only the new uncertain pairs.
- 09:05 (user asked for generic, non-dataset-specific changes and big gains only, 3 submissions left) Diagnosis says the largest remaining lever is training-data scale for the neural parts: only the val split (150K fit entities) was used; the train split has 1.99M labeled S1 (6.87M pairs).
  Plan round 2 (runs/dense_v2): mine hard negatives for 1M train anchors with the round-1 encoder, train a NEW bi-encoder from the BASE model on train-split anchors only (so val/test entities are never seen and the LightGBM's dense features become honest), re-encode val+test, rebuild pools, refit LightGBM, blend, test. Fallback: v4 file (runs/xlmr_test_v4/output).
- 09:10 v4 LightGBM test scoring done (3.23 matches/S1). XLM-R v4 stage: 1,463,788 uncertain pairs, 1,187,504 reused from the v3 cache, 276,284 to score.
- 09:12 mining smoke test (20K anchors) passed: 12.4 hard negatives per anchor, 0 true positives among negatives. Round-2 chain queued (mine 1M -> train2), starts when the v4 XLM-R stage ends.
- 09:15 outside opinions (ChatGPT, Gemini) checked against our data: entity-level decision layer 0.9744 vs 0.9743 (no gain); expected-F0.5 0.9734; Hungarian one-to-one would be wrong (an S1 has ~3.7 matches, only record-side exclusivity applies and our rule is already optimal for it); val is PESSIMISTIC on the outside-owner false pairs (689 of 1,211; upper bound 0.9783), so it does not explain the LB gap.
- 09:17 v4 test file finished and validated: runs/xlmr_test_v4/output (3.26 matches/S1; 99.5% identical to #3, +84,419 / -27,021 pairs). NOT submitted.
- 09:18 France pair inspection (unlabeled, model outputs only; no hand-labeling used for any decision): many accepted pairs at 0.75-0.90 have the same house number and city but a different street. Missing feature class: IDF-weighted (rare-token-aware) overlap for every pair -> FEATSET=v5 (addr/name cosine and coverage of S1 side / record side). Small-sample AUC of addr_cos alone 0.884.
- 09:24 fitting v5 on the v3 pool (runs/fit_v5_cap5). Gate: LightGBM-only holdout F0.5 > v4's 0.9645 by >= 0.002. Round-2 mining (1M train anchors) running on the GPU.
- 09:42 v5 (IDF token overlap) holdout: LightGBM alone 0.9710 (v4 0.9645), + XLM-R blend (xlmr_v5) 0.9784, P 0.994 R 0.945. GATE PASSED. Round-2 mining still running; v5 test file (runs/xlmr_test_v5/output) validated but held back, not submitted (user: only spend uploads on results clearing 0.98).
- 10:11 GATE 1 (round-2 bi-encoder vs round-1, holdout candidate-pool ranking, never touching test/val entities): round-2 LOST on AUC (0.986 vs 0.990), rank-1 (0.276 vs 0.278) and rank-3 (0.724 vs 0.733); only rank-5 was marginally better (0.940 vs 0.937). More train-split data did not improve the bi-encoder. Round 2 discarded; GPU freed for the cross-encoder retrain instead.
- 10:18-12:23 xlmr_big: continue training xlm-roberta-base FROM xlmr_v1 (never from scratch) on 1.18M pairs built from the same 1M mined train-split anchors (2 positives + 2 hard negatives each), checkpoint every 1000 steps. Hit the known intermittent python310.dll access-violation crash once, before the first checkpoint (lost ~4 min, not the run); restarted with checkpointing every 1000 steps and it completed cleanly, 5,989s.
- 12:30 xlmr_big holdout eval: AUC on the uncertain zone rose 0.950 -> 0.964, but F0.5 barely moved: blend 0.9784 -> 0.9799 (+0.0015). Full threshold sweep on the holdout confirms the deployed threshold is already within 0.0001 of the true holdout optimum -- not a threshold problem.
- 13:00 root-cause diagnosis (runs/xlmr_big_eval/holdout_zone_scores.parquet): split the uncertain-zone AUC by whether the record has an address -- has-address 0.984, EMPTY-address only 0.719 (LightGBM alone is BETTER there, 0.818). The cross-encoder has nothing to read without address text, so its AUC gain is concentrated where the problem was already mostly solved. Of the 3,838 remaining in-pool false negatives, ~50-68% have an empty address; borderline (near-threshold) false negatives are >= 68% empty-address, i.e. there IS residual name-only signal, it just isn't confidently enough weighted.
- 13:10 FEATSET=v6: instead of a fixed blend weight, feed the cross-encoder's own score into LightGBM as a feature (NaN outside the zone; a tree, unlike a single logistic blend, can learn to trust it only when there's an address to check) plus two features isolating the no-address case (name_conf_no_addr = name_jw * addr_i_empty, name_idf_no_addr = s1_name_idf * addr_i_empty). Also checked and rejected: an unsupervised (Fellegi-Sunter/EM) threshold for France -- validated against US/India (where we have labels) and it was off by 0.22 F0.5 on India, so not trustworthy enough to apply to an unlabeled country.
- 13:15 fit_v6 holdout: 0.9805 (single model, no separate blend step) -- beats the xlmr_big blend's 0.9799. False negatives dropped 3,838 -> 3,491 (-9%); almost the entire net recovery came from the empty-address population specifically, confirming the diagnosis rather than moving the number by coincidence.
- 13:42-13:58 scored the full test uncertain zone (1,162,365 pairs) with xlmr_big (runs/xlmr_big_test_zone; lost ~35 min early on to an untracked duplicate process fighting the same job for the GPU, before it was caught and killed -- no data lost, just time), then ran the v6 classifier over the full 27.5M-pair test pool (runs/test_v6).
- 14:11 test_v6 validated (PASS): 1,732,544 rows, 94.2% non-empty, 3.30 matches/S1; overlaps 96.0% with submission #3 and 97.6% with the unsubmitted v5 file -- a real but incremental change, not a wild swing.
- UPLOADED runs/test_v6/output/matching_results.tsv -> **leaderboard F0.5 = 0.971** (submission #3: 0.964). Val holdout 0.9805, so the val-to-test gap (1.49x loss) continued the widening trend from submissions 1-3 (1.26x, 1.44x, 1.40x): each point of val improvement is buying less leaderboard improvement, consistent with the remaining loss being increasingly the hard, ambiguous cases (ambiguous aliases, empty addresses, France's unlabeled uncertainty) rather than mechanically fixable gaps.
