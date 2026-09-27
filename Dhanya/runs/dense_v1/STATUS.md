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
