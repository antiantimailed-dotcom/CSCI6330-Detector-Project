# Milestone 2: pre-registered expectations for the unseen-day test

Written: 2026-11-14, BEFORE the unseen Thu/Fri test (`m2.score_test()`) has been run.
Frozen code/models state: commit `4c0d99f`, tag `m2-pre-test`.
Test protocol: single official model (seed 42), thresholds frozen from benign
validation scores, strict `>`, scored once.

Frozen thresholds
- Autoencoder: 0.0449 (1% FPR target), 0.2654 (0.1%)
- RF-IDS: 0.015 (1%), 0.175 (0.1%)

Decision log
- 2026-11-11: single-seed test scoring (seed 42); test run deferred until after the Milestone 2 submission.
- Nothing (thresholds, features, models, split) will be changed after the test is scored.
- Headline = 1% FPR, frozen thresholds, per-family recall and per-day FPR. 0.1% is a sensitivity check only.

What we already know (validation, seen days Mon-Wed, 1% FPR)
- RF recall 0.9995, AE recall 0.8946. AE misses FTP/SSH-Patator (0%) and about half of GoldenEye.

Expectations (confidence in brackets; edit to match what the team believes)

1. False-positive rate on the new days' benign traffic
   - Both detectors will exceed their 1% target on at least some unseen days, because the
     benign traffic comes from different days than the threshold was calibrated on. [medium]
   - We will treat up to about 2x the target (2%) as mild shift and anything above 5% as a
     serious calibration problem. Per-day FPR is reported, not just the pooled number. [medium]

2. Recall by unseen family (frozen 1% thresholds)
   - DDoS (128,016 flows): both detectors high (RF > 0.9, AE > 0.8), since it resembles the DoS
     traffic seen in training. [medium]
   - PortScan (90,819): no strong prediction; RF and AE could differ widely. [low]
   - Bot (1,953): lower than on the seen DoS families for both; AE below 0.5. [medium-low]
   - Web attacks (Brute Force 1,470, XSS 652): low for both (< 0.5), since they look like
     ordinary HTTP. [medium]
   - Infiltration (36) and SQL Injection (21): too few flows; descriptive only, no prediction.

3. Comparison
   - RF's near-perfect seen-day recall will NOT carry over: its recall on at least one unseen
     family will fall below 0.5. [medium]
   - The AE-vs-RF gap will be narrower on unseen families than on seen ones. [medium-low]
   - Pooled recall is dominated by DDoS + PortScan (about 98% of unseen attacks), so we report
     per-family recall and do not headline the pooled number. [high]

4. Near-duplicates
   - Excluding unseen rows that exactly match a training row (2.5-8.4% per day) changes
     per-family recall by only a few points. [medium]

How we will read the results
- A prediction that fails is reported as such. We do not retune or retrain afterward;
  further analysis uses the saved scores (`m2.reanalyze_saved()`), not new scoring.
