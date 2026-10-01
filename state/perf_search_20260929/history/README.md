# LB-history lens (2026-09-29, read-only)
Sources: HANDOFF.md sections 1, 7, 18, 19, 22, 23, 24, 33, 41, 43; experiments/submission_log.csv.
lb_history.csv: one row per submitted/held candidate with base, single change, public LB, delta, local evidence and verdict.
Key transfer facts:
- 6-anchor calibration (HANDOFF L810-818): LB = 0.298*local6bba + 0.668, Pearson +0.994; all-12 +0.836; 44b6 -0.590; visible-4 +0.954 (tuned against). Slope ~0.3 => +0.01 local ~ +0.003 LB. Matched 7/7 through C015, then missed C017 by -0.003 (jump-heavy movies over-represented; HANDOFF L943).
- 97-movie aggregate ordered C022>C020~C021>C018>C017 correctly (L945) but failed on C025/C028/C029 (+0.0018/+0.0045/best-local -> -0.001/-0.003/-0.004; L1021-1025).
- Head lane: local e2e ranks heads opposite to LB (s075 locally -0.0045, LB +0.001; L967, L1014); pair-level error also fails (v4 best pairs, worst e2e, LB -0.001).
- Only LB-positive levers: V1284 head (+0.006/+0.007), full head strength (+0.004 vs x0.5), readmit ON (+0.001), every-pair stabilization (+0.001), ILP-edge restore (+0.001), x138 head / 2-head mean (+0.001), C004 strict division + tight 5.2 (+0.002).
