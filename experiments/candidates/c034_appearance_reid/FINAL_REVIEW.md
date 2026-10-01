# C034 final review: insufficient negative labels, no submission

Extraction finished 2026-09-27 04:09:26 KST; reviewed at the 05:01 KST follow-up. All 22 movies completed, all source hashes still match, and all recorded C023 controls reproduce saved official edge/division counts, adjusted-edge scores and final node/edge counts. Real short-input equality smoke also passed. No runtime failure or source drift remains.

| Embryo | Movies | Ambiguous source groups | Candidate pairs | Positive | Known negative | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 | 6 | 459 | 996 | 459 | 6 | 531 |
| 6bba | 16 | 1,843 | 3,932 | 1,843 | 10 | 2,079 |
| Total | 22 | 2,302 | 4,928 | 2,302 | 16 | 2,610 |

Known negatives occur in only five movies: 44b6_267148e4 (6), 6bba_07e24132 (1), 6bba_09961292 (2), 6bba_337b1b3a (1), 6bba_3abfe10a (6). These are candidate pairs and may share trajectories; they are not 16 independent tracking errors.

The fixed diagnostic requires at least 20 examples of each class in a training embryo before fitting. Training on 44b6 provides six known negatives; training on 6bba provides ten. Both folds therefore correctly report `insufficient_known_training_pairs`. Neither geometry-only nor geometry+appearance classifier was fitted; no AUC, net recovery comparison, graph-level ReID replay, T4 run or candidate submission occurred.

**Decision: close this supervised diagnostic for insufficient evidence.** The appearance hypothesis is not experimentally disproved, and no official ReID score delta exists. The predeclared compute gate cannot be evaluated, so it does not justify graph integration or a submission. Reclassifying every unmatched detection as a wrong match would hide the sparse-annotation limitation, and was not done. No threshold tuning or expanded candidate radius is used to manufacture negative examples.

Existing pair caches and public-code provenance are retained for a future study with defensible additional supervision. A different unsupervised or synthetic-supervision experiment would require its own explicit hypothesis and validation; this report does not claim it has been tested. The remaining 75 movies were not extracted for C034.

C023/C024 and user final-pick settings are unchanged. No Kaggle push/submission and no shared slots consumed. The C034 follow-up automation is disabled after this review. `analysis.json` and `RESULTS.md` remain unmodified raw diagnostic outputs; this report records the interpretation.
