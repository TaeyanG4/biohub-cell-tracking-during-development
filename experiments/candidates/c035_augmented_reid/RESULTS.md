# C035 GT-supervised appearance diagnostic

No modified-graph official score or submission is produced by this study.

| Arm | Test embryo | Rank net vs cost | Conservative fixes / harms | Net | Positive movies | Compute gate |
|---|---|---:|---:|---:|---:|---|
| basic | 44b6 | -34 | 1 / 0 | 1 | 1 | False |
| basic | 6bba | -124 | 1 / 8 | -7 | 1 | False |
| patch_ncc | 44b6 | -8 | 1 / 0 | 1 | 1 | False |
| patch_ncc | 6bba | -46 | 2 / 7 | -5 | 2 | False |
| weak_aug | 44b6 | -26 | 1 / 0 | 1 | 1 | False |
| weak_aug | 6bba | -112 | 5 / 8 | -3 | 3 | False |

Arms warranting graph-integration review: none

- All public detectors already saw both embryos; this is not independent whole-pipeline validation.
- Training excludes all official97 movie IDs but only two embryos exist.
- Distinct GT cells give real negatives, often easier and farther apart than actual relink candidates.
- Unknown C034 targets were never used as negative training labels. Ranking success means retrieval of the matched GT child, not biological truth for every unmatched detection.
- Only sources with an annotated reachable successor are evaluated. Missing detections and assignment conflicts are outside this diagnostic.
- Movie effects are reported; pair counts are not independent samples. No threshold/checkpoint search on the target embryo.
- Ranking and conservative proposals do not execute a valid global graph update. Official replay and actual T4 checks remain mandatory.
- Patch NCC is a fixed, unaligned diagnostic control, not a tested registration/linking algorithm.
