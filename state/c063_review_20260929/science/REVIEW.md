# C063 independent saved-result review

**Close the fixed C063 component recipe without graph integration or submission.**
Both opposite-embryo overall and good-point gates fail. Large-error tails
improve in several movies, but that benefit is outweighed by damage to the
much larger well-localized population. No decoder, strength or epoch sweep is
supported by this result.

## Integrity and analysis-only failure

`recount.py` completed using existing files and CPU arithmetic only. It
independently rehashed all 2,302 registered plan inputs and 2,509 distinct
evidence files in total. All 22 feature/label/target/provenance outputs and
their input manifests passed. Both shortened-prefix proofs still match the
full original production fields/cubes and original official-zero references.
Original pair IDs, times, GT identities, coordinates, targets and feature
eligibility reproduce exactly.

All 24 compute jobs succeeded: 20 remaining extractions, two fixed fits and
two saved prediction passes. Only the final analysis assertion failed. The
prediction producer computed residuals from FP32 shifts. CSV reload promoted
their decimal representations to FP64; using these directly produces up to
1.96e-7um discrepancy against saved residuals, exceeding the 1e-9 assertion.
Restoring the shifts' original FP32 dtype reproduces every saved 3D, abs-z and
signed residual within **8.88e-16um**. This is an analysis serialization issue,
not missing fits or prediction corruption, and it does not change the verdict.

Each checkpoint contains exactly 1,200 steps and 38,400 sampled examples.
The 44b6 source has six movies/1,159 eligible original pairs; 6bba has
16 movies/7,954 pairs. Independent seed-6301 sampling reconstruction matches
the complete sequence hash, every per-pair draw count and all movie step and
sample counts. Only the stated source embryo was sampled. Checkpoint/source
hashes and recorded exact reload proofs pass; checkpoint tensors are finite.
Loss decreased from 7.193686 to 4.113752/4.320983, but lower training loss does
not establish better localization under the fixed expectation decoder.

Each source prediction retains all **16,931** original pairs. Exactly 9,113
receive proposals; all 7,818 excluded points retain exact zero shift and
unchanged residuals. No rematching or fitted validation selection occurred.
Independent original-all-node nearest-neighbor calculations reproduce all
saved ownership/out-of-image diagnostics.

## Opposite-embryo results

Errors are micrometres and include all original pairs, with excluded no-ops.

| Evaluation embryo | Stratum | n | Mean 3D before → after | Mean abs-z before → after |
|---|---|---:|---:|---:|
| 44b6 | All original | 1,590 | 1.660920 → 1.988720 | 1.035299 → 1.240044 |
| 44b6 | Good ≤2.5um | 1,334 | 1.222710 → 1.630660 | 0.711394 → 0.983266 |
| 44b6 | 3D tail >3.5um | 134 | 4.738667 → 4.555779 | 3.347015 → 3.161925 |
| 44b6 | Axial tail >3.5um | 35 | 5.874752 → 5.930282 | 5.246429 → 5.206724 |
| 6bba | All original | 15,341 | 1.793354 → 1.958010 | 1.087747 → 1.190270 |
| 6bba | Good ≤2.5um | 12,707 | 1.371646 → 1.582025 | 0.756552 → 0.900724 |
| 6bba | 3D tail >3.5um | 1,142 | 4.733828 → 4.593197 | 3.173161 → 2.992470 |
| 6bba | Axial tail >3.5um | 319 | 5.569666 → 5.447174 | 5.134796 → 4.872942 |

Pooled opposite-embryo 3D error worsens **1.780917 → 1.960894um**;
abs-z worsens **1.082821 → 1.194944um**. Of the 9,113 eligible proposals,
4,030 improve 3D error and 5,083 worsen it; abs-z improves for 3,063 and
worsens for 6,050. Overall 3D error wins in four movies and loses in 18.
There are nine new original-node ownership conflicts, zero outside-image
proposals, zero uniform fallbacks and zero radial projections.

The prospective multiple-movie tail tests do pass: 3D-tail joint 3D/abs-z wins
in 5/6 44b6 and 12/16 6bba movies; axial-tail wins in three and nine movies.
However, 44b6 axial-tail mean 3D also worsens, and both embryos fail the
overall and good-point requirements. Neither embryo passes the complete gate.

## Source-fit behavior and interpretation

This is not only a transfer failure. Source-fit all-original 3D error also
worsens: 44b6 **1.660920 → 2.010174um**, 6bba **1.793354 → 1.906577um**.
Both source-fit good-point and abs-z means worsen. Thus the fixed learner and
decoder learn some tail signal while failing to preserve original accurate
positions, even on their own fitting embryo. The records do not isolate a
single causal explanation such as multimodality or insufficient fit; that
would require a new justified protocol, not post-hoc decoder/epoch selection.

These are paired component errors with a public backbone that saw both
embryos. No graph was changed or officially rescored by this study, and no
competition-score effect is claimed. Preserve C023/C024 and the failed study
sources. Recount tables and exact gates are in `stratum_recount.csv`,
`movie_recount.csv` and `evidence.json` alongside this review.
