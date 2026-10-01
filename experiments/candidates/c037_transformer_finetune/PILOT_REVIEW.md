# C037 pilot review, 2026-09-27 11:25 KST

All 18 queue jobs completed successfully at 11:08 KST. All three fixed arms received full inference and existing-harness official replay on all 22 movies. C037 remains open for complementary-error review after C038; no Kaggle operation or extension75 run has occurred.

| Arm | All22 delta | Heldout12 delta | Confirm10 delta | Test44b6 delta | Test6bba delta |
|---|---:|---:|---:|---:|---:|
| early150 | +0.000356915 | +0.002948628 | -0.001954538 | +0.011673799 | -0.000849542 |
| late600 | +0.000148301 | +0.001271296 | -0.000661069 | +0.008183120 | -0.000679161 |
| late600_blend25 | -0.000275960 | +0.000037475 | -0.000555360 | +0.001575730 | -0.000474876 |

These are official-formula local deltas against matched C023 FP32 caches, not leaderboard gains. All22 baseline is 0.945986173. The early150 gain is concentrated: 5 movie edge-score wins versus 17 losses; the two positive44b6 movies are 44b6_12dfb391 and 44b6_267148e4. Late600 wins12/loses10, but its adjusted-edge delta is -0.000426412; removing one division false positive lifts the total slightly above baseline. Blend25 wins10/loses11/ties1. Every arm loses on confirm10 and on6bba, so none currently warrants deployment or submission on its own. This judgment concerns inconsistent generalization, not the size of a positive effect.

## Execution and supervision

- Verified all13 pinned source hashes, all4492 feature/label hashes for2246 usable windows, and all4 saved checkpoint hashes; see review_integrity.json. No mismatches.
- Original-vs-capture four-frame smoke passed both embryos: exact output and lowdetections, cached logit maximum error0. This is a two-movie smoke, not a fresh22-movie off-arm evaluation. Baseline22 comes from the matched C032 C023 FP32 controls.
- Training metadata confirms the original teacher stayed unchanged and sampled cached-forward parity passed on470/469 unique windows by step600. Unknown detections were not supervised negatives.
- Separate12 movies per embryo:44b6 has1102 usable windows and4717 positive target instances;6bba has1144 windows and8670 positives. Known competing sources per window range2-12 and2-16 respectively. These are repeated temporal instances, not independent biological examples or proof of hard-negative coverage.
- **Both folds have zero annotated division target instances.** The configured division weight2 therefore has no effect in this pilot. A change in division FP after inference is an indirect graph effect, not evidence of supervised division learning.
- FP32/TF32-off/math-SDPA policy on4070 Ti SUPER; no actual T4 check has run. Cross-embryo models cannot be deployed by hidden embryo-prefix routing.

## Follow-up decision

C038 launched in the background at11:24:39 KST (PID19048), after verifying the C037 process and related inference/replay jobs were idle. It tests off/appearance/agreement on all22 using GT-free candidates and final C023 edges. Its prepared notebooks remain local-only and must not be pushed.

Retain all C037 models and raw outputs. At C038 completion verify22 off-controls, inspect proposed versus changed edges and actual signed graph deltas, then assess complementary repairs/harms before choosing any fixed C037+C038 replay or matched extension75. C038's lone diagnostic recovery is in44b6_12dfb391, also a C037 gain movie: overlap must be checked at the edge level; gains must not simply be added. No arbitrary threshold/checkpoint sweep or GT-source whitelist. A justified pooled/fixed deployment model still requires its own real replay, portable notebook and actual T4 visible4 agreement. C023/C024 and final picks remain unchanged.
