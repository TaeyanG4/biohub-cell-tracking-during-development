# C039 final review

Completed all13 jobs at2026-09-27 16:08:49KST; reviewed on the scheduled40-minute wake. All return codes0. All37 pinned source/model/notebook inputs match; exact original/off two-embryo smoke passed under FP32/math-SDPA; all22 division-off controls reproduce C023. Post-completion output manifest records1600 files (not a launch-time baseline).

| Fixed arm | Official22 delta vs C023 | Heldout12 | Confirm10 | 44b6 | 6bba |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary_max | -0.002212135 | -0.003903105 | -0.000389213 | -0.013697318 | -0.000797308 |
| calibrated_max | -0.002212135 | -0.003903105 | -0.000389213 | -0.013697318 | -0.000797308 |
| repaired public_division | -0.006702089 | -0.006049362 | -0.007674369 | -0.014809186 | -0.005260891 |

Both detector arms have adjusted-edge delta -0.001170468; corrected division -0.000944513. All lose total and adjusted-edge score in both splits and embryos. Division counts go from C023 TP5/FP6/FN19 to TP6/FP31/FN18: one additional recovered division accompanies25 additional counted false positives. Detector arms yield TP5/FP8/FN19.

The detector arms were separately inferred. Comparing all22 cached NPZs shows only low_score differs in each movie; all other stored arrays are exactly equal, and official score tables match. Thus extra calibration changed recorded low-detection confidence but provided no scored benefit here. This is not a proof of equal voxel logits or universal algorithm equivalence. Smoke also confirms active detector output differs from original/off. The V6 missing-global repair passed actual branch execution and full22 replay without swallowed fallback, but repairing execution did not improve accuracy under frozen C023 settings.

Decision: close these three fixed arms without extension75, Kaggle push, T4 run or submission. No consistent benefit supports spending more compute/slots on these arms or arbitrary combinations. This closes the registered finite C038/C039 batch; retain all artifacts and C037/C038 conditional complementarity evidence. No blanket claim about all Transformer, appearance, calibration or division approaches; no claim that this reproduces the fullV6 or validates0.965+. C023/C024 and final picks remain unchanged at best public0.954. Shared submission slots used:0. Stop the scheduled monitor after recording this review.
