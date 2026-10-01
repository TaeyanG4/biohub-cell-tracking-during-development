# Fixed equal parent-probability mixture: closed at packet gate

The one prescribed 50:50 mixture fails the packet gate in both embryos. Do not
advance this mixture to a full-movie pilot, tune weights/temperatures, or integrate
it into a submission candidate from this evidence.

`probability_mix.py` reuses the exact193 packet diagnostic and published Hengck
API. For each target it softmax-normalizes each model across ALL source
candidates, then computes 0.5 * teacher probability + 0.5 * Hengck probability.
No GT mask participates in probability normalization or combination. The existing
C057 identity ambiguity mask is applied only when reporting known-target ranking.
No parameters were fitted and no alternative weights or thresholds were tried.

All193 teacher logit probes had zero error. The teacher-duplicate mean was
bitwise identical to the teacher probabilities, and its masked top1 and true
parent rank exactly matched the teacher logits at every one of1834 targets.
Coordinate roundtrip and published own-feature sampling controls remained exact.
Float64 probability columns sum to1 within1e-12. An initially over-strict2e-15
sum assertion failed on floating-point accumulation and was corrected to1e-12;
this did not change probabilities, model predictions, or the prescribed recipe.
Runtime was31.96seconds, FP32 model inference with TF32 off/math SDPA, batch2.

| Embryo / actual link type | Known links | Teacher correct | Mixture correct | Rescued | Harmed |
|---|---:|---:|---:|---:|---:|
| 44b6 ordinary | 151 | 150 | 150 | 0 | 0 |
| 44b6 daughters | 39 | 32 | 31 | 1 | 2 |
| 6bba ordinary | 1,398 | 1,385 | 1,384 | 2 | 3 |
| 6bba daughters | 246 | 215 | 205 | 7 | 17 |
| Total | 1,834 | 1,782 | 1,770 | 10 | 22 |

The daughter subset alone loses11 correct links (8 rescued /19 harmed). Both
embryos regress:44b6 loses1 and6bba loses11. These are conditional known-link
ranking counts, not official graph scores or independent LB estimates. Previous
single-model evidence remains preserved. This result closes the fixed equal
mixture and the proposed immediate full-movie pilot; it does not establish that
every future independently justified use of this model must fail.

Artifacts are separate from previous evidence:
`probe_result_probability_mix_193.json`,
`probe_rows_probability_mix_193.csv`, `probability_mix_193.log`.
