# C023 edge / C052 objective audit, 2026-09-29

The actionable new mechanism is identity-masked conditional parent competition. C037/C052 cross-entropy normalizes only over annotated source detections. This leaves most actual parent competitors outside the supervised objective; this differs from rerunning C052 or changing thresholds.

All C052 manifest source appearances: 44b6 4,869 / 372,194 known (1.31%); 6bba 10,270 / 157,968 (6.50%). Actual prediction normalizes over all sources. Unknown cells are not false detections. But for a known target with an annotated, detected unique parent, a source outside the fixed 7 um identity gate cannot be that parent under the existing matching convention. Such an edge is evaluably false under the existing `compute_edge_confusion`, even when the source cell is unannotated.

## Fresh bounded packet evidence

`audit_competition.py` read all 145 retained division packets plus two fixed temporal quartiles from each original ordinary movie (48 packets), 193 total / 1,834 positive target links. It made no model, graph, or training change. Packet and label SHA256 checks pass; teacher logits exactly reproduce every stored probe (maximum error 0). Opposite-embryo C052 models were also read-only evaluated.

Ambiguity rule: retain existing known-source competitors. Add only unmatched sources more than 7 um from the annotated parent, using exactly the original rounded source coordinates and physical voxel scale. Mask all unmatched sources within 7 um, since they may be duplicate/localization alternatives for the same parent. No unmatched target receives a label.

| Embryo | Known positive targets | Teacher correct with known-only competitors | Teacher correct with identity-masked competitors | Mean original CE | Mean masked CE |
|---|---:|---:|---:|---:|---:|
| 44b6 | 190 | 190 | 182 | 0.007061 | 0.290673 |
| 6bba | 1,644 | 1,642 | 1,600 | 0.015009 | 0.225629 |

There are **50 teacher mistakes hidden by the current objective**: 8 on 44b6, 42 on 6bba. 36 concern actual daughter targets (7 / 29). All 50 assigned positive parents are strictly nearer their annotated parent than every alternative predicted source after the exact integer-coordinate convention. This is model-ranking evidence, not 50 recoverable final links or independent biological events. Positive match distances can still be as large as the existing 7 um rule; the public detector/head already saw both embryos.

C052 opposite-fold models retain essentially all these mistakes (masked correct 182 / 1,597). Its known-only supervision was already saturated. Mean number of newly valid conditional parent competitors per target is 367 on 44b6 and 251 on 6bba. Most are easy; the objective focuses their probability mass automatically without mining or a threshold sweep.

## Relationship to existing error budget

Read-only reuse of `state/perf_search_20260929/errors/` (created by another current task): C02397 has 2,774 FN and 2,701 FP with exact official count parity. Only 45 FP join two known but wrong GT nodes; 1,332 attach a known target to an unannotated source, and 1,324 leave a known source toward an unannotated target. The current known-only CE principally supervises the small first class. This motivates the new loss but does not prove all unannotated-endpoint FPs are detector identity errors or safely correctable.

1,385 FNs have both endpoint nodes present, but only 35 were correct ILP links destroyed by postprocess (oracle +0.000478 total) and only 9 have both free endpoints (+0.000123). Thus merely restoring omitted ILP links or adding endpoint gaps cannot plausibly supply +0.002-0.003. Actual known-target competition is a much larger, currently poorly supervised failure class.

The unrelated `errors/SUMMARY.txt` final division-ceiling lines are arithmetically invalid; use `metric/metric_structure_summary.txt` for that term. No such line was used to rank this experiment.

## Implemented helper verification

At the coordinator's request, `src/c057_conditional_parent_loss.py` now implements only label verification, the fixed identity mask, weighted supervised CE and numerical smoke. `smoke_conditional_loss.py` verified all 193 audited packets reproduce original labels exactly (1,834 targets); 483,140 conditional competitor occurrences are admitted and 413 identity-ambiguous occurrences remain masked. Four actual packet smokes cover both embryos and both pools. Known-only control loss maximum difference is 2.8e-8; maximum gradient difference is 1.7e-8. Masked competitor and unlabelled-target supervised gradients are exactly zero; added conditional competitors receive nonzero gradients. A subsequent `py_compile` passed. Helper SHA256 is `e0d1b7e3f485f85ccd798d1235adb8483d7ad02c8bfcd009528e71cd1c56235f`. No model was trained by this audit. Proof: `conditional_loss_smoke.json`. Both audit execution sessions have completed; no process remains owned by this subtask.

## Recommended single experiment

Keep C052 packets, two whole-embryo folds, exact 300 ordinary / 300 division sampling, frozen detector, model, teacher KL, optimizer, seed, 600 steps, candidate threshold and downstream C023 pipeline. Change only supervised parent CE from known sources to the fixed identity-masked candidate set. Exclude identity-ambiguous alternative sources; preserve the original positive labels and weights exactly. Require real-packet known-only numerical loss/gradient parity and zero gradient on masked sources before fitting. Then require all22 frozen-detector / unchanged-control parity, signed edge+division results for both embryos and stage survival; promote only with actual graph evidence. No success probability or expected hidden LB gain is established by this audit.
