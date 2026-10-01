# Structural reset: localize the cell before learning another link score

2026-09-29. Read-only audit; no training, graph changes, candidate restart, or Kaggle operation. `audit_tables.py` recounts retained C023 diagnostic CSVs; input SHA256s and exact counts are in `evidence.json`.

## Recommendation

The strongest distinct structural lead is a **raw-image coordinate localizer**, tested first with C023's final node identities and edges held fixed. The existing frozen-feature coordinate heads have improved the LB before, but do not contain the spatial information or training support needed for the remaining large offsets. This is materially different from another head strength, head ensemble, Transformer loss, detector threshold, or node-budget variant.

This recommendation is a hypothesis supported by the error distribution. A large coordinate residual can also mean that the matcher chose the wrong biological cell, or that the annotation itself is offset. It is not proof that an image model can repair every listed error. The first experiment must discriminate those possibilities.

## Why another pairwise association objective is not the first choice

Fresh recount of the retained 97-movie C023 tables:

| Diagnostic | all97 | 44b6 | 6bba |
|---|---:|---:|---:|
| Missed GT edges | 2,774 | 307 | 2,467 |
| Both endpoint detections exist | 1,385 | 221 | 1,164 |
| Of these, at least one endpoint is >3.5 um from GT | **1,047** | **166** | **881** |
| Both endpoint residuals <=2.5 um | 130 | 27 | 103 |
| High residual but actual GT movement <4 um | 727 | 138 | 589 |
| Correct endpoint-to-endpoint edge existed in ILP | 35 | 4 | 31 |
| Both endpoints are free for a gap connection | 9 | 0 | 9 |

Thus 75.6% of both-present missed edges touch a large localization residual, consistently across embryos. The required connection between the matched predicted nodes has median length **8.30 um**, although its GT edge moves only **2.63 um**. Simply training the association model to prefer these long connections may teach a change of biological identity caused by the spatial matching, rather than better tracking.

Matched GT nodes touching an error have median residual **3.83 um**, compared with **1.72 um** for the other matched nodes. The fraction above 3.5 um is **55.4% versus 6.3%**. Of 1,119 unmatched GT nodes, 792 have a predicted node 7–10 um away. This latter group could contain misplaced centers or a different neighboring nucleus; no recovery claim follows from proximity alone.

Of 2,701 diagnostic false edges, only 45 connect two matched annotated cells with the wrong known association. In 2,153 one-unmatched-end cases, the expected GT partner already matches another predicted node. Sparse annotations therefore make ordinary known-known negative training a weak representation of the errors actually scored. Unknown cells must remain unknown.

These are retained diagnostic matcher counts, not a new official evaluation. The existing output-shaping audit reproduced edge counts with the vendored official evaluator on five crowded movies. Do not reuse the early `errors/SUMMARY.txt` division ceiling arithmetic, which contains an erroneous all-movie division increment; its corrected values appear in `SUMMARY_swap.txt` and the metric audit.

## What information is missing from the current head

`src/v1284_capture_local.py` stores 224 frozen-UNet features: center feature plus six neighbor differences. Its default capture gate is 4 um. `src/v1284_head_train.py` by default trains only offsets <=3 um. The local head architecture is a 224->32->3 MLP with a displacement bound below 2 um. The public x138 head's complete training provenance is unavailable, so do not assume its exact training filter matches ours.

The retained 125,696-pair audit shows x138 s075 predicts roughly 0.7–0.9 um shifts across the entire 0–4 um error range. For 3–4 um offsets, residuals remain 3.16 um on 44b6 and 3.01 um on 6bba. The 2 um bound is not saturated; multiplying its output makes all-pair residuals worse. Repeating scaling or a larger copy of the same MLP does not address the missing spatial information.

The broad image-blob audit is compatible with this lead: many 3.5–7 um offsets lie in the same bright region as their prediction, and error energy is predominantly along z (about 56–58%). However its intensity/half-height measurements cannot identify the correct nuclear center, and are not a segmentation-quality guarantee.

Coordinate learning has previously been a real high-impact mechanism: C012 improved the headless local12 by about 0.0103 and the recorded public score by about 0.006. That historical result is not an estimate of the remaining gain from C023. Later heads with lower pair residuals still lost end-to-end and on the LB (HANDOFF 21/23), so pair regression error alone cannot select a candidate.

## Concrete bounded validation protocol

1. **Freeze the question and baseline.** Use C023 final graph coordinates as crop centers and preserve its exact node IDs, node count and edges. Reuse C055's saved graphs, original image reads/crop conventions, and the existing official matcher/evaluator. A zero correction must reproduce the exact baseline. No new competition scorer or post-processing simulator is needed.
2. **Train only on known point targets.** Match training-embryo predicted nodes to known GT with the existing 7 um one-to-one contract, saving ambiguity and boundary flags. All unmatched nodes are unlabelled, never background negatives. Use a fixed full-resolution anisotropic 3D crop that covers the identity gate (for example 16x64x64 voxels), with a position-preserving local coordinate output. The existing image IO/crop tooling can be reused; the division CNN's final global average pool should not be copied blindly into a localization regressor. Training-only spatial jitter supplies exact translated coordinate targets without inventing new cell labels.
3. **Two whole-embryo fits, one recipe.** Fit on 44b6, apply to every eligible node on 6bba, and vice versa. No target-embryo threshold, prefix deployment routing, model/checkpoint sweep, or manual GT-selected repair list. Include all eligible training residual magnitudes rather than truncating the measured failure group at 3 um. Preserve the original public detector/head, graph and sources.
4. **Measure the causal bottleneck first.** Apply each fixed learner to all real nodes in its opposite-embryo final graphs. Leave synthetic nodes unchanged by an inference-available provenance rule. Convert using the existing integer writer and official evaluator. Report signed TP/FP/FN, total/edge/division score, and correct-to-incorrect as well as incorrect-to-correct matches. Also report errors <=2.5, 2.5–3.5, and >3.5 um for diagnosis only; these GT groups never select runtime behavior.
5. **Falsifiable stop gate.** Stop this recipe if either opposite embryo loses official adjusted-edge score, if the >3.5 um residual group does not improve, or if the apparent improvement is only paired-offset MSE without improved official graph counts. A gain confined to one dense movie is insufficient. On a consistent pilot result, extend unchanged to all97 before considering integration. Any later pre-association use is a separate experiment because it changes feature sampling, probabilities, topology and node selection.
6. **Deployment is separate.** A fixed model for every hidden movie needs its own complete local replay, portable/writer controls and exact T4 source/model/CSV/version verification. The two frozen public detectors have seen both training embryos; the opposite-fold test validates only the new component. Do not promise a hidden score gain or promote based on local offset error.

A two-movie full-node extraction/inference benchmark should establish cost before an unattended launch or ETA. Final97 graphs contain roughly two million nodes; per-node repeated frame decompression would be wasteful. Batch nodes by frame and reuse the existing raw-image IO.

## Lower-priority alternative: decouple association from node admission

C057 provides direct evidence that changing link scores also changes which detections survive: compared with C052 it adds 16,270 final nodes, gains 36 raw edge TPs, and loses adjusted score overall. Its review decomposes raw-edge contribution +0.001770 and node adjustment -0.002402. However 44b6 raw edge itself declines -0.010872, so holding the node count fixed is not a general rescue for C057.

A future long-horizon association model could operate on the fixed baseline node set and jointly choose continuations/division events, rather than injecting independent edge confidence into ILP node selection and then overwriting edges with one-to-one relinking. The data show that almost all missed forks with a matched parent already retain one correct daughter, while the other daughter is often owned by another trajectory (55) or orphaned (43). That supports explicit parent/daughter trajectory identity, but no successful inference-available discriminant is established here. C033 public-weight assignment, C038/C048 appearance swaps and raw second-child probabilities already failed; repeating them or merely re-solving C057 on fewer nodes is not the proposed experiment. Prioritize the coordinate-localization test over this less specified direction.

Inputs consulted: `HANDOFF.md` sections 21/23/43/44; C057 `FINAL_REVIEW.md`; `state/perf_search_20260929/{errors,model_level,detector,metric,pipeline,output_shaping,data,closures}`; `src/v1284_capture_local.py`, `src/v1284_head_train.py`, `src/v1284_patch_variants.py`, C037/C052/C057 learner and production predictor. This audit does not reopen any closed fixed arm.
