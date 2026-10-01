# Division audit, 2026-09-29

Read-only source/saved-array audit. No model fit, inference queue, graph edit, or remote operation was launched.

## Most concrete untested input: temporal nucleus size

The saved public `zhincez__a-dividing-nucleus-gets-smaller-not-dimmer` notebook measures a fixed half-height volume: in an 11-by-11-by-11 voxel box, count voxels above background p10 + 0.5*(peak-background). It follows the parent and daughters along tracks and normalizes each curve to t-6. Its authored narrative reports a daughter-volume drop strongest at t+3 (~0.27 below ordinary controls), with peak brightness broadly unchanged. The saved notebook has zero cell outputs; this is a reported hypothesis, not verified predictive precision.

This is absent from existing production candidate features. `src/division_scorer_stage.py` measures peak/mean in a 3-by-7-by-7 voxel window at P, P's predecessor, and the immediate daughters; it does not measure half-height volume or t+3. C049 consumes only t-1,t,t+1 in a fixed parent-centered crop. Therefore a trajectory-following t+3 size feature is a distinct ingredient, rather than retraining the same closed CNN/MLP recipe.

The concurrent public audit's suggestion of an AUC check “on existing crops” cannot measure this t+3 mechanism: the stored C031/C049 crops end at t+1 and stay centered at the parent. Raw frames plus predicted trajectories are necessary. A GT-trajectory EDA would characterize the reported biology, but would not certify a deployable candidate classifier.

### Reusable data and exact bounds

- C050 `known_labels.npz` retains exactly 221,133 strict-positive/known-negative candidate rows, 50,853 distinct predicted parents, and 48 original features. Whole-embryo partitions: 44b6 has 6,732 parents / 18 positive candidates; 6bba has 44,121 / 76. Unknown label rows remain excluded.
- All 94 positive parents and their D1 nodes are non-synthetic. Three positive D2 nodes are synthetic; reconstruct the real pre-division-stage graphs, do not silently drop them.
- 87/94 positive parents have at least three ancestors (p_track_len >= 3); 80/94 have at least six. 89/94 D2 branches have at least three descendants (d2_track_len_after >= 3). This is broad enough for a temporal measurement but complete two-branch coverage still needs auditing.
- The five original candidate CSVs total about 6.19 GB. They retain P,D1,D2,t; the compact C050 array retains P but drops D1,D2,t. A streaming metadata pass is needed. Do not read these huge CSVs into a dataframe merely to recover four columns.
- `division_candidates_local.py` already captures exactly the graph after the existing safe-division stage and before C016 additions. Reuse its graph capture and its `MetricLabeller`; do not implement a new fork labeler or competition scorer.
- `division_scorer_train.py` already provides `fit`, `predict`, `per_parent`, and `average_precision`. C050's recipe (60 epochs, hidden32, seed0, capped pos_weight100, fixed0.9) can be reused unchanged, with new measured columns. Saved C050 predictions provide the original-feature comparator.

### Bounded proposed study

1. Hash and stream original C050 metadata, asserting exact label/order agreement with known_labels.npz. Recreate the same predicted pre-stage graphs using existing replay/capture. Feature construction may read predicted graph IDs and raw images only; GT is used only by the existing saved label contract.
2. Reuse the public fixed half-height `probe` unchanged. Measure parent trajectory t-3,t-2,t-1 and each predicted daughter's t+1,t+2,t+3. Cache probes by (movie,node), frame reads by frame; use actual predicted trajectories, never GT trajectories. Record crop-boundary and unavailable-track flags instead of silently treating missing size as biological evidence.
3. Prespecify a compact feature block: mean daughter-to-parent volume ratio at t+1 and t+3, the corresponding peak ratio (brightness control), daughter volume asymmetry, and coverage indicators. No radius/lag/segmentation threshold sweep.
4. Keep both original whole-embryo train/test folds and the existing learner/threshold. Report exact per-parent TP/FP/FN for the fixed operating point and descriptive AP. Do not choose a deployment threshold from evaluation. Require roughly precision >=0.3 and restricted recall >=0.3 in BOTH directions before any graph-stage integration. A diagnostic pass is not an official metric gain.
5. Only after passing the classifier gate, build the same fixed measurement on C023 candidates and require actual official22/97 replay, portable/T4 parity, and scientific review. C012 historical candidate data cannot itself establish C023 improvement.

Grounded cost: C050's original CSV parse took 73.65 seconds and its two CPU fits took 6.97/18.56 seconds (`status.json`). New extraction dominates. At most ~9,700 movie frames for 97 movies (about 81 GB uncompressed) need decompression once; repeated per-candidate frame decompression is unacceptable. Measure a two-movie control before giving a full extraction ETA. This is a bounded, different experiment, not a verified +0.002 solution.

## Reverse-direction supervision: real mismatch, limited direct evidence

`src/c037_transformer_study.py:231-241` supervises the forward logits only. Production `predict_unet_transformer.py:702-746` also executes the Transformer with swapped source/target features and fuses calibrated reverse probabilities harmonically (weight 0.2 in the base notebook). Both C052 deployment models inherit the forward-only objective.

However saved C052 whole-embryo daughter rows do NOT support a claim that reverse fusion is the main division bottleneck. At the existing admission threshold 0.48:

| Embryo | Daughter annotations | Detected pair | Primary >=0.48 | Primary >=0.48, fused <0.48 | Primary <0.48, fused >=0.48 |
|---|---:|---:|---:|---:|---:|
|44b6|52|50|33|1|2|
|6bba|250|246|172|6|10|

There are 7 downward threshold crossings and 12 upward crossings after full fusion. Full fusion includes secondary-model blending, so this is not an isolated reverse ablation. Six of the seven downward crossings remain absent at final output; one is recovered later. Removing fusion is not justified.

C052 division supervision also was not nearly absent inside the sampled division batches: the actual weighted-CE share assigned to daughter labels averages 0.6721 for train44b6 and 0.3378 for train6bba on the 300 division-context batches. Across all 600 alternating batches that is about 0.336 and 0.169. Thus “300 division batches still gave almost no division gradient” is not an accurate diagnosis.

One correction to the concurrent `state/perf_search_20260929/closures/closure_ledger.md`: actual C052 label counts are 39 + 246 = 285 daughter targets, exactly the prior proposed detector-covered/known-parent ceiling of 285. Raising 39/246 “toward 285” is not a missing-data mechanism; those two numbers already sum to 285. The remaining meaningful objective issue is competition against production candidates, not a missing label-count gap.

Recommendation: prioritize a topology-justified full-parent-competition objective audit or this genuinely new temporal-size feature over reverse-only retraining. Full-parent competition must explicitly resolve whether unmatched source nodes are valid edge negatives conditional on a target with one known parent; do not quietly override the prior unknown-negative restriction. A conservative mask can exclude unmatched sources in the true parent's 7 um ambiguity region, while retaining structurally incompatible sources outside it. Reverse-only fitting is lower priority unless a direction-specific packet audit shows a substantial known-parent ranking failure.
