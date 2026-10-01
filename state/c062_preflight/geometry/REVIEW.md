# C062 frozen production spatial-feature adapter

2026-09-29. Read-only source/metadata review and CPU coordinate-function audit.
No GPU inference, feature extraction, fitting, queue or candidate-source edits.

Use a passive wrapper of the exact C023 `_v1284_refine` call for the first
executable proof. The input `feature` is already the production primary
32-channel, eight-view averaged dense feature field, and the call carries the
absolute movie and frame. Call the original refinement in `candidate` mode,
copy the requested dense neighborhoods, then return its result unchanged.
This proves access to the correct field without changing any detector, head,
edge or graph decisions. Do not run `v1284_capture_local.py` unchanged: its
`capture` mode returns the unrefined coordinates and disables the C023 head.

## Exact reusable sources

- `tmp/c023_output/tracking_repo/scripts/predict_unet_transformer.py`: immutable
  actual C023 predictor. `load_model` begins line163, `_load_frame` line208,
  frame/normalization/primary eight-view encoding lines333–443,
  `_v1284_refine` hook line636, indexed edge fields lines680–688.
- `tmp/c023_output/tracking_repo/scripts/v1284_coordinate_refinement.py`:
  original `refine` and `index_features`. The latter is the production
  trilinear lookup. Use it for each fractional-centered cube grid; do not
  substitute a differently configured `grid_sample` without exact parity.
- `src/c037_transformer_study.py:engine/smoke`: established namespace setup,
  C023 environment/head, FP32 policy, explicit trilinear monkeypatch, passive
  hook equivalence and saved Transformer-input reproduction.
- `src/run_kaggle_predict_local.py`: existing full C023 inference/ILP runner;
  pass the C023 notebook explicitly, candidate mode, public x138 head and
  `--t4-fp32`.
- `src/c058_localizer_baseline.py` and the unchanged C058 baseline/labels:
  original final integer coordinates, official known-pair IDs, synthetic-node
  provenance and actual original writer/organizer scorer. Use these original
  IDs for real-anchor diagnostics; do not rematch labels after proposed shifts.

The actual sources/checkpoint/config/notebook/head hashes are in `audit.json`.
Primary weights are `artifacts/pilkwang_support50/weights/unet_transformer/
split_0/edge_predictor_best.pth`. The C023 head is
`artifacts/anvithpothula_v1284_head_s075/v1284_head.pt`.

## Coordinate and temporal contract

The feature grid has native downsample `(1,4,4)` and physical spacing
`(1.625,1.625,1.625)` micrometres. Loading uses `raw[t,::1,::4,::4]`:
**strided samples with origin zero**, not XY average pooling. The DeepCenter
pooled-pixel-center discussion does not apply here. Convert native `(z,y,x)`
to feature `(z,y/4,x/4)` with no half-pixel or +1.5 native-XY translation.
All199 metadata shapes equal the actual strided-array ceiling shape, so the
loader's shape-mismatch interpolation branch is not needed for those movies.

Normalization is `(raw-q0.001)/(q0.999-q0.001+1e-6)`, clamped below at zero.
There is **no upper clip at3** as in the C061 raw-crop representation. Preserve
the original movie quantiles and complete frame dimensions.

The temporal encoder takes two real frames and has temporal attention at
coarser levels. At the first-seen detector/head invocation, frame0 uses
window`[0,1]`, index0. Every frame`t>=1` uses window`[t-1,t]`, index1. Using
`[t,t+1]`, a duplicated frame, an independently encoded crop, or temporal
averaging would change the representation. The source-frame field used when
predicting an outgoing edge at`t` may come from a different window than the
first-seen coordinate head: label the chosen context explicitly.

Primary `BIOHUB_EDGE_FEATURE_TTA=1` uses the exact original eight XY transforms
and addition order, then divides by8. Its field is not a single-pass encoder
output or the fused primary/secondary detection logit. No z flips occur.
The model must remain `eval()`, with all weights/BatchNorm buffers unchanged;
FP32, TF32 off, cuDNN benchmark off and math SDPA match the existing controls.

C061 reused native-integer manifests remain fractional on this grid:
5,942/6,364 direct anchors and8,609/9,294 real anchors have fractional XY.
All16 quarter-grid XY phases occur. Preserve each fractional center and
sample integer feature-grid offsets around it. Rounding the center silently
adds a new localization error. `audit.py` executes the actual production
`index_features` definition on an affine field and verifies both fractional
interpolation and integer gather exactly. Native/grid/physical round trips
are exact on all15,658 manifest rows.

For a chosen neighborhood radius, require real full feature-map support.
The production lookup clamps boundary queries; do not silently allow repeated
edge features to masquerade as complete local context. Preserve every
excluded original known pair unchanged in the diagnostic report. Recompute
support for the final selected C062 cube dimensions; the C061 raw-crop support
test is not automatically the same feature-space support test.

## Existing cache limitations and cost

The FP32 C023 edge cache contains coordinate/admission/low-detection arrays
and optional edge probabilities. C037/C052 packets contain per-node64-vectors:
32 sampled feature values plus32 positional features. The historical V1284
cache is224 values (center plus six directional differences). None is a dense
spatial field; a neighborhood cannot be reconstructed from them. Reuse their
unchanged coordinates, graphs, edge-input/logit controls and GT lineage.

Measured historical full-pipeline costs on this workstation:

| Existing completed job | Movies | Seconds |
|---|---:|---:|
| C032 control FP32 heldout |12|792.45|
| C032 control FP32 confirm |10|611.58|
| C032 control FP32 extension |75|4803.68|
| C037 capture |24|1512.80|
| C052 division capture |87|5845.43|

These are approximately64–67seconds/movie including secondary inference,
association and ILP, not a measurement of new dense extraction. They suggest
about3.5hours for all199 full movies before capture IO. The new capture size
and head fit require their own small real benchmark before an ETA.

A cheaper extraction adapter can follow the full-pipeline passive proof:
compile the unchanged primary prefix of `predict_video` (metadata/setup,
real two-frame read, normalization and eight-view encode), stopping immediately
before `secondary_unet_out = None`. Execute only requested first-seen windows,
then copy local fields. Preserve original source statements and hash the
generated adapter. For the union of current C061 direct and real manifests,
there are6,861 requested first-seen windows and10,084 unique source frames;
frame provenance includes the preceding context frame, not merely the label
frame. No secondary model or association is mathematically needed to compute
that primary field. Exact comparison to the full predictor is mandatory
before treating this shortened path as production-equivalent.

## Required concrete zero/control proof

1. On the two usual full movies, run the original and passively instrumented
   C023 predictor with the same frozen weights/config/head. Require bitwise
   equality of returned refined coordinates, all edge probabilities/distances,
   low-detection arrays and admitted/ILP results; do not rely only on totals.
2. Save feature field/cube digests and original `index_features` samples.
   Compare cropped center values and known fractional queries to the same
   live production field. If using a prefix extractor, compare its requested
   first-seen fields or full-field hashes exactly to the passive full runner.
3. An auxiliary zero head must leave original final integer nodes/edges and
   all original paired residuals exactly unchanged. For actual graph use,
   replay the existing original writer and organizer metric with zero changes.
4. Pin primary checkpoint/config, C023 source tree/notebook/head, full movie
   metadata/quantiles, **both frames in each encoder window**, source manifests,
   capture adapter, dtype/TTA/context and source-only learned normalization.
   Keep known-GT identities for evaluation, not as a runtime acceptance mask.

This proves representation fidelity, not localization efficacy. The frozen
public backbone already saw both embryos. Any learned auxiliary head still
needs source-only fits, both opposite-embryo real-anchor/tail/good diagnostics,
and separate graph evidence. Updating coordinates before association requires
fresh indexing, candidate edges, Transformer scores and ILP; stale edge caches
would not test the new geometry correctly.
