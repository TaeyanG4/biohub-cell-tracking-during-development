# C064 geometry and frozen-model evidence

2026-09-29. A narrowly defined **axial-only information probe** is justified.
The earlier full3D proposal remains blocked by the historical XY convention;
this review does not remove that blocker or reinterpret it as a pipeline bug.
No real heatmap or outcome was inspected while fixing this proposal.

## What was checked

Read AGENTS.md, HANDOFF.md, the prior geometry/deepcenter-contract review,
the actual C023 notebook score/heatmap functions, package manifest, build and
training source, split manifest, and actual best.pt metadata on CPU.
The manifest-pinned checkpoint SHA256 is
`8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0`.
Its keys are config/model_state/optimizer_state/epoch/best_score/history,
epoch2, with no coordinate tag or source revision. The saved config says
epochs50 whereas the snapshot config says1000, consistent with resumed
training and insufficient to bind the early checkpoint to a precise source
revision. The full3D ambiguity therefore remains real.

Actual split_manifest.json lists all71 44b6 movies as training and all128
6bba movies as validation. The prior generic statement that this DeepCenter
trained on both embryos is unsupported. However, best.pt was selected by
6bba validation loss, so6bba is not untouched model validation. Original C023
node detector provenance is a separate issue. These are exploratory signals.

The shipped make_heatmap places an annotation at `[z,y/4,x/4]`; the published
pooled-pixel inverse is `[z,4*y+1.5,4*x+1.5]`. Both give exactly native z.
Only XY differ by0.609375um per axis. C023's production score uses
`round(z),round(y/4),round(x/4)` and a5x5XY maximum. Its heatmap is FP32,
checkpoint-normalized and averages logits under its enabled D4 TTA (8views
on squareXY,4otherwise). No new model or normalizer is needed.

## Distinct, limited question

The retained learned field may contain axial location information discarded
when C023 reduces it to a local maximum score. That is different from C055/56
readmission, raw-brightness snapping, C060/61/63 head fitting, and C036/51
registration. The bounded HANDOFF and source review found no completed exact
axial-location duplicate. This is an evidence question, not a likely score
estimate; absolute-centre and nucleus-identity ambiguity can still defeat it.

The exact pre-outcome rule and gates are in FIXED_PROTOCOL.md, reviewed with
temporal_next_review. It uses the immutable256points/231frames/22movies
already selected for the C060 diagnosis. Every original GT assignment remains
fixed. This tail-enriched sample has128points per embryo; good66/57,
3Dtail58/64, axialtail12/18. It cannot estimate population gain.

Compute the same production5x5XY profile at z−4 through z+4. A unique maximum
sets only z; exact ties and unavailable support are no-ops. XY stay unchanged;
no XY inversion or GT-chosen peak is used. Both boundary maxima are treated
identically. Ownership is diagnostic only. All-original overall/tail/ztail
3D and absolute-z mean improvements, nonworsening good points, and more than
one movie tail improvement in each embryo are mandatory. A pass earns a
separately reviewed full-node/official study; a failure ends the fixed probe.

## Implementation and executed controls

`src/c064_deepcenter_axial_probe.py` exposes dependencies(), controls(out),
benchmark(out), probe(out), analyse(out). Default DEST is
`experiments/candidates/c064_deepcenter_axial`. Output paths are controls.json,
benchmark.json, heatmaps/<stem>/<t>.npy/json, probe/pairs.csv/receipt.json,
and analysis/decision.json plus summary/per_movie.csv. Dependencies currently
enumerate905 real paths, including231 specific raw image chunks, exact model
and notebook, labels, original baselines and GT identity data.

CPU controls passed on all256 original IDs and coordinates; C058 labels and
C060 inputs/selection were rechecked against recorded hashes, and GT residuals
were recovered from actual GEFF IDs. Synthetic9-bin sign/tie/boundary/nonfinite
and zero controls passed. The real C023 score function was AST-extracted
unchanged: its z±1 maximum exactly equals the new per-plane profile's central
three maximum, including fractional pooledXY and Python tie-to-even rounding.
This is arithmetic/source evidence, not measured localization improvement.

Benchmark is fixed to the first sorted requested frame per embryo, without computing
GT efficacy. It saves reusable exact FP32 heatmaps. Its measured seconds
predict the remaining229frame cost. No defensible runtime estimate exists
before that measurement. Root owns registration, launch, PID/ETA/Windows
notifier and heartbeat. No GPU execution was started by this design reviewer.
