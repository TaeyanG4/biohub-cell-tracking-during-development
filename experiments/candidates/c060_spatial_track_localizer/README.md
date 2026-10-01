# C060: conditional spatial localization along a predicted track

One fixed localization experiment, following the user's explicit priority.
C058's flattened vector regressor failed cross-embryo transfer and is closed.
The new22,432-parameter convolutional spatial map uses actual predecessor,
current and successor raw-image crops. See the frozen RECIPE in the driver and
model; no scale, threshold, checkpoint or architecture search.

13x49x49 real-support crops, two valid3x7x7 convolutions,9x37x37 location grid,
7um support and symmetric Gaussian center prior. Known-current-point Gaussian
cross entropy is conditional location supervision, never unknown-cell/background
classification. Stable known source-lineage triples provide1,021/6,255 labels,
including58/371 residuals>3.5um; opposite biological embryo evaluation only.
These labels test persistent position offsets; they do not directly supervise
recovery of the excluded known identity-switch cases.

Each view is centered on its own predicted node, not registered to the current
image by GT. Shared integer jitter and odd-grid reflections transform the label
exactly. At inference all8 inverse-reflected logit maps must choose the same
unique integer mode. The proposed point must remain strictly in its original
node's nearest-center region among ALL current-frame predicted nodes. Otherwise
the original position is retained. Runtime context/guards have no GT inputs.
Geometry cannot guarantee biological identity; all abstentions are reported.

The component pilot freezes final C023 IDs, times, counts and edges. Original
writer and actual vendored official evaluator are reused. Required scientific
evidence: BOTH embryos' original-pair overall, tail and persistent-tail mean
residual decrease; originally good<=2.5um mean and known identity do not worsen;
actual official edge and total do not regress. Passing only supports manual
review of unchanged97 expansion or a separate pre-association integration test.
Frozen edges cannot gain TP from smaller residuals if GT matches stay unchanged.

Only two biological embryos; movie crops overlap within embryo. The public
detector/head already saw both embryos. This is new-component transfer evidence,
not independent full-pipeline validation or a hidden-score prediction.

Preflight: data masks/known source edges/raw crop provenance; model zero/gradient,
reflection, valid-feature translation, jitter, tie/disagreement controls; actual
two-full-movie zero inference and original writer/official parity. Benchmark
must pass before plan registration or full training. Ten finite pilot jobs use
the existing Queue and pin input/source hashes. No Kaggle writes in either queue.

For each actual hidden launch record PID/process start and attach the existing
native Windows completion/failure notifier. Separate scheduled review handles
scientific continuation. Deadline2026-09-30 09:00KST; five submission slots are
user-reported and require fresh quota/dedup before a future verified submission.
