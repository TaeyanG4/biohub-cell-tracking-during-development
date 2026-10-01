# C060 saved-posterior diagnosis

The model is not a dead zero head, and the all-node ownership guard is not the
main reason so few points move in this bounded sample. **The eight unaveraged
orientation maps disagree on the integer mode for89–95% of opposite-embryo
samples.** The averaged map proposes mostly small in-plane shifts, with weak
recovery of large source-label offsets even on its own training embryo.

This is a diagnosis of the immutable completed C060, not an alternate candidate.
No fit, checkpoint selection, full-movie inference, graph edit, prior adjustment,
threshold search, or guard relaxation occurred. All comparisons keep each saved
original known predicted-node/GT identity and target exactly fixed.

## Sample and control validity

At most128 stored training-crop rows were chosen per embryo before inference.
Selection was SHA256-first up to64 tail points (>3.5um), then non-tail rows to128:
44b6 contains58 tails/70 others;6bba64/64. This residual-stratified sample is not
a population mean, and its temporal/crop-correlated points are not independent.
The source pools are1,021/6,255 stable-known labels. Each128-row sample was run
through both immutable final1200-step models without augmentation, and through
the saved zero-head prior analytically. There are256 unique inputs,512 model
diagnostic rows.

Crop/target receipts and actual bytes were verified. Original C058 label identity,
target, graph row/node ID, and saved C060 target identity agree exactly for all
sampled points. The actual `group_average_logits` and `decode_modes` functions
were reused; a direct `model.predict` equality check passed. Repeating the
production ownership calculation among all same-frame original nodes yielded
exactly the saved opposite-model full-precision guarded shifts **and mode
agreement** on all256 samples, despite using the stored float16 crops here.
Thus crop storage or this diagnostic's decoding does not explain the result.

## Where proposed movement disappears

| Evaluated sample | Model trained on | Nonzero averaged mode | Orientation disagreement | Agreed nonzero | Rejected next by ownership | Final moved |
|---|---|---:|---:|---:|---:|---:|
|44b6,128 points|6bba (opposite)|111|122|4|0|4|
|44b6,128 points|44b6 (own source)|104|122|4|0|4|
|6bba,128 points|44b6 (opposite)|98|114|1|0|1|
|6bba,128 points|6bba (own source)|93|113|0|0|0|

There were no tied individual modes in these samples. Disagreement is actual
orientation-dependent peak selection, not a numerical tie fallback. The
averaged map itself remains exactly centered on17/128 and30/128 opposite-model
samples, so pure centering explains only part of the no-op behavior. Among
nonzero averaged-mode proposals,107/111 and97/98 disagree across orientations.

The unguarded mode displacement magnitude averages only0.756/0.671um; only6/128
points in each opposite-model sample have a nonzero z proposal. This is small
against the sampled mean original errors2.736/2.972um and tail errors4.521/4.341um.
The guard does suppress proposals, but these facts do not establish that the
suppressed proposals are useful or identity-safe.

## Fixed-pair residuals, micrometres

Unguarded mode/mean below are **descriptive outputs only**, not new inference
arms, graph scores, or candidates. A mean can lie between distinct modes.

| Sample | Model | Original | Averaged-map mode, no guards | Posterior mean, no guards | Registered guarded output |
|---|---|---:|---:|---:|---:|
|44b6 all128|own source|2.73614|2.70487|2.61750|2.73374|
|44b6 all128|opposite|2.73614|2.71768|2.67053|2.73208|
|44b6 tail58|own source|4.52057|4.49882|4.34038|4.51905|
|44b6 tail58|opposite|4.52057|4.47097|4.38760|4.50616|
|6bba all128|own source|2.97230|2.92853|2.82734|2.97230|
|6bba all128|opposite|2.97230|2.98147|2.85316|2.97491|
|6bba tail64|own source|4.34150|4.26806|4.18786|4.34150|
|6bba tail64|opposite|4.34150|4.35337|4.20406|4.34150|

For originally good (<=2.5um) opposite-model points, the unguarded mode worsens
44b6's66 points1.15129→1.17195um and6bba's57 points1.42776→1.46926um. The guarded
outputs also worsen these particular samples slightly (1.15607/1.43362um).
This does not support removing agreement as a production fix. The posterior
mean shows modest descriptive residual improvement, but it has not been
validated for identity/graph safety and is not an authorized replacement recipe.

## The logits learned something, but not the required large directional offsets

Against the zero-head radial prior, opposite-model learned logit evidence has
within-map standard deviation0.824/0.803 and range3.342/3.110 on average;
posterior KL from that prior is0.540/0.584nats. Conditional Gaussian-target cross
entropy improves7.755→7.000 and7.817→7.368. The head is neither unchanged nor
numerically disconnected.

On the tail samples, cross entropy improves8.593→8.150 and8.484→8.359, but the
average **GT-point versus original-center log-odds improvement** is only−0.0068
and−0.4101. Higher absolute target likelihood can accompany redistribution
elsewhere; it does not imply the true target is preferred to the original
center or that a useful mode has been found. Own-source tail log-odds gains are
also weak (+0.0055/−0.3477), consistent with the small own-source residual gains.

| Opposite-model tail | Mean required z/y/x correction | Mean unguarded mode correction | Mean posterior correction |
|---|---|---|---|
|44b6,58|(-1.457,+0.028,-0.343)|(+0.028,-0.126,+0.147)|(+0.019,-0.016,+0.052)|
|6bba,64|(+2.057,+0.355,+1.244)|(-0.051,-0.083,+0.095)|(-0.024,+0.041,+0.070)|

The z component remains near zero with the wrong mean sign on both sampled
tails. Unlike C058's strong own-source fit and poor transfer, this C060 learner
shows limited large-offset fitting even on its source sample. The evidence is
compatible with restricted spatial evidence, annotation/image-center mismatch,
the fixed prior, or model/training limitations; this audit does not isolate a
single cause and does not justify a parameter sweep.

Artifacts: `sample.csv`, `selection.json`, `rows.csv`, `input_hashes.json`, and
`diagnosis.json`. All512 raw diagnostic rows were saved before a summary-only
Pandas attribute collision (`tail` column versus `.tail` method). The preserved
failure log documents it; aggregation was recovered with `--summarize-only`
without repeating inference. Original candidate/model/data sources are unchanged.
