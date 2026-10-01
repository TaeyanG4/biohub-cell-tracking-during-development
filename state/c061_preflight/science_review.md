# C061 independent science and protocol review

2026-09-29. Review owner: temporal_next_review. This file reviews the declared
protocol; it is not an execution record or evidence of a learned improvement.
No model inference, training, queue, graph changes, or checkpoint selection was
performed for this review. DeepCenter inference remains blocked by provenance.

## Scientific verdict

The fixed direct-GEFF axial challenge is a justified distinct, bounded test.
It replaces C058/C060's prediction-to-GT residual training assignments with
direct annotated points and forces balanced exposure to every integer axial
offset from -4 through +4 voxels. The spatial-preserving broad-context head
can observe a different signal from C060's small per-position receptive field.
This supports testing identifiability; it is not evidence that direct labels,
more source records, or balanced shifts will solve actual detection errors.

The proposed nine-way conditional loss labels the displacement of one known
target in a supported crop. It does not assert that every other visible cell
is absent. An ambiguous crop containing several plausible nuclei remains an
identity limitation. Exact annotated x/y in the synthetic challenge provides
information unavailable at real C023 anchors. Synthetic success therefore
cannot replace the separately declared real-anchor evaluation.

The declared recipe is fixed: 13x49x49 crops from real 21x49x49 support,
nine balanced shifts, encoder widths 8/16/24 and flat64-to-9 head, no prior,
1200 steps, batch36, FP32, seed6101, learning rate0.0003, final checkpoint,
and two whole-embryo fits. Do not choose another epoch, decoder, offset scale,
guard, or architecture after seeing opposite-embryo results.

## Geometry and decoding controls required before a fit

1. Preserve GEFF identity, t/z/y/x ordering, native spacing
   `(1.625,0.40625,0.40625)` um, original floating coordinates, rounded crop
   centers, and the rounding operation as separate fields. Verify the integer
   fraction statistics of the selected annotations. If all are integral,
   assert that fact; otherwise preserve the fractional residual in physical
   error calculations rather than silently treating it as zero.
2. Let `a=round(gt_z)` and let the crop center be `a+s` for s in[-4,4]. Its
   target correction class is `-s`. If classes are ordered[-4,...,4], the
   class index is `4-s`. A predicted class k changes the anchor by `k*1.625`
   um. Check both signs and every shift using coordinate ramps/impulses and
   at least one actual raw crop, including the center voxel and full slices.
3. The stored z extent21 is exactly sufficient: the 13-plane read window
   starts at `4+s` within the expanded crop. Check full bounds before
   selection; never pad, wrap, resize, clamp the selected center, or discard
   one sign only. All nine variants of each selected point must exist.
4. The model must receive only normalized image values. No GT coordinates,
   class index, stem, embryo prefix, residual, or crop-origin metadata may be
   a feature. All nine classes must occur equally in each declared balanced
   batch. Report actual realized class counts, not only intended probabilities.
5. All-equal logits must yield zero displacement under a fixed exact-tie
   convention. Plain `argmax` would choose class -4. Resolve exact ties
   without GT, for example zero displacement on non-unique maxima; do not
   add a confidence tolerance or posthoc margin. This algebraic tie rule
   does not reinstate C060's orientation/ownership guard.
6. Check zero/equal logits, finite cross-entropy and gradients, all class
   directions, batch-independent outputs, and correct model save/reload.
   If the final layer is zero-initialized, first-step encoder gradients may
   be zero; a second step must establish a nonzero upstream gradient. Do
   not misinterpret the expected first-step effect as a dead backbone.
7. A constant identical-image batch with balanced classes cannot identify
   shifts: uniform predictions give loss log(9). Use that algebraic control
   and a coordinate/known-location positive control to distinguish image
   evidence from metadata or label-index shortcuts. These are correctness
   controls, not additional fitted scientific arms.

For noninteger annotations, the synthetic error baseline for shift s is
`abs(gt_z-(a+s))*1.625`; the after error uses that same original gt_z and
the decoded correction. Even the correct integer class retains the GEFF
fractional residual. Do not report an artificially exact zero target error.

## Source-only fit and correlation boundaries

Selection of up to32 supported points per movie must be deterministic from
source identifiers and the support predicate, before model results are read.
Do not select high-residual real pairs, bright peaks, convenient targets, or
points on which one model succeeds. Record selected and excluded counts by
movie and embryo, including empty or insufficient-support movies.

For a 44b6-trained model, only 44b6 images and direct-GEFF labels may enter
optimizer batches; all 6bba outputs are held out for that component. The
reverse direction is symmetric. Source-only sampling must be asserted against
actual batch IDs and persisted in the run record. Pooled extraction is fine,
but a merged cache must never become a pooled training loader. No target
labels may set normalization, sampler weights, stopping time, or checkpoint.
Existing fixed per-movie image quantiles are unsupervised input preprocessing;
they must be used consistently and cannot be adjusted by evaluation errors.

Own-source evaluation is descriptive memorization/fit evidence only. Even
opposite-embryo component evidence covers just two biological sources; the
199 movies and sampled points overlap and are not independent validation
domains. Record exact duplicate image-context/target counts where existing
fingerprints permit, and never interpret thousands of correlated rows as
thousands of independent cells or use row-level confidence intervals to
claim broad generalization. Public detector history also limits any eventual
end-to-end independence claim.

Pin both model/data source files, recipe, selected-point manifest, GEFF
metadata/coordinate chunks, image metadata and actual raw frame chunks read,
original C058 baseline/label files, and output artifacts. Source-change or
hash disagreement is a failed reproducibility check, not a scientific result.

## Required evaluations and report denominators

### Synthetic challenge

Evaluate every selected opposite-embryo point under all nine offsets with
the final fixed model. Keep all points and every class in the denominator.
Report before/after mean and median absolute z error, signed error, exact
class confusion/accuracy, tie count, and predicted class frequencies for
each shift and each embryo direction. Also aggregate negative, zero, and
positive shifts separately. The zero class is especially important: good
nonzero recovery with damaging spontaneous moves is insufficient.

Require improvement of nonzero-shift absolute z error in each transfer
direction. A sign-specific failure must remain visible; a good aggregate
must not conceal failure on one sign. Zero-shift behavior must be compared
with its correct original-coordinate baseline. Own-source results use the
same tables but cannot satisfy a transfer gate.

### Real anchors

Use all original C058 known matches on the fixed22 movies, retaining the
original `(stem,node_id,gt_id,t,target)` association. Select only on native
13x49x49 real support and absence of synthetic-node provenance; exclusion
must not depend on observed prediction, target error, or model confidence.
Report both the complete original-pair count and the eligible count. Also
report aggregate all-pair error with ineligible points unchanged, so changes
in coverage cannot manufacture a gain. Synthetic gap insertions stay excluded
even if their image crop happens to be supported.

Apply the raw fixed axial mode to the original integer predicted z, keeping
x/y and GT identity fixed. Do not rematch GT after the change to make errors
look smaller. Report absolute z and full3D error before/after, signed z bias,
tie count, class frequencies and movement rate for each embryo and each
movie. Strata must be defined from the original error: all eligible pairs,
the existing `>3.5um` full3D tail, and the existing `<=2.5um` good group.
Additionally report original large-axial-error counts, e.g. `abs(dz)>3.25um`,
as a diagnostic with its exact denominator. Do not choose that threshold
based on results or use strata as runtime gates.

The progression gate is real original-pair overall and tail improvement in
both absolute z and3D residual in both embryo directions, with good-group
means no worse in either measure. Empty groups are unavailable evidence,
not passes. Document any movie-level failures even if embryo means improve.
An official7um assignment is not biological identity ground truth; existing
continuity/ownership categories may describe failure modes but must not
filter evaluation or select corrections. Crops around actual C023 x/y are
the essential distribution-shift check against the oracle-x/y challenge.

## Interpretation and stopping boundary

If synthetic source fit fails, this fixed recipe does not identify the
direct target. If source succeeds but the opposite embryo fails, transfer
is unsupported. If the synthetic transfer succeeds but real-anchor evidence
fails, the controlled task is not solving C023's actual errors. None of
those outcomes justifies a new guard, bias correction, threshold, class
rescaling, changed shift distribution, or refit under the same experiment.

A pass supports designing a separate actual graph study; it does not imply
official or leaderboard improvement and is not a submission candidate.
Pre-association insertion would require regenerated feature indexing, edges,
Transformer scores and ILP rather than editing coordinates beside stale
downstream caches. Original C023/C024 graphs and submitted C046 remain
unchanged throughout this phase.

Implementation evidence reviewed at creation: C058 raw model/study geometry,
baseline original-pair labels, the remaining-methods localization/geometry
reviews, and the parent's fixed C061 protocol. C061 modules were not yet
present during this review; module-specific checks remain with their owners.
