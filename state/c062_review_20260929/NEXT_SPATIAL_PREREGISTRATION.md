# Proposed fixed real-anchor pretrained-spatial component

2026-09-29. Exact proposed scientific recipe for parent review/registration.
This document performs no capture, fit, queue launch or deployment. Existing
C058/C060/C061/C062 remain closed and immutable. This new component is justified
by a different pretrained spatial representation and a real inference-anchor
task, not by a post-result change to any closed model's parameters.

## Question and scope

Can a small shared spatial scorer on the **unchanged production primary UNet's
dense feature field** locate original known GT points relative to actual final
C023 integer anchors, with both opposite-embryo transfer and already-good-point
protection? Test only this component first. No detector replacement, new node,
edge edit, graph assignment, synthetic-offset curriculum or Kaggle operation.

Use the original16,931 C058 known official pairs in the old22 movies as the
label universe. These preserve original C023 node ID, GT ID, t and native
integer center. Exclude runtime-synthetic nodes and centers without the fixed
feature support below by prediction/feature geometry, not residual magnitude,
GT continuity, confidence or an evaluation-derived mask. Report all excluded
original pairs unchanged. Use every supported known pair for its source
embryo's fit; do not select source points by the opposite fold's results.

The frozen detector/head already saw both embryos. Whole-embryo opposite fits
therefore evaluate only the new scorer's transfer conditional on those frozen
features, not a detector trained without access to the held-out embryo.

## Exact production representation

Use geometry review `state/c062_preflight/geometry/REVIEW.md` and its pinned
source/checkpoint audit. First instrument the original C023 `_v1284_refine`
invocation passively: execute its original candidate-mode head and return its
refined coordinates unchanged. Copy the production primary32-channel dense
feature field from this invocation; do not use old capture mode, which turns
off the head. No feature or BatchNorm parameter is updated.

The first-seen field for frame0 comes from `[0,1]`, index0. For each later t,
it comes from `[t-1,t]`, index1. Keep the exact eight XY transforms and addition
order, full-frame normalization with lower clamp0 and **no upper clamp3**,
FP32/math SDPA, TF32 off, cuDNN benchmark off and eval mode. Pin both real
context frames, even when only one has a known target.

Native coordinates map to feature coordinates `(z,y/4,x/4)` with origin0;
there is no pooled-pixel-center correction. Query at the original **final
integer C023 anchor**, not a rounded feature center or a GT-centered crop.
The feature field can be queried at that final anchor regardless of earlier
linefit movement; it is an image representation, not an ID-indexed node array.
Retain the original node/GT mapping for all residual calculations.

For each eligible point, sample a32x15x15x15 cube at integer feature offsets
`[-7,...,+7]^3` from the fractional anchor. Use the production trilinear
`index_features` function. Require the entire queried cube to lie inside the
real feature map so boundary clamping cannot fabricate context. Store FP32
cubes; a complete16,931-point upper bound is7.314GB before metadata, and actual
support will be smaller. This is explicit storage cost, not a request to copy
full feature volumes for every point.

A primary-only prefix extractor may save runtime **only after** its requested
first-seen field hashes/cubes exactly match passive full C023 inference on the
two usual full movies. Never approximate with an independently encoded crop,
duplicate frame, outgoing-edge window, single TTA view or secondary detector.

## One fixed shared scorer: 10,384 trainable parameters

Input is the32x15x15x15 sampled frozen field `F`. Calculate per-example/channel
local contrast `D = F - F[:,7,7,7]`, then divide by
`sqrt(mean(D**2 over all spatial positions) + 1e-6)`. No global source/target
statistics, embryo identifier, absolute-position channel or trainable class
intercept is provided. This removes constant per-channel offsets/scales as
one possible bias source; it does not guarantee domain invariance.

The head is exactly:

1. Bias-free valid3x3x3 Conv3d32→8, SiLU.
2. Bias-free valid3x3x3 Conv3d8→16, SiLU.
3. Bias-free1x1x1 Conv3d16→1, initialized to all zero weights.

No dropout, normalization layer, padding, dense class head or explicit center
prior. The11x11x11 output field is indexed by feature offsets`[-5,...,+5]^3`.
Every output location shares the same scorer; there are no independent learned
z/xy-class biases. The pretrained field already provides broad image context,
while the scorer observes a local5x5x5 feature neighborhood at each location.

Decode on the native integer candidate lattice z`[-4,...,+4]`, y/x
`[-17,...,+17]`, with spacing(1.625,.40625,.40625)um. Obtain each candidate logit
by the production trilinear lookup at feature offsets `(dz,dy/4,dx/4)` in the
11-cube. Retain only candidates whose physical Euclidean norm is<=7um:
**5,381 positions**. This includes every original integer known offset within
the existing7um official matching gate. The35x35 lateral range accommodates
native offsets6.90625um without clipping them to a coarser6.5um lattice.

Conditional CE is over the known point's one exact native candidate. Check
each original target's integer coordinates, membership and identity explicitly;
an unsupported/noninteger label stops preparation rather than being silently
rounded, clipped or removed. The loss is a supplied-point location task, not
a detection/background objective for other unannotated cells.

Inference is one unique maximum on that fixed native sphere. Tied maxima return
exact zero shift and a tie flag. No posterior mean, temperature, top-k averaging,
confidence threshold, direction-specific scale or post-result abstention rule.
The initial zero last layer gives a uniform supported map and exact zero
fallback. Report ownership conflicts and outside-image proposals, but do not
apply a newly selected guard to rescue a failed component.

## Source fitting, fixed before inspecting results

Two separate whole-embryo models, seed6301. Each uses only its source embryo's
supported real anchors. Uniform source movie, then uniform source known point;
batch32,1200steps, AdamW lr3e-4, weight decay1e-4, cosine schedule, FP32.
Final checkpoint only. No source tail oversampling, balanced displacement
classes, synthetic jitter, feature-axis reflection or intensity augmentation.
Feature-axis flips are especially inappropriate without proving how the
pretrained channels transform; source and runtime use the identical field.

This avoids C061's artificial-versus-real anchor distribution mismatch. It
does not eliminate ambiguous prediction-to-GT labels or the two-embryo sample
limitation. Report source-fit and opposite-embryo results separately, including
signed z/y/x bias. A source fit cannot select another checkpoint, recipe or
evaluation mask if the opposite fold fails.

## Required preflight, before the finite fits

1. Independent support inventory by movie/embryo, with eligible/ineligible
   counts for all original pairs, tails and good points. Do not promise the
   old C0619,294 count;15-cube support is a different geometry.
2. Exact passive full C023 inference on44b6_12dfb391 and6bba_05db0fb1, including
   coordinates, feature samples, low-detection arrays, edge values and admitted
   ILP outputs. Compare any shortened extractor to this field path exactly.
3. Affine-field query tests for all16 lateral quarter-grid phases, axis/sign/
   physical units, candidate support5381, known-label membership, sphere/tie
   handling and valid-convolution lattice origins.
4. Exact zero-head output at all eligible points, original16,931 residuals
   unchanged including exclusions, all-node frozen graph/writer/official
   parity on both control movies. No graph change is part of the trained probe.
5. Finite CE/gradient on disposable synthetic features, one bounded synthetic
   optimizer-step control, save/load exact parity and batch independence.
   Benchmark real extraction/IO and fixed batch computation before recording
   ETA. Pin source, support manifest, original identities and all frame inputs.

If either embryo has no eligible axial-tail examples, report an infeasible
registered component before fitting. Do not shrink the context after seeing
model performance. Any preflight implementation correction must be separately
recorded and preserve the scientific representation and fixed gates.

## Acceptance and stopping rule

On each **opposite** embryo require lower mean3D residual and lower mean
absolute z residual on all eligible pairs, original3D tails>3.5um and original
axial tails |dz|>3.5um. Originally good3D<=2.5um pairs must not worsen in either
mean (tolerance1e-9 for arithmetic only). Include every tie/no-op and report
all16,931 original pairs with excluded points unchanged. Require tail mean3D
improvement in at least two nonempty movies per embryo; report every movie,
not only winners. A subgroup too small to satisfy this gate is inconclusive.

Report native-z and xy error components, signed bias, moved/improved/harmed
counts, prediction-only ownership conflicts, and persistent-versus-broken-link
tails. Fixed identities remain the reference. No baseline rematching is
allowed to create a more favorable target after correction.

Failure closes this recipe without an epoch, normalization, context-size,
prior, guard, decoder or strength sweep. Passing only earns a distinct
GT-free all-node integration review with protection of originally good matched
identities and actual official score evidence. If the proposal is eventually
inserted before association, regenerate feature indexing, candidate edges,
Transformer scores and ILP. Old edge caches cannot test new coordinate geometry.
The two diagnostic source models are never deployed by embryo prefix.

Existing measured full-pipeline controls suggest roughly23minutes for22movies
before feature IO; primary-only extraction may be cheaper after exact parity.
The small scorer's fit should be secondary to capture cost, but record an
actual benchmark instead of converting this estimate into a launch promise.
No queue or ETA is created by this preregistration proposal.
