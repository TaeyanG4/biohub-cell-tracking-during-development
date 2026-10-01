# Remaining localization methods: information and geometry review

2026-09-29. Read-only assessment; no fit, inference sweep, candidate, or queue.
The completed C058/C060 recipes remain closed. This assessment does not support
changing C060's guard, prior width, epochs, or posterior decoder.

## What the evidence leaves open

Raw3D information has not been shown exhausted, but the **identity of the target
point** is the limiting scientific question. A known sparse annotation is an
operational target; an official7um prediction-to-GT assignment is not proof
that those two locations belong to the same biological nucleus. Persistent
offsets on correct paths and annotation matches switching between smooth paths
must remain separate evaluation strata.

The C060 learned logit map is nonzero and its conditional loss improves. It
nevertheless changes z on only5 of4,085 moved runtime nodes and none of109 moved
known matches. Even its own-source unguarded modes barely recover tail offsets.
That rules out treating the low coverage solely as an overly strict acceptance
test. C058 previously fit source offsets strongly and transferred them poorly.
Neither result establishes that known annotation centres are inaccessible to
a richer pretrained image representation; neither justifies another small
localizer architecture sweep.

One specific unused information source merits a cheap test: **the existing
full-frame learned centre heatmap as a coordinate signal**, rather than raw
brightness or C060's local conditional scalar. C023 uses DeepCenter mainly as
a repair acceptance score (`deepcenter_score_point` takes a local maximum);
that API deliberately discards where the peak lies. C055/C056 tested detection
readmission, not displacement of the already existing C023 nodes toward this
pretrained centre signal. Confirm there is no duplicate study before launch.

## Why larger image context is a distinct question, but not yet a candidate

Each C060 output uses two valid3x7x7 convolutions: its image receptive field is
only5x13x13 input voxels (centre-to-centre extent6.5x4.875x4.875um). The overall
13x49x49 input contains a much larger region, but there is no spatial mixing
beyond this local receptive field at a given candidate output position.
Separating nucleus shape, overlapping neighbours, and axial extent can require
larger context than local intensity texture. Per-voxel channel normalization
also removes some absolute channel level information. These are architectural
facts, not evidence that undoing either choice will improve the score.

C058's flattened head already had broad crop access, so “make the crop bigger”
alone does not isolate a new mechanism. A pretrained full-frame centre network
offers a different, already learned representation and direct point-label
objective, available without fitting another small model. The existing
`artifacts/pilkwang_deepcenter` package has a three-level3D U-Net with full-frame
input, positive-unlabelled training, and XY pooling4. On its input lattice all
axes therefore have1.625um spacing. Its exact training-domain provenance must
be checked before assigning either embryo a transfer interpretation; the earlier
claim here that this detector had seen both embryos was not established.

## Cheapest useful experiment, before new training

**A frozen DeepCenter coordinate-information audit on the already fixed256
C060 diagnostic points**, retaining original `(stem,node_id,gt_id,target)`
identities and the same128-per-embryo cap. Do not choose new points based on
heatmap agreement or measured success. This sample deliberately overrepresents
tail points and cannot estimate population gain.

Predeclare one interpretation before looking at results: the source training
heatmap's coordinate convention, and the existing fixed DeepCenter checkpoint,
normalization, and actual C023 TTA setting, without introducing another TTA arm. Reuse C023 namespace and
`deepcenter_heatmap_for_frame`; read each sampled frame once. Compare the
annotation target's learned heatmap evidence with the C023 centre and its local
spatial alternatives inside the existing physical7um radius. Record a unique
argmax proposal only as a diagnostic; never choose the peak nearest GT.

The identifiable target is the **original annotated coordinate**, not a bright
peak, a synthetic corruption target, a fresh rematched GT, or the nearest
neighbour chosen retrospectively. Record both target score/rank and the fixed
argmax's physical error against that same target. In parallel, record whether
the proposal remains in the original all-prediction ownership cell and whether
its mode persists along the supplied predicted track. GT may classify the
diagnostic strata; it must not supply search centres, proposals, or runtime
acceptance. Unknown other predictions remain competing identities, not false
cells/background labels.

Cheap go/no-go for investing in a concrete integration study: the frozen centre
signal's fixed proposal must reduce tail and overall paired mean error in both
embryos while leaving the <=2.5um good stratum's mean no worse. Report all points,
all abstentions and z/y/x separately. If this condition fails, do not tune a
radius, peak threshold, mixture or conversion offset. If it passes, it only
supports a separately registered full22 coordinate/identity/official-graph
study; it does not authorize submission from this stratified, fit-domain probe.

This is one frozen information test, not a menu of detector ensembles. If
direct source-label detector adaptation is selected instead, run this probe as
its frozen-control evidence only; do not duplicate work.

### Critical coordinate-contract issue to resolve first

The package README and gate inference use:

`y_original = y_pooled*4 +1.5`, likewise x.

However, `source_scripts/train_full_frame_center_detector.py:make_heatmap`
places the **label target** at `(z, y/4, x/4)`, without subtracting1.5 before
division. These are two different conventions. For the learnt label-location
target, the literal inverse is multiplication by4; the physical pooled-pixel
centre is multiplication by4 plus1.5. The difference is0.609375um in y/x,
not a3–7um z-error explanation. C023's current broad score window must stay
unchanged; this finding is not evidence of a scored pipeline bug.

The bounded algebraic check `deepcenter_contract.py/json` now verifies the exact
manifest-pinned `make_heatmap`: annotation(16,32,48) yields mode(16,8,12), whose
literal label inverse is(16,32,48), while the documented physical pixel centre
is(16,33.5,49.5). No real movie or model inference was used for this check.

Before a coordinate audit, inspect the exact checkpoint/source provenance and
fix the conversion from that source contract. Do not select between these conventions by whichever gives better
GT scores. If checkpoint provenance cannot establish which target convention
was trained, this coordinate probe is blocked and direct supervised adaptation
with an explicitly tested convention is cleaner scientifically.

## Axial profiles and physical geometry

Native z spacing is four times XY spacing. A one-plane z change is1.625um;
upsampling z cannot create new measured axial information. C060's own output
lattice covers z offsets through±6.5um, so its near-zero z changes are not a
missing-support implementation bug. The model did not learn/select the required
axial correction confidently under the fixed recipe.

Raw peak snapping and a symmetric PSF/Gaussian-centre fit are currently
unsupported: the fixed image audit's tail annotations lie4.99/5.85um from the
nearest smoothed peak, versus2.33/3.17um for C023 predictions, with no observed
>15% intensity valley along the sampled segments. A generic axial Gaussian peak
therefore targets a different object than the annotation unless image/label
alignment is established. No measured PSF or calibration stack is available in
the reviewed infrastructure. Fitting a PSF model or choosing an axial centroid
formula on two evaluation embryos would add assumptions and another selection
problem, not known localization information.

Whole-object axial profiles may still describe centre-label bias versus local
brightness. If needed, extract profiles at the **existing fixed predicted and
annotated points** and report them descriptively; do not feed GT-centred profiles
to a deployment model. A profile audit should answer whether the labelled centre
has a reproducible image signature, not just whether a curve has a peak. The
frozen learnt centre signal above addresses that question more directly.

## Synthetic displacement and self-supervision: useful control, not an absolute target

Shared synthetic crop translation supplies a perfectly identifiable displacement.
The existing `local_registration_probe.py:register_once/local_motion` and C060
real-support jitter/reflection controls already verify that sign, units, and
common image support can be recovered or transformed correctly. These routines
can provide a bounded positive control for a new detector's geometry, without
new evaluators or registration implementations.

But successful recovery of a deliberately shifted patch returns to its original
anchor. If that anchor is a biased C023 point, the pseudo-target inherits the
bias; if it is an unknown biological point, self-supervision never establishes
the annotation's absolute centre. Temporal registration likewise transports
the initial offset. It cannot resolve the observed persistent annotation offset
by itself. C036/C051 are already closed on actual graph evidence; repeating
them or training only synthetic shift recovery is not a new score candidate.

Synthetic displacement could later regularize a source-supervised direct centre
learner, but source labels must define the absolute target, and the improvement
must survive opposite-embryo validation. This is subordinate to demonstrating
real label-location information, not a reason to spend hours on self-supervised
pretraining before the deadline.

## Reuse and boundary of this recommendation

- `src/eval_pp_variants_local.py:build_namespace/install_frame_caches` and
  C023's `deepcenter_heatmap_for_frame` provide the actual frozen model path.
- `artifacts/pilkwang_deepcenter/ARTIFACT_MANIFEST.json`, README, config and
  `source_scripts/train_full_frame_center_detector.py` provide model/target
  provenance and `make_heatmap` geometry controls.
- `state/c060_review_20260929/model/sample.csv` pins the256 points;
  `state/localization_diagnosis_20260929/continuity.csv` supplies diagnostic
  persistent versus broken-link categories.
- `src/c058_localizer_baseline.py` supplies exact original matching, writer and
  actual official scoring when a real graph hypothesis is later justified.
- `src/local_registration_probe.py` already supplies bounded real crops and
  known-translation controls; no new NCC or postprocess scorer is needed.

The strongest remaining long-term route is direct image-to-annotated-centre
training with trustworthy sparse-label supervision and broader image context.
Other reviewers are assessing that route and its runtime. The cheapest local
geometry contribution is the frozen centre-information audit above, with its
coordinate-contract preflight. At present there is no demonstrated gain, no
new candidate ready to submit, and no reason to spend a submission on C060.
