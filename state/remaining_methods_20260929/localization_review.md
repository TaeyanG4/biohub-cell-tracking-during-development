# Remaining localization mechanisms after C060

2026-09-29. Independent source review; no training, inference, candidate edits,
or queue launch in this review. The deadline is 2026-09-30 09:00 KST. There is
room for a bounded new mechanism test, but no measured basis for promising a
score gain or giving a training ETA from the old training smoke logs.

**Priority: test whether annotations can be localized axially across embryos
when training examples are generated directly from known annotated points.**
That tests a different data-generating task from C058/C060's residual labels
on already matched final predictions. Keep a frozen-feature auxiliary spatial
head as a second option. Do not run the existing whole-detector fine-tuner
unchanged, and do not reopen C060 by changing its prior, guards or epochs.

## What the two completed failures do and do not establish

C058 learned its source residuals and applied wrong-domain displacements to
the opposite embryo. C060 instead had weak source large-offset recovery;
its unguarded maps already harmed good points on both embryos and scarcely
proposed any z movement. Its orientation check removed most proposals, but
loosening that check does not establish useful evidence. Only five of 4,085
accepted C060 movements changed z, and none of the 109 known moved points
changed z. This is a reason to test axial image information directly, not a
reason to assume axial correction will succeed.

The native voxel spacing is (z,y,x)=(1.625,0.40625,0.40625) micrometres. The
existing detector subsamples y/x fourfold and leaves z unchanged, giving an
isotropic 1.625 micrometre detector grid. Merely turning off z downsampling
would therefore change nothing. C060's two valid 3x7x7 convolutions give an
individual output location a 5x13x13 raw receptive field: 6.5 micrometres
between extreme z samples and 4.875 micrometres between extreme x/y samples.
Its full crop was larger, but that is not its per-location contextual field.
A wider anisotropic image context is a plausible missing signal, not yet a
demonstrated cause. Raw bright-peak snapping already has negative evidence.

The 1,047/1,385 association concerns missed links with both detections present;
it is neither a detector-error percentage nor a recoverable-score ceiling.
Most large-residual nodes have their known adjacent links correct. Fixed final
topology can improve official edges only through changed matching/coverage.
A successful coordinate component still needs a separate pre-association test.

## Option 1: direct-known-point axial recentering challenge, then a native-image model

**Different mechanism/data:** generate supervised crops directly from GEFF
known points in the source embryo, rather than assigning a final predicted
node to GT and regressing that assignment's residual. Recenter the *real image
read window* by balanced physical z offsets and train the known annotated
point's location in that window. Use broad native xz/yz spatial context or an
anisotropic volumetric encoder; do not collapse the input to its brightest
voxel or an intensity centroid. Initially predict only z and keep x/y fixed,
so the experiment tests the weak axis without also changing lateral identity.

This is conditional point localization, not a whole-volume detection loss.
It must not label other unannotated cells as absent. All samples have a known
target; no artificial zero-background heatmap over the rest of the image.
Multiple visible cells can still make the conditional target ambiguous, which
must remain a limitation instead of being hidden by invented negative labels.
Source GT can generate training crops; evaluation GT cannot choose runtime
nodes, corrections, confidence, or candidate thresholds.

**Smallest falsification test, before a full graph candidate:**

1. Use a fixed hash-selected set of real known-point crops from each embryo,
   grouped by biological source/context, with direct native z displacements
   (for example 0, +/-1 and +/-4 voxels, all represented equally). Require real
   support at every shifted crop and exact inverse coordinate transforms.
2. Start with exact annotated x/y as a deliberately favorable diagnostic.
   Show own-source and opposite-embryo recovery of each nonzero displacement,
   zero-offset behavior, signed z bias, and median/mean absolute z error.
   This oracle-x/y challenge is not a deployable result or official score.
   Failure even here rejects this mechanism without full-movie inference.
3. If that transfers, repeat the fixed evaluation using actual C023 prediction
   x/y and identity-frozen known pairs, including already-good points and
   genuine large-z residuals. Controlled synthetic-offset success alone is
   insufficient: realistic prediction anchoring can switch the visible cell.
   Both embryos must improve real axial error without losing good identities
   before an all22 graph intervention is considered.

Do not tune offsets or architecture against the second embryo. A single
predeclared source-fit recipe and both opposite-embryo directions are needed.
Eight-way mode agreement is not automatically the right acceptance method
for this new diagnostic; no production acceptance rule should be selected
after seeing its evaluation results.

**Available data and feasibility:** root's read-only `gt_support.json` finds
14,491/57,893 records with support for its specified z-shift challenge on
44b6/6bba, versus C060's stable matched-triple source pools of 1,021/6,255.
Those are overlapping frame/crop records, not independent cells or events.
The inventory demonstrates crop availability, not model learnability. Existing
image readers, voxel transforms and known-GT access can be reused; begin with
a small fixed subset and time actual extraction/fit before scheduling a full
study. An image-native model does not need whole-movie detector training for
this falsification test.

**Weakest link:** annotations may not coincide with any transferable raw-image
center, and exact-x/y synthetic recentering is easier than localizing the cell
at a noisy detector anchor. Direct GT examples remove prediction-matching
label noise but do not remove the two-embryo domain gap or deployment ambiguity.
If the real-anchor test fails, do not rescue it using an evaluation-derived
z-bias correction or threshold sweep.

## Option 2: auxiliary spatial localizer on immutable pretrained detector features

**Different mechanism:** freeze the production UNet, detection head,
Transformer and normalization state. Capture an actual spatial neighborhood
of its pretrained feature map for a supplied candidate, and learn an auxiliary
conditional spatial localization head. The V1284 capture infrastructure
already reaches this point of the pipeline, but its 224-dimensional
center-plus-six-neighbor-differences vector is not a spatial feature cube.
A dense local feature field retains competing locations and the pretrained
encoder's larger visual context; it avoids training a small raw-image encoder
from scratch and avoids updating the feature space used by the edge model.

Use exact C023 inference as the capture source, not the historical C011
default in `v1284_capture_local.py`. Known-point conditional supervision and
source-only normalization are required. A plain retraining/scaling of the
existing 224-to-32-to-3 V1284 MLP is the old family, not this option.

**Smallest falsification test:** first prove zero-head/full-movie exact parity
and that frozen feature hashes/edge logits stay identical. Then train one
fixed small head in each source embryo on a bounded set of known crops and
evaluate fixed-pair axial/tail/good residuals on the other. Require source
large-z learnability and improvement in both opposite directions; otherwise
stop before changing graphs. This can reuse Option 1's point challenges to
compare information content, but should not become a head/scale sweep.

**Weakest link:** the frozen features may encode precisely the same localization
bias as the original detector. Their grid is still 1.625 micrometres in every
axis, so this cannot claim new native lateral detail. Existing V1284 evidence
does not prove a new dense head will transfer. Fresh spatial-feature capture
and safe integration add more work than the direct raw-crop challenge.

## Why unchanged detector fine-tuning is not a justified shortcut

`src/train_local_unet.py` imports `compute_detection_loss` from
`artifacts/pilkwang_support50/repo/scripts/train_unet_transformer.py`. That
function creates a zero target everywhere and marks only known GT voxels
positive. Its BCE penalizes every non-GT voxel, even with small `neg_weight`.
Unannotated real cells are therefore negatives. Setting that weight to zero
alone creates a degenerate all-positive detector objective; it does not solve
sparse-label detection. A valid alternative needs a separate justified
conditional localization task or explicit teacher-consistency objective, with
its own failure controls.

The wrapper also optimizes all model parameters, uses movie-ID validation,
accepts checkpoint key mismatches with `strict=False`, and selects checkpoints
by edge accuracy times recall. Those defaults do not demonstrate unchanged
production features, whole-embryo transfer, or official graph improvement.
The saved training history has only one epoch; its 12.7-second train duration
does not establish full training time because the script supports bounded
`--max-iters` smoke runs and the saved config omits that execution detail.

C008's 0.926 public outcome and the external V1327/V1329 0.942 outcomes are
negative evidence against indiscriminate detector replacement. They do not
prove that every protected localization learner fails, and their historical
failure attribution is not a controlled causal isolation. Preserving the
production feature path with an auxiliary head addresses a concrete hazard;
it is not evidence of a score gain by itself.

## Integration and stopping boundary

Prefer Option 1's smallest falsification test first; consider Option 2 only
if the evidence says pretrained context is worth a distinct bounded trial.
Neither is ready for submission. Only a transferable component should earn
the cost of immutable zero controls, all22 official graphs, unchanged97
review, portable runtime, actual T4 and fresh quota/dedup submission checks.
If inserted before association, regenerate feature indexing, candidate
edges, Transformer scores and ILP from changed coordinates; editing cached
coordinates with stale downstream features is invalid. Reuse the existing
official scorer and replay/inference infrastructure throughout.

Evidence read: C060 science/model reviews; C058 model diagnosis; localization
continuity/identity/image reports; `train_local_unet.py`; imported training
loss/model/dataset; V1284 capture/head trainer; HANDOFF C008; public-ideas
learned/alternative audits. No new empirical performance result is claimed.
