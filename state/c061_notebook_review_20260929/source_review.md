# C061 public-source precedents: bounded local code review

2026-09-29. Only already downloaded notebooks, public asset source files and
prior local audits were read. No new crawl, network request, submission-score
poll, notebook execution, model inference or training was performed. Cell
indices below are zero-based notebook cells; source line numbers are one-based.
Notebook version freshness and radar score checks belong to the parent review.

## Answer

**There are close public precedents for learned coordinate correction and
direct annotated-centre supervision.** C061 should not be described as the
first attempt to learn cell centres or to use image context for correction.
The closest frozen-feature coordinate-refinement precedent is x138/V1284,
which is already part of our C023 baseline. The clearest locally available
full training-code precedent is xiaoleilian's Gaussian centre-heatmap U-Net.
Hengck's StrongUNet and the DeepCenter package provide related spatial-centre
representations and point-supervision source functions.

I did not identify C061's **specific combined protocol** in these reviewed
source versions: native13x49x49 crops generated directly from GEFF points,
exactly balanced axial displacements[-4..4], conditional nine-class correction
with no background labels, whole-embryo opposite fits, and a separate real
C023-anchor test preserving original GT identities. This is a bounded source
comparison, not a claim of originality across Kaggle or all previous versions.

| Public source | Actual related implementation | Difference from C061 |
|---|---|---|
| [anvithpothula/biohub-x138](https://www.kaggle.com/code/anvithpothula/biohub-x138) | Frozen UNet feature centre plus six directional differences ->224-dimensional feature ->32-hidden-unit MLP ->bounded3D displacement; coordinates and association feature indexing updated | Matched detector-to-GT residual head, small bounded3D correction; not direct-GEFF balanced raw axial classification. Already used by C023. |
| [xiaoleilian/biohub-unet3d-v2models-training-code](https://www.kaggle.com/code/xiaoleilian/biohub-unet3d-v2models-training-code) | Reads GEFF centres, builds Gaussian heatmap targets, trains spatial3D U-Net with weighted BCE and image augmentation | Whole pooled-frame detection, soft background penalties, movie splits inside both embryos; not conditional recentering or whole-embryo transfer. |
| [hengck23/cell-point-detector](https://www.kaggle.com/code/hengck23/cell-point-detector) | StrongUNet3D3Level returns three spatial feature maps and centre logits; attached model source supplies Gaussian point targets and balanced BCE | Frozen detector demo using downsampled volume and global peak extraction; no demonstrated C061 offset training or correction study. |
| [gautiermarti/biohub-deepcenter-unet3d](https://www.kaggle.com/code/gautiermarti/biohub-deepcenter-unet3d) | Full-frame centre heatmap used as repair-point acceptance evidence; linked DeepCenter source trains from GEFF Gaussian centres | A detector/gate, not nine-class local recentering. Exact historical checkpoint coordinate convention remains unresolved for a new coordinate use. |
| [binasalama/biohub-learned-unet-transformer-ilp-gap-recovery](https://www.kaggle.com/code/binasalama/biohub-learned-unet-transformer-ilp-gap-recovery) | Temporal UNet point detector and spatial feature lookup for node association; direct-GT detection objective described and present in support source | Learns whole detector/association system; unknown non-GT locations receive weak negatives. This is not C061's known-point conditional loss. |

## 1. Closest coordinate-head precedent: x138/V1284

Local notebook: `state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb`.
Cell4 line511 contains the literal source written to
`v1284_coordinate_refinement.py`. That string was extracted with AST literal
inspection only into `x138_cell4_v1284_coordinate_refinement.py` beside this
review; it was not imported or executed.

The extracted source establishes:

- Lines12-21: `Linear(224,32) -> SiLU -> Linear(32,3)`, zero final layer, and
  `2.0 * delta / (1.0 + norm(delta))` physical displacement bound.
- Lines24-33: seven frozen-feature lookups (centre and +/-one lattice step
  on each axis), concatenating the32 centre channels and six32-channel
  neighbour-minus-centre differences. This is224 numbers, not a dense feature
  crop processed by a second spatial CNN.
- Lines36-57: trilinear indexing at corrected positions for downstream node
  features. Lines60-85 load the public head, normalize features with its saved
  mean/scale, add its3D displacement, clamp image bounds, and validate the
  two-micrometre bound.
- Original cell4 lines524-530 install the correction before coordinate
  offsets are finalized, preserve fractional coordinates, and replace the
  downstream feature indexer. This is a concrete precedent for updating
  association features when coordinates change.

Original cell4 lines515-518 says the head was trained on x107's20-movie
capture containing4,136 detection-to-GT pairs, and reports movie-held-out
residual reductions. These are **author comments**, not a newly reproduced
training result or whole-embryo validation proof. The saved notebook contains
the inference/capture module and weights mount; the exact original training
optimizer/loss is not exposed by that module. The related public
[head-s075 notebook](https://www.kaggle.com/code/anvithpothula/biohub-v1284-head-s075-notebook)
in our cache is only a23-line starter/input-listing cell, not missing training
code discovered under another name.

The public dataset reference is
[anvithpothula/biohub-v1284-head-s075](https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075).
Our historical C023 integration is recorded in HANDOFF lines955-962; it is
already an exploited precedent rather than a new untested upgrade. The wider
frozen-UNet spatial-neighbourhood option would extend its representation, but
must not be presented as an entirely new mechanism without this qualification.

## 2. Closest direct-GEFF training code: xiaoleilian

Local notebook:
`state/notebook_radar/pulled/xiaoleilian__biohub-unet3d-v2models-training-code/biohub-unet3d-v2models-training-code.ipynb`.
Cell5 line2 contains a literal `FILES` dictionary with six original Python
scripts. Those strings were extracted without executing them into the
`xiaoleilian_cell5_*.py` files next to this review.

In extracted `train_unet_v2.py`, `load_movie_frames` lines54-69 directly reads
GEFF nodes and maps centres to `(z,y/4,x/4)` on the image's XY-pooled lattice.
`stamp` lines40-51 builds Gaussian targets around rounded centre coordinates.
`build_sample` lines99-120 applies consistent XY flips/rotations to images
and target maps, image brightness/background/noise augmentation, and the
weighted supervision map. Line157 computes weighted BCE. The v3 script uses
the same target construction with caching/prefetch/AMP-related execution
changes, rather than a new balanced axial correction task.

Its weight recipe is positive12, low-intensity background1, other unlabeled
locations0.05. Therefore even its weakly ignored regions retain a nonzero
background contribution. It does not meet our stricter requirement that
unknown cells never become background negatives for the new task.

The source header calls its splits embryo-grouped, but the **actual**
`splits.py:42-54` allocates12 movies from each embryo to validation and the
remaining movies from those same embryos to training. This is movie-level
separation stratified by embryo, not C061's opposite-whole-embryo evaluation.
Its validation chooses the best checkpoint by point recall. Neither code
comments saying leak-free nor a detector-training result establishes hidden
embryo generalization in this competition.

The public notebook is an especially useful precedent for direct annotated
point-to-image supervision and broad3D context. It is not a drop-in C061
recipe. Author timing notes in cell0 are historical for its full detector
schedule and must not replace measured C061 runtime.

## 3. StrongUNet: explicit spatial feature maps and point-target helper

Local notebook:
`state/notebook_radar/pulled/hengck23__cell-point-detector/cell-point-detector.ipynb`.
Cell1 lines139-159 imports `model_v5.StrongUNet3D3Level` from the public
[hengck23-cell-point-detector-demo asset](https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo).
Cell2 loads the frozen checkpoint, reads image values at `[:,::1,::4,::4]`
(line33), and obtains `[f0,f1,f2], node_logits` (line121). Thus “full resolution”
in the model docstring means its already downsampled input lattice, not the
original native256x256 lateral image.

The downloaded `artifacts/hengck_point_detector/model_v5.py` returns its
feature pyramid and logits at lines143-145. Its `make_gaussian_center_target`
at lines152-229 rounds supplied centres and stamps3D Gaussian targets.
`soft_gaussian_balanced_bce_loss` at lines232-262 averages foreground and
complement losses separately. This source supplies a related learning
objective; this review does not assert that its mere inclusion cryptographically
proves the historical checkpoint's exact training run.

The notebook demonstrates global peak extraction from that frozen field, not
a local axial displacement learner or a trained dense-feature correction
head. Existing C056 StrongUNet readmission was a different integration test;
its outcome cannot establish success or failure of every possible feature
localization method, and is not a reason to repeat readmission.

## 4. DeepCenter and base learned UNet

Gautiermarti's notebook cell18 defines `deepcenter_heatmap_for_frame` at
lines329-357, `deepcenter_score_point` at358-385, and repair acceptance at
386-412. The score function queries a local heatmap maximum and discards its
location. It is evidence that a learned centre map is used, not that those
peaks have been validated as displacement proposals for existing nodes.

The attached DeepCenter source reads GEFF coordinates and constructs Gaussian
targets (`train_full_frame_center_detector.py:263-305`), then trains a full
pooled-frame model (`340-365`, weighted loss425-428). Its nonzero ignore weight
also differs from C061's conditional known-target supervision. The previously
documented package coordinate conflict and missing historical checkpoint-to-
source binding remain unresolved. This review does not reopen the blocked
frozen-coordinate probe or choose a conversion from GT scores.

The learned-UNet notebook's markdown cell3 explicitly describes detection
positives at annotated nodes and weak negatives elsewhere. Its public support
source `artifacts/pilkwang_support50/repo/scripts/train_unet_transformer.py`
actually implements that objective at528-588. The same file's feature-indexing
and encoding methods at447-510 expose a reusable spatial representation for
nodes. They establish technical feasibility of feature access, not validated
performance of a new frozen spatial correction head.

## What this means for the current experiment

C061 combines familiar learned-centre ideas with a narrowly specified new
diagnostic: direct known-point labels, balanced native-z displacement, a
conditional target that avoids assigning unknown background labels, and
strict real-anchor/opposite-embryo evaluation. The strongest distinction is
the supervision and evaluation protocol, not that a CNN sees raw cells.

Public precedents support the hypothesis being reasonable to test. They do
not demonstrate that C061 will work, justify changing its running recipe, or
replace actual graph/official-score validation. This review recommends no
new model or extra queue. `evidence.json` records source hashes, notebook
cell/line snippets, asset links and the bounded scope of the comparison.
