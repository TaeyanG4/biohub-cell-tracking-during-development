# C061: direct known-point axial localization

Prepared2026-09-29 after user requested continued work. This is a bounded
component falsification experiment, not a submitted candidate or graph variant.
C058/C060 sources and results stay immutable. C023/C024 remain scored anchors.

## Mechanism

Supervision starts directly at known GEFF coordinates in the source embryo.
Crop centers are displaced by each integer z offset from -4 through +4 native
planes (1.625 um per plane). Every batch represents all nine displacements
equally. All shifted views contain real pixels from a 21x49x49 envelope and
have shape13x49x49. Positive supplied-point localization never labels other
unannotated nuclei as absent. No prediction-to-GT assignment defines source
targets. Known coordinates must be integral for this first fixed recipe;
the actual data audit must verify that requirement, not silently round labels.

Use one spatial CNN recipe with whole-crop access through its flattened head,
based on the existing C058 encoder pattern, and nine conditional z classes.
There is no center-favoring prior and no tail-dependent sampler. Exact tied
maxima return zero displacement. X/y reflection augmentation preserves z labels;
no z reflection or evaluation-chosen coordinate convention is used.
The model corrects only z; x/y stay fixed. Model source contains the complete
fixed recipe: two source-only1200-step fits, AdamW, batch36, FP32, final checkpoint.

## Data and separation

Read all199 available movie annotations; select at most32 fully supported known
points per movie using a stable hash of source identity and location. Record
overlap/correlation limitations; these are two biological embryos. Each model
trains on exactly one embryo and is evaluated on the other. Own-source results
describe fitting only. Source-only data determine optimization, with no target
checkpoint selection, threshold sweep, deployment embryo router or unknown negatives.

The synthetic challenge uses all nine displaced views of the registered known
points, including zero. The real-anchor challenge uses all eligible ORIGINAL
C023 known pairs in the old22 movies, retaining original node and GT identities;
no post-correction rematching or successful-case selection. Real anchors use
their actual C023 x/y/z centers and a13x49x49 crop. Synthetic gap nodes and crops
without real support are excluded by the original image/topology conditions.
All exclusions and per-embryo coverage are reported. GT selects diagnostic labels
only; this component phase does not define a runtime node whitelist.

## Controls and decision

Before registering fits: exact crop origin/sign/units, all nine shifted real
views, original GEFF label equality, integral coordinates, unique selection,
fold separation, equal class sampling, ties/zero output, finite loss/gradients,
actual data extraction, and disposable timed training steps must pass. Pin the
selected input image chunks, metadata, labels, graphs, source and control artifacts.

After fitting report BOTH own-source and opposite-embryo results. For each
nonzero synthetic displacement and each opposite embryo, mean absolute z error
must improve over no correction. At zero displacement mean error must be no
more than one native z plane. This diagnostic gate alone proves no graph gain.

The real-anchor gate requires in BOTH opposite directions: overall mean 3D
residual and mean absolute z error improve, original >3.5 um tail residual and
the subset with |dz|>3.5 um improve, and original <=2.5 um good-point residual
mean and mean absolute z error do not worsen. Require absolute z error as well
as3D residual improvement for the tail strata. Report persistent versus broken
known links where present. Report all16,931 original pairs, separating9,294
eligible inferred pairs from7,637 ineligible pairs held unchanged.
Every metric includes exact ties/no-change outputs. Report endpoint proposals
outside image bounds or outside the original prediction's spatial ownership;
do not convert these into a post-hoc guard or tuned threshold.

Failure closes this fixed recipe; no epoch/decoder/strength sweep. A pass only
earns a separately reviewed GT-free all-node integration with exact controls,
identity protection and actual official scoring. Fresh pre-association feature
indexing, candidate edges, Transformer scores and ILP are required before any
association-gain claim. No official score is claimed for this component probe.

## Execution

Reuse existing frame reader, bounded crop, normalization, Queue and hash helpers.
No evaluator, notification tool, crawler or visualizer is rebuilt. No Kaggle
write occurs here. A launched queue records actualPID/time/measuredETA and gets
the existing native Windows completion/failure watcher, with observer logs
outside the scientific output-hash tree. Scientific review uses the same
biohub-t4 heartbeat at ETA-10minutes then20minutes while work remains.
