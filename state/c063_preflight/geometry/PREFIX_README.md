# C063 exact primary-prefix extraction

New adapter: `src/c063_frozen_spatial_extract.py`. Capture/model/registered
preflight sources are unchanged. No GPU extraction, proof or queue was run
while implementing it. CPU AST/metadata controls passed at
`state/c063_preflight/geometry/prefix_controls/controls.json`.

The adapter extracts the original predictor's AST, retains its24 setup
statements and first6 per-window statements exactly (the sixth is the complete
primary8-view TTA block), and stops before `secondary_unet_out = None`.
Only the window list is narrowed to requested first-seen windows. It adds a
read-only cube callback and releases temporaries. Production frame reader,
metadata/quantiles, preprocessing, original model config/checkpoint, FP32
policy, temporal context, every transform/inverse and addition order remain
the original code. No independent encode or TTA implementation is written.

The callback subclasses the established passive capture only to accept a
sparse frame set. It uses that exact13cube extraction/interpolation, axis
order, original final-anchor selection, all-original exclusions and FP32
memmap writer. It does not invoke the coordinate head, detector selection,
secondary model, association or ILP. Exact equality to the already verified
live passive field is mandatory before this shortened path is reusable.

Interface (root supplies the global Python and finite queue):

```text
python src/c063_frozen_spatial_extract.py controls --output-dir STATE_DIR
python src/c063_frozen_spatial_extract.py prove --movie 44b6_12dfb391 --output-dir PROOF_ROOT --reference-dir experiments/candidates/c063_frozen_spatial/capture
python src/c063_frozen_spatial_extract.py prove --movie 6bba_05db0fb1 --output-dir PROOF_ROOT --reference-dir experiments/candidates/c063_frozen_spatial/capture
python src/c063_frozen_spatial_extract.py extract --movie STEM --output-dir EXTRACT_ROOT --proof-dir PROOF_ROOT
```

The first two proof commands each perform one fresh primary-prefix inference.
They require that the matching full capture has passed original/passive/cached
ILP and actual official zero checks. They compare every requested complete
feature-field SHA/shape/context/anchor count, the entire FP32 feature file and
all numeric cubes, selected/all-original CSVs and every target NPZ array.
`proof.json` is created only after exact equality and a final input rehash.

The `extract` verb requires both benchmark `proof.json` files. It verifies
their extractor/capture/production AST source and every shared source/config/
checkpoint/notebook hash before proceeding. This prevents a later changed
checkpoint or feature implementation from borrowing an old proof. Each movie
must execute in a fresh process; output folders cannot be overwritten.

Python API: `dependencies(stems=cap.STEMS, reference_dir=None, proof_dir=None)`,
`movie_plan(movie)`, `controls(out)`, `prove(movie,out,reference_dir)` and
`extract(movie,out,proof_dir)`. Dependencies include all files read by the
selected path and the optional proof/reference files. Whole-dataset C058
input artifacts are already pinned by their preserved baseline/label files;
the prefix does not reopen GT or perform fresh matching.

The exact raw image chunk list is saved per movie in `data_plan.json`:
for requested`t=0`, read0 and1; for requested`t>=1`, read`t-1` and`t`.
Deduplicate windows and chunks. The data are verified to have complete-frame
chunks `(1,Z,Y,X)` and default slash-separated keys, so every required image
chunk is `data/train/STEM.zarr/0/c/T/0/0/0`; both Zarr metadata files are pinned.
Missing/sparser storage fails rather than silently fabricating a frame.

Metadata audit across the original22 movies finds16,931 original known pairs,
9,113 complete-cube nonsynthetic anchors,1,945 requested windows and1,978
distinct image chunks;7,818 excluded original pairs stay unchanged. The
registered set for this path has2,089 existing concrete dependencies before
adding two-movie proof/reference artifacts. This inventory is not timing or
efficacy evidence. Live prefix timing becomes available only after proof.

Outputs match the parent learner contract: `features.npy`, `labels.csv`,
`all_original_pairs.csv`, `targets.npz`, `frame_provenance.json`, `summary.json`.
Additional receipts are `data_plan.json`, `prefix_ast.json`,
`generated_prefix.py`, `input_hashes.json`, `prefix.log` and, for proof runs,
`proof.json`. All are under`OUTPUT_ROOT/STEM`; there is no graph artifact or
new competition-score claim.
