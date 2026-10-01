# C061 independent code review

2026-09-29. Read-only review of the new driver, model and data module.
No fit, model inference, queue change, or candidate-source edit was performed.
One read-only pandas check validated registered manifests and the continuity
join. This review does not claim end-to-end execution of the new queue.

## Blocking finding

**P1: analysis selects a nonexistent continuity column.**
`src/c061_direct_axial_study.py:221` selects `node_id` from
`state/localization_diagnosis_20260929/continuity.csv`, whose actual node key
is named `p`. This raises `KeyError` after training and evaluation complete.
Rename `p` to `node_id` before selecting fields, assert unique `(stem,node_id)`
keys, and retain the validated many-to-one merge. A read-only corrected join
on the registered real manifest has zero missing continuity records.
The parent was notified immediately. Fix and verify this before source pinning.

## Verified behavior

- **Actual folds:** `source_data()` filters the registered direct manifest by
  biological embryo and requires exact extracted file-list equality. The
  training sampler picks one source movie and only points within that movie;
  each actual batch asserts exactly four examples of each correction class.
  The two final checkpoints record source embryo, full source stems, recipe
  and1200 steps; reload validates these and uses strict state-dict loading.
  Target-embryo predictions are never read by optimizer/checkpoint selection.
- **Interfaces:** `data.select/load_manifest/extract_one/input_paths`, the
  expected `direct`/`real` file naming, `AxialLocalizer`, `shifted_batch`,
  `reflect_xy_batch`, `loss`, `decode` and optimizer interfaces agree with
  driver calls. Registered output shape is21x49x49 for direct crops and
  13x49x49 for actual prediction anchors. The training/source-loader special
  case for `benchmark_data` is confined to the disposable timing directory.
- **Signs:** expanded slices start at `4+s`, target correction is `-s`,
  decoded class maps to native[-4,4], synthetic signed error is `(s+dz)*1.625`.
  Real labels store `GT-prediction`; corrected real target residual is
  `old_dz-predicted_shift*1.625`. Reported signed error is its negative,
  consistently representing prediction-minus-target in both evaluations.
- **Integer contract:** the data module checks all GEFF t/z/y/x values for
  exact integrality before selecting or casting. Direct centers equal GT,
  so the driver's integer synthetic error formula is valid for this dataset.
  Real labels are checked against original GT and original saved residuals.
  The model's fractional-label support is not silently exercised here.
- **Ties/zero:** model decoding uses top-two exact equality and returns zero
  on any nonunique maximum. The zero-initialized head therefore cannot move
  all nodes by -4 through an argmax-index accident. Existing model and real
  pixel preflight reports verify zero output, every shift sign, gradients,
  class decoding and two-movie pixel parity. I read those reports rather
  than rerunning model controls.
- **Fixed original identities:** real predictions alter a local coordinate
  array only; graph files are not modified. Real residuals retain original
  node-to-GT pair and x/y error. Ownership/outside-image flags are diagnostics
  and do not suppress proposals. No GT rematching or successful-case filter
  is used.
- **Pandas operations:** after the continuity-key fix, grouping and masks do
  not use known conflicting DataFrame methods such as `.tail`. Boolean
  eligible/tied flags are explicitly created. The real-pair duplicate check
  is keyed by source model, stem and node. The continuity merge's cardinality
  is appropriate and was checked on actual input keys.

## Actual manifest and denominator checks

The registered direct manifest contains6,364 rows from199 movies:
2,268 from44b6 and4,096 from6bba. `(stem,gt_id,t)` duplicates are zero.

The real manifest contains9,294 pairs from all22 planned movies. There are
no `(stem,node_id)` duplicates, no entirely empty eligible movie, and exact
real-stem parity with the data plan. The original C058 labels contain16,931
known pairs. Consequently, the current `analyse()` reconstruction yields
9,294 eligible and7,637 ineligible-unchanged pairs per source model. Original
noneligible pairs keep original3D/absolute-z errors and zero displacement.

The continuity input uses `(stem,p)` and has no duplicate keys. Renaming `p`
to `node_id` and joining the9,294 real manifest rows preserves all rows and
finds every continuity label. Therefore there is no underlying data mismatch;
the blocking problem is the column name in the driver.

## Gates and interpretation

The synthetic gate requires each nonzero shift to improve separately in
each opposite-embryo direction. The zero-shift gate allows up to1.625 um mean
absolute error, as explicitly written in the current README. That is a
permissive diagnostic tolerance, not zero-preservation. Preserve/report the
actual zero-shift mean and class distribution even if this gate passes.

The real gate checks eligible overall, original3D tail>3.5 um, axial
tail>3.5 um, and good3D<=2.5 um in both directions. All three nongood strata
must improve both mean3D and absolute z; the good stratum must worsen neither
measure (within1e-9). The exact eight-row gate and positive denominator checks
prevent an empty real subgroup from passing. All-original and unchanged
denominators are separately reported. The raw row files retain class outputs
for later confusion/frequency review, although no dedicated confusion matrix
is written by this driver.

Passing these gates remains a component result only. The decision writes
`official_score_computed=false`, `graphs_modified=false`, and requires review
before a separate integration study. This boundary matches the declared task.

## Nonblocking hardening before pinning, if convenient

The current data includes all22 real movies, so there is no observed omitted
movie. Still, `real()` iterates a file glob while denominator reconstruction
iterates only observed `(source,stem)` groups. Asserting real file-list parity
with `real_selection.csv` and exact evaluated pair-key parity would catch an
accidentally missing whole crop file before a partial result is accepted.
Likewise, assert the expected36 synthetic summary rows to make full coverage
explicit. These checks should not change selection, outputs, or scientific
criteria.

Consider comparing current preparation inputs with the stored
`data_input_hashes.json` digests before registering new hashes. Extraction
already checks its registered source/metadata subset; an explicit preparation
comparison would also bind all originally inspected GEFF data to the selection
receipt if an input changed between selection and registration.

## Reviewed source snapshot

- `c061_direct_axial_study.py`:
  `8a5cc67b498940d6089402f4ecc51c5809b6c4c9e60804c3c1409182d6e45d2a`
- `c061_axial_model.py`:
  `71d1a9cd131323801350e20ef70762cf7c6a59d9794528d6a1fc57fe4ca5c2ff`
- `c061_axial_data.py`:
  `1819b6c08a6b48688d80ff2e0afbfa18b44264dab27488535d729e41ce5bd4b6`

`preflight.json` had not yet been written when this review read the candidate
folder. Existing `state/c061_preflight/model_controls.json` and
`real_controls.json` report passed controls; launch registration remains the
parent's responsibility. No other blocking defect was identified in this
bounded review.
