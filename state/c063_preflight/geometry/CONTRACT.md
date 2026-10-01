# C063 passive frozen spatial capture

`src/c063_frozen_spatial_capture.py` implements only CPU controls and live
passive feature capture. The root owns any GPU execution/queue. No GPU command
or training was run while writing this adapter. Python AST parsing and the
CPU-only controls passed under `geometry/local_controls/controls.json`;
all384 concrete dependencies exist. Two-movie inventory:588/938 eligible
original pairs,43/65 tails;194/277 excluded pairs retained unchanged.

Interface:

- `dependencies(stems=STEMS)` returns concrete source/model/config/head,
  notebook, reference graph/label, metric/DeepCenter and complete raw movie
  Zarr/GEFF/cache input paths. A single movie string is also accepted.
- `controls(out)` / `controls --output-dir DIR`: actual production trilinear
  function on a synthetic affine field, exact cube orientation and unchanged
  wrapper return/field checks, plus two-movie eligibility inventory. No encode.
- `capture(movie, output_dir)` / `capture --movie STEM --output-dir DIR`:
  creates `DIR/STEM/`; refuses overwrite. The process must be fresh for each
  movie to avoid imported predictor/global-cache leakage.

The root-approved input contract is **FP32 `(N,32,13,13,13)`, ordered channels,
z,y,x**, sampled at feature-grid offsets `[-6,+6]` on each axis. The field is
the original C023 primary32-channel eight-XY-view mean at the first-seen
detector time window. Anchors are the original **final integer C023** positions
from C058 official pairs, converted exactly to `(z,y/4,x/4)`, including quarter
XY phases. No new native XY detail or half-pixel translation is introduced.
Spacing is1.625um; intended valid3x3x3 head output is11x11x11 offsets[-5,+5].
All original7um3D labels fit the output coordinate range. No spherical label
mask is applied: fractional7um targets can need vertices outside that sphere.

All original known pairs are retained in `all_original_pairs.csv`. Eligibility
uses only nonsynthetic provenance and complete13cube support in the actual
feature-map bounds. Excluded rows have `cube_row=-1`; they must remain unchanged
in later paired reports. The historical C058 raw eligibility is preserved as
`historical_c058_raw_eligible`, never reused as C063 support.

Per-movie data:

- `features.npy`: FP32 memmapped cubes, rows in ascending original label order.
- `labels.csv`: selected rows, with `cube_row`, original node/GT/baseline row,
  time, residuals, native and fractional feature centers, signed3D targets.
- `targets.npz`: `node_ids`, `gt_ids`, `times`, `baseline_rows`, `cube_rows`,
  `centers_native`, `centers_feature`, `targets_um`, `targets_grid`.
- `frame_provenance.json`: full first-seen field SHA and shape, encoder frame
  pair/index, anchor count and unchanged original-head result SHA per frame.
- `input_hashes.json`: before/after verified source inputs.
- `receipt.json` and identical `summary.json`: sizes, counts, exclusion/tail
  counts, full control results, source model states, hashes and phase seconds.

Live proof runs the full exact C023 predictor twice: original head and passive
wrapper. Candidate mode and the original public head stay active. The wrapper
calls the original head, uses only read-only production interpolation, returns
that same result, and verifies the complete field hash did not change. It
records every first-seen frame, writes cubes in batches of at most8 anchors and
requires exact centers against an independent production lookup.

The live runs must have bitwise-equal complete coordinate/low-detection/admitted
edge caches and semantic ILP graphs. The passive output must also equal the
existing matched FP32 C023 cache and ILP reference. The existing C058 helper
then replays that **fresh live** graph/cache through unchanged C023
postprocessing, original integer rounding, actual organizer matching/scoring
and C055 reference parity. All final IDs/coordinates/edges/synthetic flags and
actual official metrics must equal the original C058 reference. These are
no-intervention proofs; no new score is claimed.

The adapter copies the original repository only into its own fresh output
directory using the existing `prepare_repo`; no pinned sources are edited.
Primary/secondary state dicts must remain unchanged in both live runs.
Prediction logs, cubes and verification artifacts stay under each movie.

The prior spatial review at `state/c062_preflight/geometry` remains relevant
background despite C062 having taken the distinct trajectory direction. C062
trajectory is closed/inconclusive and is not repeated here.
