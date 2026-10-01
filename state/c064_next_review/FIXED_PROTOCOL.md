# C064 axial-only fixed proposal and component gate

Frozen before real DeepCenter profiles are inspected. This is an information
probe, not an actual graph candidate or untouched-embryo evaluation. No knobs
may be selected using outcomes of this probe.

## Exact proposal

Read the unchanged256-row `state/c060_review_20260929/model/sample.csv` and
cross-check every `(stem,row,node_id,gt_id,t)` and original residual against
C058. Use the C058 final integer C023 `(z,y,x)` node centre. No new sample,
target rematching, class balancing, fitting, model/TTA arm or changed checkpoint.

Obtain the exact configured C023 DeepCenter full-frame heatmap. Use production
score-point conversion `z0=round(z)`, `yp=round(y/pool_factor)`,
`xp=round(x/pool_factor)`, with pool_factor4 and existing `SCORE_WIN_YX=2`.
Require the complete5x5 lateral region and all9 axial planes `z0-4..z0+4`
inside the real field. Boundary/no-field/nonfinite exclusions must be explicit
and retain their original point exactly. Only9 candidate displacements exist:
`(z-z0)*1.625um`, or−6.5 through+6.5um, each within the original7um radius.

For each plane take the maximum of the same5x5 XY region. Do not maximize
over neighbouring z planes, interpolate between bins, fit a centroid/Gaussian,
renormalize logits, multiply shifts or blend coordinates. A unique maximum
selects its z. Exact multi-plane ties (including a flat profile) return zero.
Original x/y remain exactly unchanged. An edge-of-window unique maximum still
uses the same rule; do not retrospectively add an endpoint confidence guard.

Record per-plane values, unique/tied/finite/support status, proposed z and
exact physical shift. Evaluate all256 original rows including unchanged rows.
No GT residual category, identity label or embryo name determines a proposal.
All-node ownership and image-boundary checks are reported as diagnostics,
not used as a tuned runtime filter. Other predicted nuclei remain unknown.

## Fixed strata and gate

Strata are defined once from the original residual: all256, good<=2.5um3D,
tail>3.5um3D, and axial tail |dz|>3.5um. Also report eligible/no-op/excluded
subsets, never substitute them for the all-original denominator. Save signed
prediction-minus-target z/y/x bias,3D/abs-z means and medians, improved/worsened
counts, and movie counts. Source44b6 and validation-selected6bba are separate.
Neither is an untouched test: DeepCenter best.pt was selected using6bba.

To qualify for later independently registered all22/all-node investigation,
**each embryo** must satisfy all of:

- Lower3D mean and lower abs-z mean on all original sampled rows.
- Lower3D mean and lower abs-z mean on original3D-tail rows.
- Lower3D mean and lower abs-z mean on original axial-tail rows.
- Good3D and abs-z means no worse (only1e-12 floating arithmetic tolerance).
- For each tail stratum, at least two movies show improvement in both3D and
  abs-z means. An empty required stratum or insufficient movies fails.

No effect-size threshold pretends to forecast+.002–.003. Ownership-conflict
burden and the source/validation bias require independent review even on a
numerical pass. A pass means `component_review_required`, never automatic
graph integration or submission. A fail closes the registered fixed probe.

## Mandatory implementation controls

1. Source AST/checkpoint/config/manifest/sample/labels/baseline hashes, exact
   z coordinate evidence, configured C023 heatmap/TTA/runtime, frozen model
   state, and actual full-frame image hashes. The old XY inverse ambiguity
   remains documented and is never resolved by GT performance.
2. CPU synthetic **heatmap** controls for centre, positive/negative z maxima,
   all9 candidates and physical signs, exact ties/flat zero, missing real
   support and nonfinite rejection. These are arithmetic controls only.
3. An off/zero proposal preserves every original ID, x/y/z and residual.
   Strong assertion that no proposal changes x/y or exceeds6.5um z.
4. Max of the new per-plane profile restricted to production z±1 equals the
   unchanged `deepcenter_score_point` on the same synthetic field. Exercise
   fractional native XY phases and Python's tie-to-even `round` exactly.
5. Real benchmark uses two deterministic requested movie/frame pairs: the
   lexicographically first pair from each embryo, chosen before outcomes,
   solely timing and numerical/source identity. It
   may save reusable heatmaps with full provenance but must not report or
   select efficacy from that subset. Estimate all231 unique frames from its
   measured cost with conservative overhead; no head fit or optional benchmark tuning.
6. Saved profile recount must regenerate exactly the same proposals,
   original-pair errors, no-op counts and fixed gate. Group overlapping
   sample rows honestly;231 frame reads/22 movies remain two embryos.

The implementation should use original evaluator namespaces and raw readers.
No new scorer, custom heatmap model or alternative source copy is warranted.
Any later graph study requires original writer and actual official controls;
these component metrics alone are not a graph-score result.
