# Independent C062 prototype code review

2026-09-29. Reviewed `src/c062_tracklet_probe.py`, the reused C038 recorder and
appearance stage, C035 crop path, C058 passive baseline capture and the user's
HANDOFF duplicate-check report. No draft source was edited. Executed only a
small CPU fixture under this directory; no image/model inference, fit, graph
mutation or queue launch was performed.

**Verdict:** the core frozen three-history/three-future diagnostic is
scientifically coherent and distinct from the completed single-frame arm.
The prediction-only selector passes the topology fixtures below. Before
promotion or unattended execution, finish the proposal-level decision audit
being added by the other reviewer and verify original-center provenance in
the actual passive capture. These are specific validation requirements, not
grounds to change the scientific formula or tune thresholds.

## Valid scientific and data-flow properties

- `AllNodeRecorder.begin` replaces the original GT-dependent recorder entry
  point, enables every predicted node, and never calls matching. The recorded
  finite candidate sets come from C023's actual motion-relink passes.
  `select_groups` accepts only recorder/final-prediction/image-shape inputs.
  GT labels cannot choose the runtime groups or their temporal contexts.
- Source history is the actual unique predecessor chain `[t,t-1,t-2]`;
  each alternative's future is `[t+1,t+2,t+3]`. All nodes require original
  image-center availability, real crop support and no synthetic/fork/gap
  protection failure. If **any** candidate lacks context, the entire group
  is excluded. That preserves a paired common support rather than silently
  removing difficult alternatives from the three-frame score.
- The alternate child may have no incoming edge while still having a valid
  future. Such a target is permitted, consistently with later unused-target
  handling. Multiple parents, multiple children and nonconsecutive temporal
  edges are protected rather than treated as ordinary linear context.
- The chosen checkpoint is the immutable C048 model trained on the opposite
  biological embryo. Checkpoint hash, recipe, arm, plan hash and training-stem
  membership are checked. No fitting, model averaging or target checkpoint
  selection occurs. Exact original C048 crop/normalization is reused.
- Both methods use the same checkpoint and original per-node embeddings.
  A normalized mean of three vectors changes time evidence only. Duplicate
  vectors reproduce the single-frame cosine within declared floating-point
  tolerance and must produce the same proposal. CPU eval mode avoids a
  batch-dependent learned normalization state.
- `proposed` preserves C048's cosine-gain, first-versus-second margin and
  geometry-cost conditions. Every candidate admitted to this diagnostic has
  already passed the context protection. It produces a proposal only; it does
  not silently perform a unilateral target steal or graph assignment.

Known labels refer to the original saved official node/GT assignment. A source
needs one **annotated** adjacent successor to define its target identity.
Candidate label1 matches that known successor; label0 is a known distinct
annotated identity; label-1 remains unknown. A positive-reachable/known-negative
subset is used only for diagnostic reporting after runtime selection. It does
not become a deployment whitelist. Describe label0 as known-distinct relative
to the annotated successor, not evidence that the candidate is a false cell or
that every biological daughter has been annotated.

The base `frozen.capture` computes official baseline controls using GT before
returning. Therefore `gt_used_after_selection=True` is accurate for *group
labeling*, not literally every GT read in process time. The meaningful guarantee
is the absence of GT data flow into the selector or prototype scores, which the
reviewed interfaces preserve. Baseline evaluation does not invalidate that
guarantee.

## Issue 1: finish proposal-level evidence before promotion

At the reviewed draft's initial `analyse` implementation, the boolean signal
requires positive known-only rank rescues and nonnegative all-candidate rank
rescues. This can be true even when an unknown target remains top in every
group, the fixed margins reject every change, or useful proposals do not
improve over baseline. It is insufficient as a promotion gate by itself.

The other reviewer is explicitly adding proposal counters and distinct GT
identity accounting. The completed review should report and evaluate:

1. Actual single-versus-three **proposed** choices under unchanged margins,
   including confirmed rescues, harms, unknown-involved changes and unchanged
   choices. Merely taking a top rank is a separate diagnostic.
2. Proposals versus the original current C023 child, including beneficial and
   harmful known-distinct changes. Require useful proposed rescue evidence on
   both embryos, not zero-versus-zero satisfaction of a nonnegative test.
3. Unique `(stem, source_gt, successor_gt)` links and source identities behind
   apparent gains. Adjacent frames and crop overlaps remain correlated; these
   counts are not independent biological cells or embryos.

A passing paired proposal signal earns a separately registered collision-safe
graph replay. It is not an official edge count. Reciprocal-swap/unused-target
acceptance can still eliminate a good proposal or couple two changes; actual
official scoring and exact off controls are required after integration.

## Issue 2: establish original-center correspondence in actual capture

The draft selector requires `rec.nodes[id]` and final node time to agree, but
does not compare their spatial coordinates or object provenance. C058's prior
work explicitly established that postprocessing can recycle IDs. A CPU fixture
with an alternative's final position changed while preserving its ID/time is
accepted by the selector. This is an **unchecked assumption demonstrated by a
fixture**, not an observed mismatch in these two movies.

Before launch, either assert that all actually needed original centers refer
to the same final prediction and coordinates (with the expected un-stabilized
coordinate convention), or preserve and validate the correct object-origin
correspondence. If an intentional coordinate transform exists, document it and
prove image-center equivalence; do not simply replace original image centers
with stabilized positions. Checking exact final graphs against C058 alone
does not prove the recorder-to-final correspondence.

Also mirror the existing cropper's arithmetic when asserting support:
`patches` stores tzyx in float32 before rounding, while draft eligibility rounds
the float64 recorder values directly. Values near a half-voxel boundary can
round differently. A direct required-center/actual-pixel support control or
matching the existing float32 rounding convention resolves this without
changing the intended crop recipe. This matters because `crop_at` can pad
out-of-bounds support, whereas the new protocol promises real support.

## Executed topology controls

`topology_controls.py/json` exercise the actual imported selector with nine
predicted nodes and two alternatives. The valid group returns history3/2/1,
futures4/5/6 and7/8/9, and all nine required nodes. Ten perturbations correctly
reject the entire group: missing alternative future, synthetic source history,
synthetic alternative future, absent original alternative center, alternative
crop boundary, wrong candidate time, wrong context time, original time mismatch,
branched alternative context, and branched source. There are no GT labels in
these fixtures. The same-ID/time coordinate mismatch fixture records the
provenance limitation above rather than pretending it passed that requirement.

Selector source was hashed in the proof. The other reviewer is editing the
driver's analysis/benchmark functions concurrently; recheck selector changes
before treating this fixture receipt as covering a different implementation.

## Interpretation boundary

No evidence yet establishes that temporal prototypes improve ranking, accepted
edges, node identities or the leaderboard. C048's small/mixed result remains
the frozen control's limitation. Bad existing tracks can contaminate temporal
means, and strict whole-group eligibility can remove many difficult cases.
Report that coverage, all unknown choices and failures explicitly. Do not
respond to a negative result by adjusting the window, margin, normalization,
checkpoint or graph solver. The bounded duplicate check supports testing this
specific missing signal; it does not establish a success probability.
