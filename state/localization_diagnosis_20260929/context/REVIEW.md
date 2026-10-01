# C060 temporal-context feasibility and data adapter

2026-09-29. The initial feasibility pass used metadata and the22 immutable
C058/C023 baseline graphs only. It reused saved official labels; it did not
fit a model, recompute scores, edit a graph, or launch a queue. A subsequent
small data-adapter control read real images only to verify extraction.

## Runtime context coverage

A context is the actual previous/current/next prediction at consecutive times.
All three nodes must be nonsynthetic, every node in the triple must have
incoming and outgoing degree at most one, and every requested crop must fit
entirely inside the image. There is no padding, invented neighbor, GT filter,
or embryo routing in this runtime rule.

| Count |44b6|6bba|All22|
|---|---:|---:|---:|
| Nonsynthetic nodes |158,237|355,947|514,184|
| Central13×49×49 crop fits |95,250|190,801|286,051|
| Central15×57×57 expanded crop fits |88,492|173,297|261,789|
| Real nonbranching consecutive triple exists |142,798|325,697|468,495|
| All three13×49×49 crops fit |84,610|169,969|254,579|
| All three15×57×57 crops fit |78,590|153,928|232,518|
| Known labels in expanded triples |1,072|6,707|7,779|
| >3.5um labels in expanded triples |90|550|640|

Full inference context covers49.5% of real nodes. The smaller odd crops improve
spatial support, but a full temporal context still cannot cover all nodes.
Ineligible nodes need unchanged output, not a GT-dependent fallback.

## Training supervision and hard cases

Of7,779 known expanded-context centers,7,293 have both predicted neighbors
matched to known GT;17 have an actual context edge not present in the GT graph.
The fixed source-supervision rule requiring **both known neighbors and both
directed GT edges** yields:

| Source supervision |44b6|6bba|All|
|---|---:|---:|---:|
| Known consistent triples |1,021|6,255|7,276|
| >3.5um targets |58|371|429|

This filter belongs only in source-embryo training. Runtime eligibility remains
the GT-free topology/support rule. A missing GT neighbor is unlabelled, never
a false-cell or false-edge label. GT edge absence and different GT weak
components are separately recorded in `evidence.json`; neither establishes
biological truth beyond the annotation graph.

The640 expanded-context tail labels include483 whose available known GT links
are all retained by C023,149 with a missing known link, and8 without a matched
GT neighbor. Among the149 wrong-link cases,148 have at least one predicted
context neighbor with unknown GT identity. Thus source filtering removes
many of the hardest cases from supervision. It must not remove them from
evaluation: the proposed learner's ability to transfer to these cases remains
unproven. The inference crop includes699 known tails, including168 wrong-link
cases; diagnostic strata must never become runtime masks.

## Identity geometry: use directions, not an isotropic radius

The nearest other real final-node distance has median8.52um. A7um correction
ball reaches beyond at least one competing Voronoi bisector for489,740 of
514,184 real nodes (95.2%). Across these nodes, the portion beyond the nearest
competitor's bisector has median10.0% of the ball volume. This is a lower bound
on the union outside the self Voronoi cell, not a full multi-competitor volume
calculation; image boundaries are not included in that volume calculation.

An isotropic bound `min(7um, d_nearest/2)` has median4.26um. It would reject
339/640 (53.0%) expanded tail targets, including112/149 (75.2%) wrong-link tail
targets. Yet none of those640 targets lies outside the exact self Voronoi
region. A point can move farther than half the nearest-neighbor distance in
a direction away from that neighbor and still be closest to its own node.

The supplementary `all_node_ownership.json` checks the intended production
scope: **all** same-frame output nodes, including synthetic/ineligible ones.
Zero of7,779 expanded targets and zero of7,276 stable-source targets lie
strictly outside that cell. Four expanded targets (three tails) are exactly
on a boundary; three of those are stable-source targets (two tails). A
conservative tie-unchanged policy has that small documented unreachable set;
do not remove it using GT at runtime. Exact proposed-point ownership is
therefore more appropriate than the isotropic radius, but preserving a
geometric cell still does not prove biological identity.

## Implemented data API and checks

`src/c060_track_localizer_data.py` provides:

- `load_graph(stem)`: copy of immutable C058 baseline/provenance.
- `context_indices(graph, TZYX_shape, crop)`: previous/next row arrays and
  GT-free eligibility; invalid context indices are−1.
- `source_labels(stem, graph, context)`: known, both-edge-consistent training
  rows only, retaining original official offsets.
- `extract_one(out, stem)`: normalized real float16 crops shaped
  `N×3×15×57×57`, ordered previous/current/next and centered on each node's
  own predicted location. Companion NPZ contains `targets` inum, node/GT IDs,
  central and context row IDs, context node IDs, and physical context-center
  displacements. Each source image frame is read once per movie.

All22 runtime and expanded masks exactly match the prior independent metadata
audit. All22 stable source selections match the audit; input graph arrays are
unchanged. Direct GEFF GT edge reads matched the unchanged C023 graph loader
on all22 movies and16,551 edges (`gt_io_controls.json`). Separate-state smoke
extraction produced36 and94 examples using38 and71 unique frames in1.59 and
2.73seconds. Twelve saved channel crops exactly matched freshly reread
original normalized pixels; shapes, IDs, finite values and central displacement
were checked (`module_controls.json`). The smoke's NPZ key was `targets_um`;
the final adapter contract was renamed to `targets` at the driver's request.
The main driver's real two-movie benchmark validates that final contract.

No training has been run by this audit. Opposite-embryo validation, original
writer/official scoring and zero controls remain mandatory before promotion.
