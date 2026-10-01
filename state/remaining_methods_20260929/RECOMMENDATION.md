# Remaining methods after C060

Decision date: 2026-09-29. Deadline: 2026-09-30 09:00 KST.
This is a research prioritization record, not a registered or launched candidate.
No improvement or submission readiness is claimed.

## Recommendation

The highest-priority new learning task is direct known-point axial recentering,
followed by a fixed-pair test at real C023 prediction anchors. Use known GEFF
points to define the target rather than prediction-to-GT matches to create
source labels. Deliberately offset real crop centers along z and learn the
annotated point's location. A broader image representation, especially an
immutable pretrained spatial feature field, is worth testing if a small direct
image challenge cannot recover these labels. Do not choose among repeated
architectures or thresholds by the opposite embryo's results.

The existing full-frame DeepCenter location field offers a cheaper preliminary
information test without fitting. Its coordinate convention and exact checkpoint
provenance must first be established. This is a distinct question from C055/C056
node readmission and the historical StrongUNet detector-correspondence sweep;
those experiments do not establish useful displacement of existing C023 nodes.

## Priorities and decision points

| Priority | New information or supervision | Smallest useful test | Main limitation |
|---|---|---|---|
| Cheap prerequisite | Frozen DeepCenter spatial location, currently reduced to a repair score | Fixed existing 256-point sample, exact original identities, one predetermined coordinate conversion and proposal; tail/overall/good residuals by embryo | Checkpoint/label convention must be known; detector saw both embryos; sample is tail-enriched |
| 1 | Direct known-GT point supervision with balanced large z displacements | Whole-embryo fits; first exact x/y diagnostic, then actual C023 anchors with identity-frozen real residuals | Synthetic displacement recovery can succeed without fixing real detection errors |
| 2 | Frozen production UNet spatial feature neighborhoods | Auxiliary location head with unchanged encoder/features; opposite-embryo real-pair axial and good-point checks | May preserve detector bias; requires exact production capture and zero controls |
| 3 | Multi-frame identity evidence | Frozen C048 encoder: fixed three-frame history versus candidate future mean embeddings, paired against same-model single-frame scores | Existing trajectories may already contain identity errors; appearance may not distinguish neighbors |

Priority 2 is an alternative representation for the location task, not an
automatic second training sweep. Priority 3 can supply an independent bounded
diagnostic once localization work has a concrete finite phase. C059 remains
deferred; this review does not restart it.

## Measured feasibility and interpretation

The all199-movie metadata inventory finds 72,384 known annotation records with
real image support for 13x49x49 crops at z offsets -6.5,-3.25,0,+3.25,+6.5 um:
14,491 in 44b6 and 57,893 in 6bba. The old22 subset supplies 8,073 of these.
These are correlated, overlapping records from just two biological embryos,
not 72,384 independent cells or extra independent validation domains.

C060 leaves the actual official all22 score exactly unchanged. Its accepted
4,085 moves include only five z changes and zero z changes among the 109
originally matched moved points. Its own-source large-offset fit is weak, and
unguarded modes harm good points in both embryos. This does not justify simply
relaxing its agreement guard or changing its prior/epoch/decoder.

The 75.6% figure is 1,047 of 1,385 missed GT links whose endpoint detections are
both present and which touch a >3.5 um original residual. It is not the share
of all nodes that are wrong, nor a demonstrated recoverable fraction. Most
large-residual nodes have their known adjacent links correct. Every component
review must distinguish persistent offsets from matched-identity switches.

## Path from a useful component to a scored candidate

Require actual real-anchor improvement in both embryo directions, including
large-z tails, while protecting already-good identities. Synthetic/oracle-x/y
results are diagnostic only. A successful component then earns a separate
pre-association experiment: changed coordinates require fresh feature indexing,
candidate edges, Transformer scores and ILP. Frozen final-edge scoring can miss
an association benefit when the GT matching is unchanged; stale downstream
features after coordinate edits are not a valid test.

Reuse the existing writer, official evaluator, zero controls, all22 and unchanged
97 review, portable checks and actual T4 verification. Unknown cells remain
unknown, never background negatives. No evaluation-derived node masks, embryo
deployment router, threshold sweep or quota-filling submission. C023/C024 remain
the 0.954 anchors and C046 is already submitted.

New background phases require an actual timing benchmark, finite pinned queue,
Windows completion/failure notifier, and the existing ETA-10/20-minute review
schedule. No queue or ETA is established by this document.

## Evidence

- `localization_review.md`: direct supervision and frozen spatial feature review.
- `geometry_review.md`: DeepCenter, coordinate contract and raw geometry review.
- `association_review.md`: trajectory appearance and uncertainty review.
- `gt_support.csv/json`: all199 real-support counts.
- `deepcenter_contract.json`: algebraic label-versus-pooled-pixel convention check.
- `track_context_inventory.csv/json`: available multi-frame predicted contexts.
- C060 `FINAL_REVIEW.md` and `state/c060_review_20260929/model/science`.
