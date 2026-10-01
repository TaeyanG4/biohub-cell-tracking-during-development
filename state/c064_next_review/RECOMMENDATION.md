# One next bounded experiment: frozen DeepCenter axial location information

2026-09-29, after C063 component failure. This is a recommendation only:
no model inference, training, candidate source change or queue was launched.

Recommend **one fixed axial-only DeepCenter information probe** on the
already pinned256 C060 diagnostic points. Keep every C023 x/y coordinate and
node/GT identity unchanged. Use the existing frozen full-frame learned centre
heatmap to propose only z. This is an inexpensive new measurement; there is
currently no evidence that it produces a+0.002–0.003 score gain.

## Why this survives the duplicate and provenance review

C058/C060/C061/C063 trained new absolute-coordinate components and did not
preserve good opposite-embryo points. Their failed guards, epochs, strengths
and decoders should stay closed. C063 preliminary recovery evidence suggests
tail improvement with overall/good harm, which supplies no reason to tune it.
C023 already uses DeepCenter as a repair acceptance score, collapsing a local
heatmap region to a maximum and discarding its spatial position. Its frozen
learned **axial position** is a distinct unmeasured information source.

The prior full3D DeepCenter proposal remains blocked: historical checkpoint
source revision is unbound and the shipped XY label inverse conflicts with
the README's pooled-pixel convention. The independent current geometry review
loaded the actual checkpoint safely and confirmed that limitation. It also
confirmed that **both candidate conventions use z_original=z_index**. The
proposed probe never inverts XY or chooses a convention from GT performance.
Its XY query is exactly C023's existing score-point window, an operational
inference rule rather than a new claimed historical label-coordinate mapping.

This does not resolve all provenance limitations. The actual DeepCenter split
is71 training movies, all44b6, and128 validation movies, all6bba; best.pt is
epoch2 selected using6bba. Thus44b6 is source-domain and6bba is validation-
selected, neither a new untouched domain. Report those labels explicitly.
See `geometry.md` and its metadata evidence for the independent source check.

## Fixed first probe

1. Reuse `state/c060_review_20260929/model/sample.csv` exactly:256 original
   IDs,128 per embryo. This is a tail-enriched diagnostic sample, never a
   deployment whitelist or an estimate of population score. Its22 movies
   contain231 unique movie/frame reads (107/124 by embryo). Retain all rows,
   including abstentions/exclusions, with their original C058 targets.
2. Build the unchanged C023 replay namespace and use its actual
   `deepcenter_heatmap_for_frame`, original checkpoint/config, normalization,
   numerical policy and configured TTA. Reuse the existing raw reader and
   frame cache. Do not introduce a second heatmap model, normalization or TTA arm.
3. At the original C023 anchor, reuse the score-point XY centre
   `round(y/4),round(x/4)` and its existing2-grid-cell XY half-window. For each
   **individual z plane**, take max over that same5x5 XY window. Do not call
   `deepcenter_score_point` once per plane, because its additional z±1
   maximum would manufacture axial plateaus. Original x/y never change.
4. Consider integer z planes whose displacement from the original integer z
   is within7um: offsets−4..4, or±6.5um. Require full real support for that
   registered axial/XY window; excluded rows remain unchanged. Choose its
   unique maximum, with exact ties retaining the original z. No confidence
   cutoff, shrink scalar, interpolation decoder, radius/window sweep or
   alternative conversion is introduced. This is the one declared proposal.
5. Record its raw proposed shift, all-node Voronoi ownership/ties and image
   bounds as diagnostics. Do not select acceptance using GT, a good/tail mask,
   embryo identity or measured score. Unknown neighbouring nuclei are not
   biological negatives. Report paired physical3D and abs-z errors and signed
   bias on original identities, source/validation-selected separately.

The sample contains good<=2.5um counts66/57,3D-tail>3.5um counts58/64, and
absolute-z-tail>3.5um counts12/18 (44b6/6bba). These are enough for a falsifiable
signal check but remain small correlated subsets of two embryos. Every
original point must remain in its denominator; no favourable examples are
selected after extraction.

## Required controls and stop condition

Before real predictions, verify the exact source AST, checkpoint/config and
runtime bindings; reuse existing heatmap and scorer, not a new implementation.
An artificial heatmap with a declared z maximum must give the known signed
integer displacement, constant/tied profiles exact0, and boundary cases exact
no-op. Assert x/y, node/GT IDs and original targets never change. Verify the
new per-plane profile maximum agrees with the existing score-point result
when maximized over its original z±1 span; this proves the XY slicing contract.
Save raw per-plane profiles so the fixed proposal can be independently recounted.

Continue only if **both embryo groups** improve original-pair3D and abs-z
means overall and on3D/axial tails, with good3D/abs-z means no worse. Require
tail improvement in more than one movie per embryo, and independently review
ownership-conflict burden. A fail closes this fixed probe; no peak threshold,
different lateral window, coordinate mixture or decoder tuning follows.

A pass is only a reason for a separately registered all22/all-node test with
GT-free runtime eligibility and original-writer/official zero controls. It
does not establish biological identity, graph recovery, or leaderboard gain.
If upstream coordinates change, regenerate features/candidates/Transformer/
ILP; stale association caches are invalid. C023/C024 and accepted C046 remain
unchanged, and no submission is justified by the256-point probe alone.

## Cost and alternatives rejected for now

Implementation/control planning allowance:20–40minutes using existing APIs.
Compute is231 full-frame heatmaps, with at most8 existing XY TTA forwards per
frame; maximum1,848 forwards. This is **not measured yet**. Time a fixed first
eight requested frames and extrapolate231/8, then set a finite ETA. A few
minutes to tens of minutes is plausible on the4070TiSUPER, but the benchmark
must decide. Before any inference, the implementation fixes the benchmark to
two frames, the lexicographically first requested frame from each embryo,
reused by the full probe, with conservative extrapolation overhead. This
replaces the planning estimate of eight frames above. No training/feature-cube
storage is required. Local execution is
the short critical path; a Kaggle upload/queue offers no demonstrated saving.
Any hidden phase needs the existing Windows notifier and separate review wake.

Dense nonrigid optical flow is technically distinct from C036 translation NCC,
but currently has no measured headroom: the tracker already smoothly follows
unknown continuations at74/75 tail-directed views, while the official matcher
often switches to another path. More smooth motion can reinforce the same
identity mismatch. Existing local-neighbour flow, current-pair ILP seeding,
global stabilization, frozen-frame correction and relink gate sweeps are
already tested in C011/C018/C019/C020 and HANDOFF21–23. Another neighbour field
is not a supported fast next candidate. C062's temporal appearance probe also
had zero confirmed additional rescues and stays closed.

C059 temporal division morphology is still distinct, but it is deferred and
less aligned with the user's localization priority. Its current draft calls
an absent `c059_temporal_features.py`; it is not a ready runnable experiment.
The two-movie evidence covers four annotation windows and has strict daughter
size ratios0.657,1.037,1.778, with~98% unknown triples. Full paired48/56-feature
extraction and opposite-embryo precision evidence are needed; no universal
shrink rule or fast score gain is established. It should not be silently
reopened merely because several localization recipes failed.

If the axial DeepCenter source-binding/control requirements fail, report that
there is no presently supported fast localization experiment. Do not substitute
another architecture/guard guess or fill a submission slot without evidence.

Evidence read: HANDOFF current history and21–30; remaining_methods
RECOMMENDATION/association/geometry/deepcenter reviews; research_reset structural
and temporal reviews; localization identity/continuation/context reviews;
C062 FINAL_REVIEW; C059 draft; current independent DeepCenter metadata review.
