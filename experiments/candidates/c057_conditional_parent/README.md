# C057 conditional parent competition

Registered 2026-09-29 following the user's request for another +0.002–0.003.
C053 public score 0.954 was reported by the user today; C054 remains unknown.

## New mechanism

C052 learns the correct parent only against annotated competing source cells.
Those comprise 1.3% of the source candidates on 44b6 and 6.5% on 6bba in
its saved training manifest. Most real competing detections receive only the
teacher preservation signal. On the fixed C02397 graphs, 1,332 counted false
edges have a matched child and an unmatched predicted parent.

For a child with a known annotated incoming edge and an identified parent,
the identity of its parent supplies a conditional association label. A source
cell outside that parent's identity ambiguity region cannot be its parent.
This is an edge identity constraint, not a label that unannotated cells are
false detections. C057 retains the original known-label target set and excludes
identity-ambiguous unknown candidates. Targets without a known parent never
receive this loss. Label provenance and the exclusion must execute and pass
before fitting. The exact fixed exclusion policy is saved in label_audit.json.

## Fixed comparison

Reuse C052 production packets, known-positive labels, 300 ordinary and 300
division batches, exact context sampling clusters, seed3701, lr1e-5,
teacher KL0.1, AdamW, cosine600 and the sole600-step checkpoint. Retain the
original forward objective, all inference fusion, ILP and C023 postprocessing.
Change only the supervised parent competition set. No threshold, checkpoint,
blend, reverse-loss or inference-rule sweep. Original studies stay immutable.

Train on one WHOLE embryo and evaluate ONLY on the opposite embryo. Both
public frozen detectors and the coordinate head saw these embryos, so this
tests transfer of the new component, not independent whole-pipeline validation.
No embryo-prefix router may be deployed.

## Controls and finite queue

Existing Queue and existing official notebook replay only. Twelve jobs:
label provenance, real original/off capture and loss parity, two fits,
fold/sample proof, two off replays, two inference/replay pairs, official analysis.
Require exact captured forward logits, original known-source loss parity,
finite new gradients, identity-mask assertions, all600 sampled steps per fold,
22 exact off rows and88 exact intermediate/final off graphs,22 frozen detector
arrays, pinned source/input/output hashes. Analysis compares against BOTH
C023 and C052 on22, heldout12, confirm10 and both embryos, including actual
edge/fork confusion and graph changes.

Estimated65minutes from launch (launch.json is authoritative);4-hour start-job
budget. Same biohub-t4 follow-up: first ETA minus10minutes, then20minutes.
Use a hidden process; no active agent wait loop and no Kaggle operations here.

## Review and continuation

A gain must come from actual edges/forks and survive opposite-embryo review.
Inspect C052-relative results to isolate the new objective. A small consistent
gain may justify an unchanged75 extension; probabilities or training losses
alone cannot justify promotion. If promising, use existing fixed deployment,
portable12/actual writer4 and actual T4 exact source/model/CSV/version checks.
Fresh shared quota and dedup precede already-authorized worthwhile submission.
Keep C023/C0240.954 as final picks unless stronger real evidence arrives.

The numerical target is not a prediction: four additional correct forks with
no false forks give +0.00203 to the97 division term, before associated edge/node
changes. Local improvement does not establish hidden leaderboard improvement.
