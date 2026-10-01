# Search for another +0.002–0.003, 2026-09-29

The strongest newly measured lead is C057: teach the existing Transformer to
distinguish the true parent from the actual detection candidates. C053's user-
reported public0.954 confirms no displayed improvement from C052's old objective.
C054's score remains unknown; no submission scores were polled.

## New evidence, not a repeat of a closed model recipe

The C052 supervised parent loss sees only annotated competing sources:1.31%
of source appearances in44b6 and6.50% in6bba. Teacher preservation sees the rest,
but the supervised labels do not directly distinguish them.

A new193-packet audit covers all145 division packets plus fixed temporal
quartiles from24 ordinary movies,1,834 positive links. The base model's captured
logits reproduce exactly. After excluding unknown detections inside the known
parent's7µm identity ambiguity region,50 real parent-ranking errors are invisible
to the old loss;36 are daughter links. Both embryos contribute. C052's existing
opposite-embryo models leave essentially all of these errors in place.

These are ranking errors, not a claim of50 recoverable final links or independent
events. They provide a specific new training mechanism with actual examples.
Source: `state/boost_audit_20260929/edge/REVIEW.md` and its hashed packet audit.

## Numerical target

The existing official C023 aggregation on97 movies is0.9404900129096955:
edge TP/FP/FN67758/2701/2774, division TP/FP/FN27/46/124.
Keeping all other terms fixed, +4 true forks with0 false gives+0.002030;
+6 true/+5 false gives+0.002631; +8 true/+10 false gives+0.003203.
These are division-term arithmetic scenarios, excluding accompanying edge/node
changes. They are not forecasts for the local experiment or the public board.
Executable arithmetic: `score_requirements.py` / `score_requirements.json`.

## Priority and alternatives

1. C057 changes only the supervised competition set, retaining positive labels,
   ambiguous-identity exclusions, two whole-embryo folds, original sampler,
   optimizer,600steps and complete inference/postprocessing. Real loss/control
   checks precede training; official22 graph evidence precedes a75 extension.
2. Explicit temporal nucleus size remains a genuinely new secondary signal:
   predicted parent/daughter trajectories at t+3 were absent from C049's3-frame
   CNN and C050's peak/mean features. It needs production-candidate measurement
   and opposite-embryo precision evidence. No new morphology study launched here.
3. Hengck v12 independent association model is distinct from C056's v5 detector,
   but integration and complementary ranking are unverified. No download or
   model experiment was started.

The reverse-direction train/deploy mismatch is real, but only7 annotated
daughter links cross downward at the fixed threshold after full fusion while12
cross upward. That evidence does not justify removing fusion or prioritizing
reverse-only retraining.

The public-source audit reused the existing radar and made one bounded refresh
of9 third-party references (6 successful,3 SSL errors). No newly proven best
score improvement appeared. Own submission status/score was never queried.

## Limits and next review

All frozen public detectors/head saw the two known embryos. Whole-embryo folds
validate only the new component's transfer. The pilot must report signed effects
against both C023 and C052, actual edge/fork confusion, graph changes, and
detector/control parity. No mask or GT information enters production inference.
Only a worthwhile candidate that passes existing local/portable/T4 provenance,
full-precision score and fresh-quota/dedup checks may use prior submission
authorization. C023/C024 remain the scored0.954 anchors.

Some concurrent `state/perf_search_20260929` notes contain incorrect division
ceiling arithmetic and label C054 counts as C023. The exact C023 counts and
official aggregation recomputed above supersede those numbers for this review.
