# Predicted-track temporal nucleus size: measured feasibility, not a gain

2026-09-29 09:54 KST. This is a new feature hypothesis worth a bounded
classifier pilot, but the two-movie measurement does not establish useful
precision. No model was fitted, no graph changed, and no remote action occurred.

## What is genuinely new

C049 used GT-centred image crops at t-1, t, t+1. C050 reused the C016 48 features,
including local peak/mean intensities and t+2 sister separation, but no explicit
nucleus-volume trajectory. Follow **predicted** parent predecessors at t-3..t-1
and both predicted daughter branches at t+1..t+3; measure size separately from
brightness. GT may label/evaluate triples, never supply feature centres or paths.

The existing cached public notebook `zhincez__a-dividing-nucleus-gets-smaller-not-dimmer`
provides `probe`: an 11x11x11 voxel box, local p10 background, peak minus
background, and count above background + half peak contrast. Its exact function
was extracted by AST and reused unchanged. Source SHA256:
`1dd8e153fdb16061f61d107a1357fca0ab1839eac0e51f00a774ad6465d6d9ab`.
The original EDA has no saved outputs; its reported shrinkage is not evidence
of production-candidate precision.

## New real-data measurement

`feasibility.py`, `feasibility.json`, and `probe_rows.csv` reproduce this audit.
Two predeclared dense heldout movies, one per embryo, use the existing exact
C023 final control graph snapshots from C055. Existing `enumerate_candidates`
enumerated all geometry-qualified triples at all times; existing `MetricLabeller`
labelled them. Ancillary intensity/DeepCenter/edge features were zero because
this measurement only consumes the enumerated P/D1/D2 identities; geometry
selection does not depend on those ancillary values. A future learner must use
their real values. This final-graph stage differs from C050's C012 pre-stage.

| Movie | Predicted triples | Known negative | Positive rows | Unknown | Full predicted t-3..t+3 paths among positives |
|---|---:|---:|---:|---:|---:|
|44b6_12dfb391|244,237|4,473|3|239,761|1/3|
|6bba_05db0fb1|584,874|10,833|7|574,034|7/7|

There are **4 GT annotation windows**, not 10 independent positive events.
The 10 positive rows are 4 strict and 6 early alternatives. Only 3/4 strict
rows have full predicted trajectories. Unknown is label -1 and stays unknown;
it was never converted to a biological negative.

All positives plus deterministic SHA256-first 32 known-negative and 32 unknown
triples per movie were probed: 138 rows, 118 complete/valid paths, 11 of those
touch an image boundary. All feature centres come from predicted nodes. Unique,
consecutive predicted predecessor/successor paths are required; missing paths
or forks are recorded, never silently replaced by a GT path or first child.

The usable early 44b6 example has daughter-to-parent size ratio 0.608 at t+3.
The three usable strict 6bba examples have ratios **0.657, 1.037, 1.778**.
Thus even these few real examples do not support a universal shrink rule.
Do not report AP, population precision, biological significance or score gain
from this label-stratified image sample. Positives are overlapping alternatives.

Total measured movie times: **10.89 / 23.44 seconds**. Enumeration consumed
7.81 / 17.85 seconds. Probing 497 / 492 unique predicted nodes required 94
frames per movie (0.789 GB uncompressed each) and 1.42 / 1.46 seconds including
frame reads; this is a local warm-cache measurement. Full-candidate feature
extraction needs far more probes, so these times are not a full 97-movie ETA.

## Available data and effective independence

The 199 movies are only **2 biological embryos**, 71/128 movies. Existing
division census: 26/125 annotation occurrences. C052's exact-context audit
finds 25/125 groups (150 total), with 1/9 annotations lacking its bounded raw
context. These are at most 150 distinct exact contexts, **not 150 established
independent biological events**; overlap beyond exact context remains unresolved.

Existing C050 caches contain 221,133 known rows, 50,853 predicted parents and
94 strict positives (18/76) in 97 C012 movies. They are useful metadata/control
evidence, not C023 integration proof. C050 opposite-fold precision at fixed0.9
was 0.0375 / 0.0588; C049 fixed0.5 precision was 0.1563 / 0.2453. Repeating those
unchanged inputs is unsupported. The new ingredient is the measured future
size block, tested against a same-data original-feature control.

## One fixed, implementable next pilot

1. Use the saved C023 97-movie graph snapshots at one declared stage, or capture
   that stage with the existing replay wrapper. Reuse `enumerate_candidates`,
   `MetricLabeller`, real cached edge/detection features and existing raw reader.
   Deduplicate probes by (movie,node), process each frame once. Include every
   candidate in coverage counts. Retain label -1 outside supervised fit; use
   the existing strict-positive policy and drop non-strict positives rather
   than mislabelling them. Keep early candidates as separate diagnostics.
2. Add one fixed 8-column block to the original 48: mean daughter/parent
   half-height volume ratios at t+1 and t+3, corresponding peak-contrast ratios,
   daughter volume asymmetry at t+3, complete-track flag, valid-probe flag,
   boundary flag. Parent reference is mean t-3..t-1. Missing ratios are neutral
   one with explicit flags (asymmetry zero); no lag/radius/threshold sweep.
3. Two WHOLE-EMBRYO outer folds; paired original48 versus augmented56 fits.
   Reuse `division_scorer_train.fit/predict/per_parent`, fixed C050 recipe
   (60 epochs, hidden32, lr.002, wd.001, seed0, pos_weight capped100, threshold.9).
   These paired controls isolate the new input, not a revival of the old arm.
   Save embryo/checkpoint/sample proofs. Do not use movie-random folds or a
   deployment embryo router. Original fold fits took about 7/19 seconds; real
   feature extraction dominates. Benchmark one full movie before queue ETA.
4. Report both-fold fixed-threshold per-parent precision/recall, descriptive AP,
   strict-event coverage and unknown-output burden. Multiple temporal versions
   of a fork must not become multiple recovered events. A plausible go gate is
   precision about0.3 and restricted recall about0.3 in **both** folds, with a
   clear same-data control improvement; no selected evaluation threshold.
5. Only then consider existing C016 acceptance and official C023 22/97 replay.
   A future graph stage can reparent existing daughter nodes without adding
   detections, but that alone does not ensure edge/lineage safety. Require
   measured signed edge/fork changes, unchanged off controls, then portable/T4
   verification. This audit supplies no candidate to submit.

Main risks: only two embryos and very few 44b6 positives; production paths can
already be wrong or unavailable; the fixed voxel cube is physically anisotropic
and overlaps nearby nuclei; boundary truncation; severe class imbalance and
98.2%/98.1% unknown triples in the measured movies. Frozen public detectors/head
saw both embryos. Even a component cross-embryo gain is not an independent
whole-pipeline validation or a leaderboard forecast.
