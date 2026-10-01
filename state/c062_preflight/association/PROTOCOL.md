# Fixed two-movie trajectory-appearance information probe

2026-09-29. Implemented but not launched by this subagent. No training,
background queue, new model, graph edit, Kaggle operation or ETA is registered
by this document. Parent owns review, controls, input hashing and any launch.

## Question and distinctness

Does averaging frozen C048 appearance evidence along actual predicted
trajectories improve candidate identity ranking beyond the **same encoder's
single source/child snapshots**, on exactly the same full candidate groups?
The only changed signal is a three-observation normalized mean embedding.
There is no retraining, threshold/weight/window search, new assignment solver,
or C059 division-morphology work.

C035/C038/C041/C046/C048 used snapshot appearance and restricted reciprocal or
unused-target edits. This probe retains their encoder/normalization and fixed
cosine gain0.10, margin0.05, cost-increase1.0 proposal checks, and adds temporal
identity evidence. C032 changed temporal detector inputs; C057 changed parent
supervision on pairwise Transformer packets. Neither tested this fixed
trajectory-prototype aggregation. This is related to the closed appearance
family, and its added evidence may still fail because neighbouring cells look
similar or the predicted trajectory already has a wrong identity.

It is **not a direct re-evaluation of C057's50 selected parent errors or36
daughter links**. Those are a different pre-ILP packet domain, while the
requested cheap infrastructure captures C023's motion-relink source/child
groups and protects final forks. GT-selected C057 examples must not become
a runtime candidate whitelist. The signal can be assessed independently,
but should not be advertised as targeting those50 errors.

## Inputs and exact implementation

Standalone source: `src/c062_tracklet_probe.py`.
Fixed movies: `44b6_12dfb391` and `6bba_05db0fb1`.

- `c038_complementary_stage.AllNodeRecorder` and the existing
  `reid_probe_local.install_recorder` observe actual C023 relink candidates.
  `c058_localizer_baseline.capture` replays the existing cached C023 graph,
  verifies its existing C055 controls, and captures actual synthetic origin.
  New capture must also match immutable C058 IDs, integer/float positions,
  edges and synthetic flags exactly.
- `experiments/candidates/c048_embryo_holdout_appearance/models/6bba_weak_aug.pt`
  is used for44b6; `44b6_weak_aug.pt` for6bba. Model checksum, recipe,
  source-only training stems and original C048 plan hash are verified.
  Existing `fold_proof.json` and model receipts supply the completed provenance.
- `c038_complementary_stage.appearance_vectors` reads native8x32x32 crops at
  original recorder positions through unchanged C035 patch normalization and
  frozen32-dimensional encoder. A small reader wrapper delegates actual frame
  decompression to the existing `frame_motion_audit.read_frame`. Encoding is
  CPU FP32 with four threads, so this stage need not occupy the localization
  training GPU.
- Existing C048 artifacts contain learned checkpoints, GT-centred training
  crops and graph/replay outputs. They do **not** supply a complete saved
  embedding table for all runtime nodes in these three-frame contexts. The
  original C035 evaluation crops cover selected labelled groups, so reusing
  them alone would introduce a GT whitelist. New bounded encoding is necessary.

## Prediction-only selection

First reproduce C038's source/current-target/candidate existence, consecutive
time and fork/gap protections on the unchanged final graph. For every retained
source use its actual history `[t,t-1,t-2]`. For every original target candidate
use its actual future `[t+1,t+2,t+3]`. Every context must be unique/consecutive,
unbranched, non-synthetic, have an original-image recorder location, and admit
the full native8x32x32 crop. Actual C058 gap2 insertion provenance is included.
The crop center is computed through the original float32 conversion and
Python rounding used by C035 patches. It must equal the final original C058
integer center; same-ID/time coordinate replacements are excluded and counted.

If any candidate lacks its complete context, reject the **whole group** and
report that reason; do not remove only that candidate. Both scoring formulas
therefore see identical candidates, original current choice and geometry.
All remaining prediction-only runtime groups are evaluated, including unknown
sources and targets. No GT-continuity mask selects a group.

The prediction-only groups are persisted before reading label identities.
Original C058 official prediction-to-GT pairs and source GT edges then annotate
candidate rows: expected known successor1, distinct known target0, unknown-1.
GT division/unknown successor groups retain unknown labels. Unknown targets
are never biological negatives and never enter a fit (there is no fit).

## Scores, controls and interpretation

Compare `dot(e_source,e_target)` with
`dot(normalize(mean(source_history)), normalize(mean(target_future)))`.
All vectors come from the same frozen opposite-embryo checkpoint, at its
original inference centers. Exact repeated-vector triplets must reproduce
single-frame cosine within2e-6 and every fixed C048 per-group proposal exactly.
An undefined mean aborts; it never silently removes a hard group.

Report full eligibility/exclusion counts, unique encoded nodes, actual frame
reads, unknown candidate and selected-choice burden, and all candidate scores.
For labelled diagnostic groups with a reachable positive and at least one
known competitor, compare known-only rank rescues and harms. Separately
compare actual all-candidate top choices: only known0->known1 is a confirmed
rescue, known1->known0 a confirmed harm; all changed choices touching unknowns
remain explicitly unknown. Ranking among known candidates is retrospective
information evidence, not a deployable exclusion of unknown cells.

The fixed C048 per-group proposal rules are also evaluated as diagnostics.
No reciprocal acceptance, target stealing, assignment or graph mutation is
performed. A proposal count is not an actual recovered edge count. A positive
signal on both movies only earns review of a separately registered broader
integration; it does not establish official score improvement.

The decision gate uses the **actual unchanged per-group admission decisions**,
not the known-only ranking table. For each embryo it requires at least one
confirmed known0-to-known1 rescue versus single-frame appearance and strictly
more rescues than harms, also counting unique `(stem,GT source,GT child)` links.
Choosing a known-distinct target must not increase versus either single-frame
appearance or original baseline geometry. No actual known single-frame mistakes
means inconclusive, not passed. Explicit tables also report rescues/harms versus
baseline, examples rescued versus both controls, and all changes involving
unknown choices. Unknown-to-known transitions never supply confirmed rescues.
`admission_choices.csv` retains every original/fixed-rule choice and label;
`review.json` includes unique GT link identities to prevent counting candidate
rows as independent recoveries.

## CLI and expected cost

Use the existing global Python, with no hidden launch until the parent pins
the finite inputs and attaches the existing Windows watcher.

1. `python src/c062_tracklet_probe.py controls --out <new-output>`
   runs algebra/proposal/topology tests only; no model inference.
2. `python src/c062_tracklet_probe.py inputs --out <new-output>`
   lists source, checkpoints, original graph/label inputs and both movies'
   image/GT files for parent-owned hashing. It does not hash or execute them.
3. For each fixed stem, run `capture --stem <stem> --out <new-output>` then
   the optional `benchmark --stem <stem> --out <new-output>`. The benchmark
   uses at most64 captured nodes on each of three evenly spaced real frames,
   the same CPU encoder/path, and saves `<stem>/benchmark.json` with
   `status=passed` and `estimated_full_encoding_seconds`. It fits nothing and
   saves no predictions. Estimate is measured seconds times the larger of
   node/frame expansion ratios, with20percent allowance.
4. After parent review, run `encode --stem <stem> --out <new-output>` for both.
5. `analyse --out <new-output>` writes `candidate_scores.csv`,
   `admission_choices.csv` and `review.json`.

The parent driver registers a five-job preflight (controls, two captures, two
benchmarks), followed only after review by two CPU encodings and analysis.
Each capture records seconds and coverage; each encoding
records seconds, unique requested nodes and frame reads. The existing C048
six-movie44b6 heldout replay log reports3.5minutes, but this includes a different
number of candidate snapshots and is **not an ETA for this probe**. A few
minutes is a scheduling hypothesis, not a measured promise. Full C023 capture
still uses the existing DeepCenter acceptance gate, so coordinate-convention
uncertainty for a new peak-displacement use is neither exercised nor resolved.

Outputs are new files beneath `<new-output>` only. C048/C058/C057/C052 sources,
weights, graphs and records remain immutable. There is no continuous agent
wait; any background launch gets the repository's normal notifier/heartbeat.
