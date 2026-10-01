# Temporal division component: bounded next-step assessment

2026-09-29. Read-only source/artifact assessment; no fit, graph change, candidate
ID, background queue, or submission launched by this assessment. C058's fixed
raw localizer is not a basis for a parameter sweep.

## Decision

The already proposed paired **48 versus 56 input-feature** study is implementable
with existing infrastructure and is a distinct, falsifiable mechanism. The next
authorized work should first implement and benchmark extraction on the same two
predeclared movies, with real original features and exact C023 graph controls.
Only then register a finite 97-movie component study and its measured ETA. The
two-movie feasibility timings are not sufficient to launch a timed full queue.

Mechanism: a true daughter pair may have a characteristic joint evolution of
size, brightness, and size symmetry across predicted tracks. C050 had geometry,
one-frame intensity, and t+2 separation, but no measured nuclear-volume time
series. The data reject a universal shrink rule; the fixed existing learner
tests whether the joint new block transfers between embryos.

## Exact stage and infrastructure

Use **C023's final graph returned by its unchanged `filter_output_graph`, before
the original CSV writer**. This matches the two-movie temporal feasibility
stage, and is an inference-available stage for a later additive fork proposal.
Keep its full node and edge dictionaries, including all actual flags. Do not
reconstruct them from the C055 integer snapshot: those snapshots retain IDs,
coordinates, and edge endpoints, but discard real edge/provenance metadata.
Use final floating coordinates consistently for both feature blocks, and the
original rounded coordinates for matching and graph parity; freeze this choice
before fitting. The earlier feasibility only tested integer-centred probes.

Reuse these APIs; no new competition scorer, learner, or visualization harness:

- `src/c055_guarded_readmit.py`: `BASE`, `GT_DIR`, `split_stems()`, `run_dir(split)`,
  `pred_path(split, stem)` locate all 97 existing C023 FP32 caches and controls.
- `src/eval_pp_variants_local.py`: `build_namespace(BASE, {}, edge_cache)`,
  `install_frame_caches(ns)`, `load_raw_graph(ns, path)` and the notebook's actual
  `filter_output_graph` replay C023. Verify returned rounded IDs/txyz/edges against
  `experiments/candidates/c055_guarded_readmit/graphs/<split>/<stem>_control.npz`.
- `src/c058_localizer_baseline.py`: `capture` is the exact replay/parity pattern
  and demonstrates passive recognition of unflagged `recover_strict_gap2`
  insertions. Reuse that provenance pattern if synthetic eligibility/flags are
  needed; do not infer provenance from recycled numeric node IDs. Its
  `official_match`/`official_score` and `write_frozen_csv` are existing actual
  organizer-metric/writer checks, distinct from the replica metric.
- `src/division_scorer_stage.py`: `load_dump`, `make_intensity_fn`,
  `enumerate_candidates`, `FEATURES`, `make_model`, `score_candidates`, and later
  `add_scored_divisions`. Geometry is fixed at parent14um/sister16um. Supply real
  admitted probabilities, low-detection peaks, `deepcenter_score_point`, and
  intensity values. The feasibility probe's zeros were only geometry testing.
- `src/division_candidates_local.py`: `MetricLabeller` supplies strict/early/grand,
  known-negative, and unknown labels. Its CLI captures an earlier safe-division
  stage, so do not use that CLI unchanged for this final-stage experiment. Feed
  it the actual final-stage mapping (existing actual official matching API).
- `state/research_reset_20260929/temporal/feasibility.py`: reuse its unique,
  consecutive predicted-chain logic and unchanged AST-extracted public `probe`
  (source SHA256 `1dd8e153fdb16061f61d107a1357fca0ab1839eac0e51f00a774ad6465d6d9ab`).
  Centres/paths must never use GT. Deduplicate each `(movie,node)` probe and
  group reads by frame. Missing tracks/forks remain explicit missingness.
- `src/division_scorer_train.py`: existing `fit(x,y,epochs,lr,wd,seed,pos_weight,hidden)`
  supports either feature dimension, plus `predict`, `per_parent`, and
  `average_precision`. Its CSV `load` hardcodes the original FEATURES, so use
  its exact label policy in a thin paired-array adapter; do not monkeypatch the
  pinned learner or silently omit the extra columns.
- `src/c050_division_candidate_audit.py`: finite orchestration/fold/proof pattern;
  `src/run_last_days_local.py:Queue` for any later registered queue.

## One fixed recipe and falsifiable gate

Append exactly the previously proposed eight columns: mean daughter/parent
half-height-volume ratio at t+1 and t+3; corresponding peak-contrast ratios;
daughter-volume asymmetry at t+3; complete-track flag; valid-probe flag; boundary
flag. Reference is the mean of predicted parent's t-3..t-1 measurements. Probe
radius remains5 voxels. Missing ratios=1 and asymmetry=0 with flags. No choice of
lags/radii/weights/thresholds after evaluation.

Paired original48 and augmented56 fits use the same new C023 candidate rows,
strict-only positives, known negatives, and whole-embryo opposite folds. Drop
early/grand positives from supervised fitting, never convert them or unknown
rows into negatives. Preserve all rows for coverage/output diagnostics.
Keep C050's fixed recipe:60 epochs, hidden32, lr.002, wd.001, seed0,
pos_weight=min(negative/positive,100), threshold.9, final checkpoint only.
Train-derived normalization and exact saved-prediction reload are required.

**Go criterion for graph integration, not submission:** in each opposite-embryo
test, augmented56 fixed-.9 known-label per-parent precision and restricted
recall are both >=.30, with precision and recall each >= the paired48 control
and at least one strict improvement in each embryo. Report matched annotation
window/exact-context recovery as well as parent counts: no duplicate candidate
or early/late alternative may be presented as another recovered biological
event. Failure closes this fixed feature block without threshold search.

`per_parent` on known rows is a restricted component diagnostic: its best row
can differ from the best row among all candidates when an unknown wins. Score
all candidate rows too and report those changed winners, accepted unknown
burden, early/grand alternatives, and collisions. Unknown winners are not FP by
assumption. The component gate alone cannot establish production precision.

If the gate passes, use existing acceptance code at a separately fixed stage
recipe and existing official C023 graph evaluation. Actual graph benefit must
include positive total score and true-fork recovery, nonnegative adjusted edge
score in both embryos, unchanged off controls, and signed TP/FP/FN attribution.
No graph integration is currently implemented or justified by observed gains.

## Actual sample size and why C050/C049 do not already answer this

I intersected C052's saved `event_audit.json` with C055's 97 stem lists: the 97
contain all151 annotation occurrences, 26/125 by embryo, across21/66 movies.
Their exact-context groups are25/125, with1/9 lacking bounded context. Those are
at most150 distinct exact contexts, **not150 independent biological events**;
non-identical overlapping crops remain unresolved. The only fully separated
biological validation units are the two embryos. Whole-embryo fitting prevents
their overlap from crossing a fold; do not claim independent-event significance.
Attach known strict candidates to these existing `(stem,GT-parent)` context
groups for coverage and recovered-event reporting; do not merge by reused GT IDs.

C050 historical C012 data had221,133 known rows but only94 strict parent cases
(18/76), and fixed-.9 precision .0588/.0375 and recall .0556/.0789. C049's corrected
GT-centred image classifier also failed transfer/useful operating precision.
Neither established predicted-track volume trajectories as useful features.
The paired same-data original48 control is essential to isolate their benefit;
comparing a fresh C023 model only against historical C012 C050 would confound
the new feature effect with graph/label changes.

## Runtime evidence and scheduling limit

Actual feasibility:244,237/584,874 geometry-qualified candidates,7.81/17.85s for
enumeration, and497/492 sampled unique node probes across94 frames each took
1.42/1.46s including reads (0.789GB uncompressed/movie). Total10.89/23.44s was a
label-stratified sample, not full feature extraction. Full graphs have44,749 and
69,794 nodes. Naively scaling the entire sample probe time by node count gives
roughly128/207s per movie, but overcounts repeated frame-read cost and omits
real original48 DeepCenter/intensity work; it is only a sizing heuristic.

Dominant unknown cost is **real feature extraction**, especially the numerous
candidate-specific DeepCenter midpoint/intensity calls and full node volume
probes, not the learner. Keep deduplicated per-node measurements and the
existing frame/heatmap cache. Original C050 training took6.97/18.56s for its two
folds; four analogous paired fits are small relative to extraction and I/O.
Benchmark two complete movies with all candidate features/labels/output
diagnostics before extrapolating by nodes/candidate counts and registering ETA.
Do not use34s/2movies as an ETA for the97-movie queue.

No unattended work was launched here. A justified new finite queue must record
exact PID/process creation time and ETA, attach the existing native Windows
watcher immediately, and separately anchor the scientific review heartbeat at
ETA-10min then20min. Deadline remains2026-09-30 09:00KST; user-reported remaining
submissions5 does not justify submitting unsupported candidates.
