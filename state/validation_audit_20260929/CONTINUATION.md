# Validation methodology audit — 2026-09-29

## Latest state: primary audit COMPLETE; legacy backfill follows

The main audit completed 14:30:28 KST, native toast accepted 14:30:31.
`independent_verification.json`: 2,498 inputs/394 outputs, 2,893 unique hashed
paths, 388 graph rows and 20 summaries independently passed. See Korean
`REVIEW.md`. Actual97 deltas C052 +0.001807801748, C053 +0.001108508379,
C054 +0.001272997272 versus C023; ranking unchanged. C052 actual division
TP +1/FP -2, correcting the replica's old net-TP-zero claim. Upstream organizer
GitHub main matches the pinned metric commit. 22-subset composition is unstable:
C054 negative in 15.9% of 2,000 same-6/16 subsets. NOT a confidence interval.

The finite follow-up is `legacy/audit.py` (pinned): 2,821 inputs, 313 saved
graphs: C037_late600 parent control97, C055_readmit_off97, C055_v12_guarded97,
C05722. C037 graphs are the C040 appearance-off snapshots with its unchanged
late600 Transformer, checked against original manifests. No refit, inference,
graph change or closed-arm tuning. Main actual rows are reused as comparators.
Read `legacy/run/{status,launch}.json`. Initial ETA8minutes is based on the
actual main249.49seconds/364calls, scaled313calls ×1.5 plus2minutes.
Native watcher `state/background_notifications/validation_audit_20260929_legacy_*`.
Read this new phase rather than relaunching the completed main audit.

On legacy completion independently verify plan and output hashes,313unique
rows/graph bindings, aggregate against mainactual rows using organizer summary,
and actual group deltas. In particular compare C052 with C037_late600 to isolate
the supervision-data change; compare C055_v12 with readmit_off and both embryos;
compare C057 with C052 on22/splits/embryos. Correct prior interpretations if
needed. Do not assume changed scorer reverses any negative result. A supported
new finite improvement may follow; otherwise record disposition/delete heartbeat.
No unregistered pooled-model refit has been launched or selected.

User asks whether current validation, especially 22 movies, is adequate and why
gains of 0.002–0.003 are difficult. Continue this new evidence-driven audit;
closed component experiments remain closed. Read AGENTS.md and HANDOFF.md first.

## Completed primary finite work (preserve, do not rerun)

`audit.py` and `launch.ps1` are pinned by `plan.json`; do not edit either.
2,498 fixed input files; 388 saved integer graphs (C023/C052/C053/C054 × 97).
Historical graph manifests verified. The 22 C023 actual organizer rows are
reused only after original ids/txyz/edges equality and receipt hashes pass.
Two C053 benchmark results are independently pinned; 364 remaining fresh calls.
No training, inference, graph changes, T4 or Kaggle writes.

Hidden owner PID 33248, creation 2026-09-29 14:26:00.1156198 KST.
Initial ETA 14:56 KST from the actual two-movie benchmark with 1.5× allowance.
Native Windows completion/failure watcher PID 55192 attached immediately;
observer files are outside this audited tree, under state/background_notifications.
Same biohub-t4 heartbeat recreated and anchored at 14:46 KST, then 20 minutes.
`run/status.json`, `run/launch.json`, `run/logs/rescore.log` are authoritative.

On completion verify all plan inputs and run/output_hashes.json independently;
recount 388 unique candidate/movie rows, graph hashes, full integer graph cases,
benchmark hashes and exact reused receipts. Recompute summaries via the actual
organizer aggregator and compare candidate deltas/rankings on 22, 75, 97 and
both embryos. Inspect metric_mismatches.json; an absolute metric discrepancy
does not prove that relative candidate rankings reverse. Record aggregate
weights and small-subset sensitivity as descriptive diagnostics, not statistical
confidence based on 97 independent biological specimens.

## Confirmed methodology findings

- All 199 movies belong to TWO embryos, with proven overlapping crops.
  Whole-embryo exclusion is needed for the newly learned component; the public
  frozen backbone already saw both embryos. Increasing crop count does not
  create independent biological validation. Do not call 97 or 199 untouched.
- 22 movies have 17,144 annotated records: 1,601 in six 44b6 movies, 15,543 in
  sixteen 6bba movies. 97 has 73,017 records across 27/70 movies. Records are
  correlated, not unique independent cells. 22 is a smoke/gross-harm screen,
  not a final selection/generalization test.
- The historical notebook replica and actual organizer code give DIFFERENT
  scores on the SAME 22 C023 integer graphs: 0.9459861726138064 versus
  0.9431126093954155. Division TP/FP/FN 5/6/19 versus 4/5/20. This is not
  coordinate rounding; both receive identical arrays. Earlier files named
  official_* in C052/C053/C054 contain replica results. Preserve them and add
  corrected interpretations rather than overwriting historical records.
- Official score is adjusted edge Jaccard + 0.1×micro division Jaccard, not
  accuracy. 0.954 is not 95.4% correct or a demonstrated ceiling.
- Requiring every residual stratum/axis to improve with good-point tolerance
  1e-12 is an engineering screen, not a necessary condition for official gain.
  New methods should be judged by actual graph score and reported subgroup
  tradeoffs; do not retroactively tune closed recipes to pass revised gates.
- C063 had only 1,159/7,954 eligible source labels from 6/16 movies, and harmed
  source-fit means too. Its failure is not proof all localization is futile,
  nor is it attributable only to small held-out data. C063/C064 gross observed
  harms remain measured facts; no official score gain was tested for those.
- 75.6% means 1,047 of 1,385 both-present missed edges touched >3.5um residual;
  it is neither all-node error rate nor a causal/recoverable fraction.

## Next disposition

Finish REVIEW.md with confirmed actual ranking after queue completion. Keep
C023/C024 public0.954 anchors; C0540.953 is user-reported; no previous-submission
score polling, C04656654681 resubmission or quota use. Any further improvement
must be distinct, supported by the corrected metric evidence and HANDOFF.
Do not launch an unchanged closed arm or silently reopen deferred C059. If no
justified finite work remains, record why, notify the user and delete heartbeat.
Deadline 2026-09-30 09:00 KST. Every new hidden stage still needs actual PID/start/
ETA, the existing native Windows notifier and a correctly anchored heartbeat.
