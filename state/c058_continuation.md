# C058 continuation — 2026-09-29

Latest user correction10:48KST: localization remains first priority. Follow
state/localization_diagnosis_20260929/CONTINUATION.md. Temporal C059 is deferred,
unregistered and untested; historical next-temporal wording below is superseded.

**CLOSED after independent review10:36KST.** Actual complete10:31:40KST,
all10jobs/3744input/2693output hashes passed. all22official delta−0.006187127;
44b6−0.026944712,6bba−0.003931318;edgeTP−47/FP+64/FN+47,division unchanged.
633 eligible>3.5um residuals worsen4.64225→4.71107um. No97/T4/submission or
fixed-arm retry/sweep. FINAL_REVIEW.md/state/c058_review_20260929 are authoritative.
Native Windows completion notification delivered10:32:32; completed heartbeat
deleted. Next distinct C059 temporal-feature preparation is not training yet.

User notification update: native Windows popup+sound on background command
completion/failure, not app turn notifications. Existing C058 now has a separate
hidden watcher using tools/notify_background_completion.ps1; see
state/background_notifications/c058_watcher_launch.json and completion receipt.
Attach this same helper to each newly launched finite background queue, using
its exact PID/start launch.json and unique receipt. Do not edit pinned sources.

**ACTUALLY LAUNCHED 10:20:35 KST, owned PID13968,3744 pinned inputs/10 jobs.**
Measured ETA10:47:35KST; first10:37:35then20min. Same biohub-t4 ACTIVE,
target current task01a0ea5b-927e-73d2-94ef-c9aed10ec3ed verified. Original
create rejected anchored DTSTART; safe paused create followed by tool update
accepted exact anchored schedule. No approval failure. Source hashes passed
before first capture_all job. The fixed models themselves train after capture,
extraction and label audit jobs; do not infer training completion from launch.

User: continue grounded score improvements without fixation on +0.001–0.002.
Latest deadline: **2026-09-30 09:00 KST**, five submissions remain per user.
This is not a fresh shared-quota readback; refresh quota/dedup before submission.

C046 already submitted56654681. C057 and other fixed closed arms remain closed.
C023/C0240.954 final picks preserved; no submitted-candidate score polling.

New candidate C058 tests full-resolution raw-image localization with final C023
IDs/counts/times/edges frozen. Sources: src/c058_raw_localizer_study.py,
src/c058_raw_localizer_model.py, src/c058_localizer_baseline.py. Read candidate
README/plan/benchmark/status/launch under experiments/candidates/c058_raw_localizer.

Two full-movie zero controls passed114,543 total nodes/61,312 eligible. Raw crop
labels:549/815 examples,43/45 tail>3.5um; actual official versus replica matches
equal on these two. 30-step disposable timing model was not saved or selected.
Actual original C023 writer + organizer CSV evaluator roundtrip passed one movie.
Full22 zero and learned writer/official tests remain part of the finite queue.

Preflight fixes: notebook replica requires tuple edges. More importantly,
recover_strict_gap2 interpolates nodes WITHOUT gap_synthetic; passive function
observation now captures actual newly inserted object identities. Combined
synthetic mask excludes those and explicitly flagged synthetic nodes. No graph
is edited during provenance capture. C055 final integer graphs remain exact.

Recipe:407,659-parameter spatial CNN,16x64x64 normalized full-resolution crop,
known actual-official7um matches, no unmatched negatives,18x72x72 real-support
source for exact z±1/yx±4 jitter. Whole-embryo opposite fits, fixed1200steps each,
uniform source movie/known eligible point, final checkpoint only. Inference all
non-synthetic supported nodes; no GT filtering. All22 are component validation;
frozen public detector/head saw both embryos. Large paired residuals can still
reflect wrong biological identity; report ambiguity and damage, not just MSE.

Finite10 jobs: capture_all,extract_all,audit_labels,train44b6,train6bba,infer44b6,
infer6bba,evaluate,writer_verify,analyse. Source/input hashes at start/end and
output_hashes at completion. Existing Queue only; no Kaggle writes or automatic
97 extension. Windows process checks: Get-Process/Get-CimInstance, never kill0.

On completion independently verify all inputs/outputs and10 jobs; inspect
label_audit, fold/stems/1200step receipts, official_rows, paired_residuals,
match transitions, writer_verification, analysis, decision. Judge actual
vendored official scores, not notebook replica. Both opposite embryos must
improve adjusted edge, edge TP, fixed-GT >3.5um residual, with>=2 movie wins,
and all22 total positive. If passed, review unchanged97 expansion then fixed
deployment/portable/actualT4/source/model/CSV/version/fallback0/degradation0.
Existing worthwhile submission authorization applies; no renewed permission.
If this recipe fails, close it and assess the already documented distinct
temporal-division hypothesis before any new queue; do not sweep this recipe.

Monitor actual PID/start/ETA in launch.json/state/biohub_t4_monitor.json.
Each new phase schedules same biohub-t4 heartbeat firstETA−10,then20min. Bind
to current task, never old failed target. Quiet while unchanged; notify only
completion,failure,meaningful result/submission or required user action.
No continuous active waits. Stop/delete monitor if no justified work remains.
