# Priority correction2026-09-29

Latest11:06KST: preflight passed; C060 FULL PILOT now actually runningPID36444,
10jobs/3824pinnedinputs. Follow state/c060_continuation.md instead of historical
preflight/design notes below. ETA11:45:38,first11:35:38then20min;native Windows
watcher59788 attached. Do not edit pinned sources or start a duplicate queue.

Latest11:00KST: C060 actual two-movie zero benchmark launched hiddenPID10876,
preflight_run/status/launch under experiments/candidates/c060_spatial_track_localizer.
ETA11:10:19; first11:00:19 then20min SAMEbiohub-t4 ACTIVE/current task verified.
Native Windows watcherPID6288 attached; success/failure receipt will be
state/background_notifications/c060_preflight_completion.json. No registered
production fit yet; disposable30step timing model is not saved or selected.
Model/data algebra and all22 source lineage/triple support passed. Strict
source counts1021/6255, tails58/371; runtime254579 supported triples. Exact
Voronoi guard avoids radialdNN/2 unnecessarily rejecting339/640 tail targets.
Review benchmark, independent model/data proofs and driver; run actual original
writer/official preflight, then pin and launch fixed1200step opposite-embryo
pilot only if valid. No source edits while benchmark is executing. Sources:
src/c060_track_localizer_study.py, c060_track_localizer_data.py,
c060_spatial_localizer_model.py. CandidateREADME and proposal.md describe it.

Latest user explicitly prioritizes properly correcting the localization issue
behind the75.6% missed-link correlation. C058 fixed recipe remains closed, but
do not abandon localization just because this recipe failed. C059 temporal
division development is DEFERRED, not launched/trained/registered. Its draft
source is incomplete and cannot be treated as a ready candidate.

Current bounded diagnostics: model/ source-vs-opposite axis/target/generalization
audit; identity/ raw-image/longitudinal matched-cell audit; continuity.py original
GT-pair residual/track-continuity decomposition. All reuse existing official
matching, images and frozen outputs. No new model/queue/ETA yet.

Distinguish actual cell-center errors, annotation positioning and wrong matched
identities. Analyze why C058 harms already-good points before another recipe.
No magnitude/threshold/epoch sweep or selective retention of winning movies.
Lowering coordinate residual with frozen final topology and unchanged GT mapping
cannot itself recover missing edges. A successfully validated localizer may
need pre-association use as a later separately controlled experiment; C058's
paired residual already fails, so this observation does not rescue that arm.

New finite background commands must use tools/notify_background_completion.ps1
for Windows completion/failure notifications and same scheduled review only
after actual launch/estimated cost. Deadline2026-09-30 09:00KST;five submissions
user-reported, fresh sharedquota/dedup required. C023/C024 anchors preserved.
