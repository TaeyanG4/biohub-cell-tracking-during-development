# C061 direct axial localization continuation

## CLOSED after independent review, 2026-09-29

All eight jobs completed at12:17:08KST; native Windows notification accepted
at12:17:13. All9,357input and937output hashes independently match, with no
observer exceptions. Both final1200-step fits and balanced4800/class counts
pass. Both scientific gates FAIL: opposite-embryo real9294pair mean3D error
1.781070→2.781440um and absolute z1.114106→2.284686um;20/22movie means worsen.
Both embryos' overall/tail/good means worsen;7637excluded pairs remain unchanged.
Synthetic large-shift recovery does not overcome zero/±1plane transfer failure.
No official graph score was computed, and no submission was made. Close the
fixed recipe: no97/upstream/portable/T4 or threshold/epoch/guard rescue.

Authoritative review: `experiments/candidates/c061_direct_axial/FINAL_REVIEW.md`
and `state/c061_review_20260929/{verification.json,science/REVIEW.md}`.
No active local queue or pending new experiment. Completedbiohub-t4 deleted.
Public-source review: `state/c061_notebook_review_20260929/source_review.md`;
x138 coordinate head is already inC023, xiaoleilian provides actual direct-GEFF
heatmap training, StrongUNet/DeepCenter offer related spatial representations.
No exact C061 combined protocol identified in bounded reviewed sources; no
new performance gain established. These references do not register a new fit.

## Historical launch/protocol (superseded by closure above)

2026-09-29: user explicitly requests continued work. Deadline2026-09-30 09:00KST.
**ACTUALLY LAUNCHED12:10:24KST, hiddenPID30488;8jobs/9,357pinned inputs.**
ETA12:22:24KST,first review12:12:24then20min. Samebiohub-t4 ACTIVE, explicit
DTSTART anchored/current task01a0ea5b-927e-73d2-94ef-c9aed10ec3ed; tool update
and TOML readback verified. NativeWindowswatcherPID56792 attached; logs and
future c061_completion.json receipt under state/background_notifications.
Both processes alive and stderr empty at initial verification. No duplicate
launch or edits to registered inputs. Queue first verifies all9,357input hashes.
Read candidate plan/status/launch for actual progress; do not equate ETA with completion.

Candidate: `experiments/candidates/c061_direct_axial`.
Driver: `src/c061_direct_axial_study.py`.
Data/model: `src/c061_axial_data.py`, `src/c061_axial_model.py`.

Direct GEFF known-point targets replace prediction-to-GT residual supervision.
All133,318 original GT points have integral t/z/y/x coordinates. Hash-first up
to32 supported points per each199 movies yields6,364 points:2,268 in44b6 and
4,096 in6bba,57,276 possible nine-shift views. Only two biological embryos;
overlapping movie records are correlated. Every source-only batch has four of
each z displacement -4..4 (1.625um/plane), crop13x49x49 from real21x49x49.
One broad spatial CNN315,889parameters, two separate1200-step fits, final
checkpoint only, FP32, seed6101. No unknown cells become negatives.

Evaluate synthetic shift recovery in both own-source and opposite directions.
Own-source is fit evidence only. Exact annotated x/y makes this an easy
diagnostic, not proof of correcting actual errors. Actual C023 evaluation uses
9,294 original eligible known pairs including767 tails>3.5um;7,637 ineligible
pairs are separately reported unchanged, total16,931original correspondences.
Targets never rematch and no graph is modified. All-node inference and actual
official scoring are separate future work only if this component is useful.

Read README for the fixed gates. Every nonzero synthetic shift must improve in
both opposite directions; zero mean error <=1.625um. Actual eligible overall,
tail and axial-tail must improve in both absolute z and3D error, and already-good
points must not worsen in either mean. No threshold/epoch/strength/decoder sweep.
Ownership/boundary violations are diagnostic reports, not a post-hoc correction
filter. Do not infer an official gain from these component measurements.

Preflight: model signs/ties/loss/gradients; whole source sampling and realized
class counts; exact GEFF/source crop reconstruction; actual two-movie pixel
extraction; zero output; exact saved-model reload; original44C058 hash controls;
all9,294continuity joins; actual two-movie analysis zero-output controls.
Review found a continuity column mismatch (`p`, not `node_id`), now explicitly
adapted with uniqueness assertion and executed join/analysis proof. Evidence
under `state/c061_preflight`; preflight.json pins the included receipts/scripts.
No full graph replay is claimed or needed for this no-graph component phase.

8finite jobs: extract+data audit; train44b6;train6bba; synthetic44b6;real44b6;
synthetic6bba;real6bba;analyse. Input hash check before and after, output hashes
at completion, no Kaggle operations. Never edit registered sources or start a
duplicate run. Monitor Windows process with Get-Process/CIM, never os.kill(pid,0).

Every hidden queue gets tools/notify_background_completion.ps1 immediately.
Watcher logs/receipt live under state/background_notifications, outside the
candidate tree. Scientific review uses SAME biohub-t4, firstETA-10 then20min;
no active agent polling. Delete heartbeat when no justified pending work remains.

C054 public0.953 is user-reported this turn; C046 still pending per user and
must not be score-polled. C023/C0240.954 anchors and C04656654681 preserved.
A colleague may submit separately; no action needed, just use fresh shared quota
and dedup at any actual authorized future submission. Five slots was an earlier
user report, not a fresh quota check. Existing autonomous worthwhile verified
push/T4/submission permission remains; do not ask again.
