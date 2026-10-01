# C066 — broaden ordinary-link supervision using preserved production packets

Prepared after the user's validation/time-budget reset. This is the second
priority after C065, not an already launched fit or demonstrated improvement.
All work including submission must finish before2026-09-30 00:00KST.

## Distinct change and available data

C052/C065 ordinary batches use only24movies/2246windows/13387positive target
records. Existing C052 capture/labels on87other movies contain7791 unused
ordinary windows/63328additional positive target records. This yields111
ordinary movies(33/78 by embryo),10037ordinary windows and76715ordinary target
records. The145division-bearing windows and144existing sampling clusters are
unchanged. These overlapping crop/time records are not independent cells.

Use the unchanged C052 learner/sampler/600steps/seed3701/300division and300ordinary
batches/teacherKL0.1/lr1e-5/weightdecay1e-4/divisionweight2/dropoutoff/FP32.
Only expand the ordinary sampling pool with existing zero-division-target
packets. Reuse original known-only labels without unknown negatives; verify
original artifact hashes and label bounds. No new extraction/labels, epoch,
threshold, architecture or sampling-weight sweep.

## Validation before production promotion

Fit two whole-embryo models and a separate pooled production model with this
one recipe. The pooled fit's existence is not permission to deploy it before
the opposite-embryo checks. Reconstruct every600sample trace and enforce strict
source membership. Original source-only teacher/logit checks and unchanged
runtime reload apply to all three final checkpoints.

Evaluate the two opposite-embryo models on the fixed97, with unchanged C023
inference/postprocessing and exact frozen-detector controls. Use the actual
organizer metric and compare C023 and C052 on same97/22/75/each embryo/movies.
Pooled production results are a separate fit-domain check, not validation.
Do not select epochs/thresholds or route a hidden deployment by embryo name.

The expanded recipe must have worthwhile additional actual graph evidence over
the C052 recipe; more training records alone are not evidence. Report total,
edge/division TP/FP/FN, node cost and both-domain tradeoffs. Small changes in
surrogate subgroups are not automatic vetoes. If unhelpful, stop this fixed
recipe without a data-weight/epoch sweep.

If worthwhile, the already fitted pooled checkpoint gets its own full97
portable execution and actual T4 checks before any authorized submission.
Use C065 corrected production machinery with truthful C066 provenance, not the
obsolete mean-model-only log assertion. Preserve actual original writer parity.

## Time and scheduling

Only one urgent local GPU queue. Prepare/hash data whileC065runs; no C066 GPU
fit until C065's local graph/portable phase has completed. Fit has an18:00KST
latest start guard; still recalculate the entire path before launch. Estimated
6minutes fitting/controls,130minutes opposite97,145minutes production97,
at least60minutes upload/T4/submission, plus margin. C065 remote T4 can overlap
the later local C066 work. If actual time makes midnight infeasible, do not
start a partial validation that cannot yield a properly checked submission.

Five slots user-reported, fresh sharedquota/dedup before submit. C023/C0240.954
anchors preserved; prior submissions are not polled or repeated. Each new
hidden phase needs real PID/start/ETA, existing native Windows notifier and
the same biohub-t4 heartbeat. No Kaggle writes in data/fit phases.
