# C032 temporal context — local research, not ready for submission

**Closed 2026-09-27; no submission.** Both finalists reverse on extension75: mean_det -0.001141, future_det -0.001610. Official all97 totals: C023 0.940490013, mean_det 0.940095180 (-0.000394833), future_det 0.939671800 (-0.000818213). Both lose adjusted-edge score in both embryo groups. All queued jobs succeeded. Final review: `state/last_days_local_20260926/FINAL_REVIEW.md`; no Kaggle push/submission, C023/C024 unchanged. Pilot and launch notes below are historical.

Base: scored C023, x138 s075 head, unchanged post-processing. No Kaggle operations.

Fixed arms (each has its own notebook and kernel-metadata.json):

- `mean_det`: mean detection logits from `(t-1,t)` and `(t,t+1)`; original first-seen head features.
- `future_det`: next-window detection map for interior frames; original head features.
- `mean_det_head`: mean detection logits and mean primary head features from both windows; edge features stay pair-specific.

Every window is encoded once. At most the current/next packet is retained; actual peak VRAM/runtime is to be measured on full inference. First and last frames use the available single context. Node order and existing edge inference loop are preserved.

`src/verify_temporal_context_local.py` completed a real four-frame/two-embryo smoke check. Original C023, patch-off and delayed past-only controls have exactly equal refined coordinates, edge arrays/probabilities and lowdet arrays. All three active modes executed with finite outputs and consecutive-time edges. This does not measure official score or hidden generalization.

The finite background queue is `src/run_last_days_local.py`. Results and job state:

- `state/last_days_local_20260926/status.json`
- `state/last_days_local_20260926/RESULTS.md`
- `state/last_days_local_20260926/logs/`
- `state/last_days_local_20260926/replay/`
- Detailed inference logs: `e2e/<arm>/predict.log`.

Numerics: global Python + RTX 4070 Ti SUPER, `run_kaggle_predict_local.py --t4-fp32` (TF32 off, math SDPA, cuDNN benchmark off). Control and arms use identical settings. Existing T4 C023 caches are compared diagnostically, but T4 runtime/replay acceptance for a new notebook is still pending and belongs to the user's Kaggle run.

Queue: matched C023 control + three fixed arms on heldout-12 and confirm-10. Only arms improving adjusted edge on both sets, total by >=0.001 on heldout and >0 on confirm, advance to 75 additional movies. This gate allocates compute; it does not establish private-LB value. No settings sweep, upload, push or submission is performed. Do not edit running hashed scripts or start a duplicate queue.

When execution finishes, inspect failures/status first, then compare total and edge deltas, both embryo groups, per-movie regressions, node counts and divisions. Keep C023/C024 final picks until stronger evidence exists. Candidate notebooks are build artifacts, not approved submissions.

Pilot reviewed 2026-09-26: all inference/replay/smoke jobs succeeded. Total deltas on heldout12 / confirm10: mean_det +0.000674 / +0.002948; future_det +0.001303 / +0.001138; mean_det_head +0.002310 / -0.000759. Future_det adds 2 division FPs on confirm10 and has a worst single-movie adjusted-edge delta of -0.061629 on heldout12. Mean_det also has losing movies; these are not hidden-test gain estimates.

The queue stopped before extension because the C016 lists use commas. Driver fixed and original run archived; scientific sources unchanged. Reviewed extension includes automatic future_det plus `--extra-extension-arm mean_det` because both pilot totals and edge deltas are positive. This extra allocation was chosen after review and is recorded separately from the original gate. The 75 extension movies are disjoint from the pilot 22. See `state/last_days_local_20260926/PILOT_REVIEW.md` and live status.json.
