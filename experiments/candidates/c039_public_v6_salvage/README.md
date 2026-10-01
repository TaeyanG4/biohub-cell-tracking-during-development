# C039: public V6 claim audit and isolated salvage

User explicitly requested investigating the0.965+ claim and testing usable mechanisms even if the claim is unsupported (2026-09-27), then requested identifying and repairing the execution error. This study is prepared after C038, using the existing inference/replay/queue. It must not contend with the running C038 followup_extension queue.

## What the public evidence actually shows

- Public notebook: amanatar/optimized-biohub-max-score, V6/scriptVersion353147685. Latest submission56594457 has no exposed score at inspection. Best public0.953 isV4/scriptVersion352554580. An earlierV4 source pull returned403. No evidence establishes0.965+; that string is a configuration label printed by the notebook.
- Downloaded only selected actualT4 output: submission.csv, run_stats.csv, ppsweep_selected.json, retention report, patched predictor and run log (about13MB, no weight archive). Source of truth: `state/notebook_radar/pulled/review_20260927_user_screenshot/v6_actual_output/`.
- All4 visible movies hit `NameError: SAFE_DIV_HORIZON_FRAMES is not defined`. The submission writer catches exceptions and saves basic-filtered originalILP graphs. `repair_fallback=1` on all4 explains why the kernel still reports COMPLETE. It did not successfully execute the intended full post-processing pipeline.
- Existing pinned official evaluator scores that actualCSV **0.9250646856464496**, TP2040/FP80/FN87 and divisionTP0/FP0/FN3. This is visible4, not the hidden LB. C023's matched visible4 is0.9333514696, roughly0.008287 higher. The broken full result does not identify which individual new mechanism helps or hurts.

## Error and repair

The author set `BIOHUB_SAFE_DIV_HORIZON_FRAMES` and `BIOHUB_SAFE_DIV_HORIZON_DIVERGE_UM` in os.environ, and read them for CONFIG_DISPLAY, but never assigned the uppercase globals used by the division function. A dictionary value or environment variable does not create a Python global. Original source/logs are preserved unchanged.

The C039 division notebook binds both globals from the environment with the published defaults3/2.8 and rejects invalid values. It embeds the published function with only a function-name change for dispatch. `division_branch_regression.json` records direct CPU execution of the actual source: original fails at the first name; fixing only that one fails at the second name; both bindings permit the intended near-sister lookahead branch. A missing-successor case rejects, and the published>=4um shortcut accepts without successors. The latter is algorithmic behavior, not a runtime fix; its effect must be scored.

Existing official replay calls filter_output_graph directly, so an exception fails the job instead of silently substituting a lower-quality graph. Any eventual submission must additionally inspect runtime run_stats and reject nonzero repair_fallback. Kernel COMPLETE alone is insufficient. This does not modify C023's fallback implementation or its pinned files.

## Three fixed arms, C023 settings frozen

1. `primary_max`: elementwise max(primary logits, existing aligned weighted blend).
2. `calibrated_max`: published extra mean/std rescaling of the already-aligned secondary map, then weighted blend and elementwise max with primary. CPU formula parity against the predictor downloaded from actualT4 output is exact on three controlled distributions. This is mean/std calibration, not quantile estimation. C023 already has a clipped mean/std alignment; the extra rescaling can effectively remove its scale-ratio clamp. These details distinguish it from C032 temporal-window averaging.
3. `public_division`: the published multi-step division function with only the missing bindings fixed, under existing C023 distance gates/caps and other settings. This isolates the function rather than bundling V6's lower detector threshold, altered retention, gap5.8, motion-z weight, relaxed geometry/caps and short-track rescue.

Elementwise max preserves/increases voxel logits, **not necessarily local-max nucleus detections**: a raised neighboring peak can suppress the original local maximum. Neither detector arm is presumed safe. All receive actual full22-movie official evaluation; no arbitrary threshold sweep.

## Tools, controls and follow-up

`src/c039_public_salvage.py prepare|branch_check|smoke|run|analyse`. Prepared repo copies C023; detector notebooks embed the same checked text replacement. Original C023/C024 and all C038 pinned inputs remain unchanged. `plan.json` hashes37 sources/models/notebooks/config inputs. Pre-launch UTF-8 correction is archived under preflight_revision1; no inference had run and scientific strings were unchanged. `preflight.json` verifies source/model hashes, formula parity and actual generated global assignments. `division_branch_regression.json` is an execution regression, not official-score evidence.

Launch hidden `python -u src/c039_public_salvage.py run` only once C038 GPU queue is idle. The driver refuses otherwise; do not start a sleep/poll relay. Existing biohub-t4 heartbeat is responsible for sequencing. Ten-hour serial finite queue, source lock/hash. Two-embryo four-frame original/off execution must preserve exact nodes/edges/lowdetections before inference. Local4070 Ti SUPER uses FP32, TF32 off, mathSDPA. Two detector arms reuse existing full inference+ILP and official replay on12+10 movies. Division arm reuses matched C023 caches with22 exact off-controls. No new scorer or training, no unknown-label substitution.

Review signed official22/heldout12/confirm10/embryo and division effects; retain small consistent gains for matched extension and justified fixed combinations. Do not auto-submit this study or push unverified notebook builds. Actual portable implementation replay and T4 visible4 matching, runtime fallback checks and quota/version safeguards are still required. No Kaggle writes/slots so far. Preserve C023/C024 and final picks.

## Final status

Completed16:08KST2026-09-27; all three fixed arms negative in both splits/embryos. See FINAL_REVIEW.md. Closed without Kaggle operations; scheduled follow-up stopped.
