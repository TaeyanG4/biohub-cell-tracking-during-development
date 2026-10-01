# C033 fixed structured assignment — local research, not ready for submission

C023 followed by the public indarkarhana structured-trajectory assignment, using its published 18 fixed weights. No local fitting, new detector, coordinate shift in output, or node-count changes.

- `raw`: public features in original coordinates.
- `stabilized`: both initial/final feature graphs have cumulative high-confidence ILP median displacement removed; original coordinates are returned after assignment.

Original source/model/NOTICE/provenance: `artifacts/public_structured_trajectory/`. Adapter: `src/structured_trajectory_stage.py`; builder: `src/build_structured_candidate.py`. Attribution and source/model are embedded in each notebook. Existing scorer integration: `eval_pp_variants_local.py --structured-trajectory artifacts/public_structured_trajectory`.

The assignment preserves per-node degrees and division/gap incident edges. This does not guarantee unchanged division metric, because longer branches can change. Replay measures the actual metric.

Execution is queued in `state/last_days_local_20260926/status.json`. Two-movie actual replay and notebook/harness equality are checked before long experiments; syntax/build checks alone are not validation. The same-policy fresh C023 controls are then used for 12+10 movies, with conditional 75-movie extension. Results: `state/last_days_local_20260926/RESULTS.md`.

No upload, kernel push or submission. Final picks remain C023/C024. A failure in this stage is recorded without blocking the independent temporal-context experiment arms.

Completed pilot (2026-09-26): both raw and stabilized reduce total/adjusted-edge score by -0.001057 on heldout12 and -0.000313 on confirm10. Division counts unchanged. Actual embedded-notebook/harness comparisons passed on existing caches and fresh heldout controls. Neither mode advances to the 75-movie extension; this fixed public-assignment direction is closed for this batch. See `state/last_days_local_20260926/PILOT_REVIEW.md`.
