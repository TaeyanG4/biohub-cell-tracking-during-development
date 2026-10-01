# Experiment area (not a candidate): ILP division weight test

**Status**: Closed 2026-09-24. HANDOFF section 21.

- Change: Local inference with `BIOHUB_ILP_DIVISION_WEIGHT` 0.7 / 0.5 (`e2e/head_v1_ilpdiv0.7|0.5`).
- Build / verify: `python src/run_kaggle_predict_local.py ... --env BIOHUB_ILP_DIVISION_WEIGHT=0.7`
- Evidence: 0.7 makes 578 forks and 0.5 1,838 forks against ~15 true divisions; the motion relink drops them again, and re-injecting them loses at every threshold.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
