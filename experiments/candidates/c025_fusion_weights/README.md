# Experiment area (not a candidate): inference-time knob runs

**Status**: Closed 2026-09-25. HANDOFF section 22.4 items (d)-(e).

- Change: Local runs (`e2e/<arm>`): secondary detection weight 0.70 / 0.90, secondary edge weight 0.25 / 0.10, ILP disappearance 1.5 / 3.0 / 4.0, and the combination 3.0 + 0.10, on the held-out 12 and the confirm 10.
- Build / verify: `python src/run_kaggle_predict_local.py --repo tmp/c024_tta_pool/tracking_repo ... --env <KEY=VALUE>` (pooling off = C012 inference)
- Evidence: x138's values are optimal: detection 0.80, edge 0.15; disappearance 3.0 only moves the node count (helps over-predicted movies, hurts the others: confirm-10 -0.0065).

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
