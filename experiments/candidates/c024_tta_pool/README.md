# Experiment area (not a candidate): probability-level edge TTA pooling

**Status**: Closed 2026-09-25. HANDOFF section 22.4 (idea 9).

- Change: Local runs with the patched repo from `src/build_tta_pool_repo.py`: `e2e/off_check` (exact C012 reproduction), `js8`, `js8_incmean`, `js4_identity`.
- Build / verify: `python src/build_tta_pool_repo.py`; `python src/run_kaggle_predict_local.py --repo tmp/c024_tta_pool/tracking_repo ... --env BIOHUB_EDGE_TTA_POOL=js_log_pool`
- Evidence: Every pooling variant lowers the adjusted edge score (held-out 12 0.9416 -> 0.9393-0.9399; 7-10 of 12 movies worse).

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
