# C018 — C017 + frozen-frame catch-up prior

**Status**: Rejected as a candidate (97 movies +0.0004); submitted anyway as a slot-use probe on the user's order, ref `56518071`: public **0.952**. HANDOFF section 22.2.

- Notebook: `biohub-c018-frozen-catchup-prior.ipynb` (sha256 `52cf1c2e0fd84e46...`)
- Kaggle kernel: `taeyangg4/biohub-c018-frozen-catchup-prior` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: The notebook detects byte-identical consecutive raw frames; the relink seed pass of the pair after k frozen pairs uses the last clean pair's flow x (k + 1). Toggle `BIOHUB_AFTER_FROZEN_PRIOR`.
- Build / verify: `python src/build_c018_candidate.py`
- Evidence: vs C017: 97 movies +0.0004 (24 better / 14 worse, worst -0.0179); held-out 12 unchanged; confirm-10 +0.0014. T4 visible-4 0.93852 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
