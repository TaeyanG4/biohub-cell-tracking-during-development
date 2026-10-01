# C030 — C023 + AdaBN test-time BatchNorm adaptation

**Status**: Built, never pushed: unstable by embryo. HANDOFF section 24.

- Notebook: `biohub-c030-x138head-adabn.ipynb` (sha256 `5273325f3cc2861a...`)
- Kaggle kernel: `taeyangg4/biohub-c030-x138head-adabn` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`
- Change: Cell 4 appends three anchored patches to the predict script (`src/build_adabn_repo.py`): before each movie the BatchNorm3d running statistics of the primary and secondary UNets are recomputed from that movie's frames; restored afterwards. `BIOHUB_ADABN=1`.
- Build / verify: `python src/build_c030_candidate.py` (asserts the embedded patches equal `build_adabn_repo.patch_text`).
- Evidence: Held-out 12 with the x138 head: both UNets -0.0004 (6bba +0.0047, 44b6 -0.0193, nodes +11 % on 44b6); primary only -0.0009 (6bba +0.0062, 44b6 -0.0231). Swings of +-0.02 by embryo make the hidden-embryo effect unpredictable.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
