# C020 — every frame pair stabilized

**Status**: Submitted 2026-09-24 09:43 UTC, ref `56518073`: public **0.953**. HANDOFF section 22.2-22.3.

- Notebook: `biohub-c020-stabilize-all-pairs.ipynb` (sha256 `118fabbce0eae211...`)
- Kaggle kernel: `taeyangg4/biohub-c020-stabilize-all-pairs` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: Same builder as C017 with `STAB_MIN_UM` 0.001, i.e. the global shift of every frame pair is removed before the relink.
- Build / verify: `python src/build_c017_candidate.py --min-um 0.001 --slug biohub-c020-stabilize-all-pairs --dir experiments/candidates/c020_stabilize_all_pairs --label C020`
- Evidence: vs C017: 97 movies +0.0018 (more movies touched, larger losers than 4.0), held-out 6bba +0.0020. T4 visible-4 0.94069 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
