# C022 — C020 + ILP-edge restore

**Status**: Submitted 2026-09-24, ref `56519304`: public **0.953**. Post-processing base of C023-C030. HANDOFF section 22.2.

- Notebook: `biohub-c022-stabilize-all-restore.ipynb` (sha256 `c84ca04c5d2a1328...`)
- Kaggle kernel: `taeyangg4/biohub-c022-stabilize-all-restore` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: C020 (every pair stabilized) + the C021 restore stage at p >= 0.7.
- Build / verify: `python src/build_c021_candidate.py --base experiments/candidates/c020_stabilize_all_pairs/biohub-c020-stabilize-all-pairs.ipynb --base-label C020 --slug biohub-c022-stabilize-all-restore --dir experiments/candidates/c022_stabilize_all_restore --label C022 --min-prob 0.7`
- Evidence: vs C017: 97 movies +0.0029, held-out 6bba +0.0015, confirm-10 +0.0003. Notebook replay == harness on all 12; T4 visible-4 0.93893 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
