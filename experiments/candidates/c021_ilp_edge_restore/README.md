# C021 — C017 + ILP-edge restore (josephadamski V1057)

**Status**: Submitted 2026-09-24 on the user's order, ref `56519685`: public **0.953**. HANDOFF section 22.2.

- Notebook: `biohub-c021-ilp-edge-restore.ipynb` (sha256 `65bed7384a0f9026...`)
- Kaggle kernel: `taeyangg4/biohub-c021-ilp-edge-restore` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: `filter_output_graph` is wrapped: after all post-processing, ILP edges with p >= `BIOHUB_V1057_MIN_PROB` (0.7) that the relink displaced are put back (conflicts by descending probability; forks and their daughters untouched; no nodes added or removed). Harness option `--ilp-edge-restore`.
- Build / verify: `python src/build_c021_candidate.py`
- Evidence: vs C017: 97 movies +0.0017 (36 better / 11 worse). T4 visible-4 0.93631 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
