# C019 — C017 + ILP-seeded relink flow

**Status**: Rejected locally, never pushed. HANDOFF section 22.2.

- Notebook: `biohub-c019-ilp-seeded-flow.ipynb` (sha256 `e8a384f9b143f835...`)
- Kaggle kernel: `taeyangg4/biohub-c019-ilp-seeded-flow` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: The relink seed pass of pair t uses a k-NN flow field built from the ILP's own links of pair t instead of the previous pair's field. Toggle `BIOHUB_ILP_SEED_FLOW`, `BIOHUB_ILP_SEED_MIN_PROB`.
- Build / verify: `python src/build_c019_candidate.py`
- Evidence: vs C017: held-out 12 +0.0000 (p >= 0.5) / +0.0004 (p >= 0.8); confirm-10 +0.0002 / +0.0006, worst -0.0165. The seed prior hardly matters because x138 re-derives the flow from the same pair's seed matches.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
