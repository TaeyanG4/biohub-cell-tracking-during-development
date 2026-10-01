# C023 — C022 with x138's public V1284 head  (FINAL PICK)

**Status**: Submitted 2026-09-25 12:27 KST, ref `56538803`: public **0.954** (best). Recommended final pick together with C024. HANDOFF sections 22.4 and 23.

- Notebook: `biohub-c023-x138-head-stabilize-restore.ipynb` (sha256 `aa6aaaaf5af076f6...`)
- Kaggle kernel: `taeyangg4/biohub-c023-x138-head-stabilize-restore` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`
- Change: Cell 4 + metadata only: the coordinate head is mounted from the public CC0 dataset `anvithpothula/biohub-v1284-head-s075` (SHA256 625a0d93..., the head behind x138's 0.953) instead of our head v1.
- Build / verify: `python src/build_c023_candidate.py`
- Evidence: Locally worse than C022 (held-out 12 0.9547 vs 0.9592) but +0.001 on the LB: heads are ranked by the LB only (our heads were trained on the same two embryos as the local movies). T4 visible-4 0.93335 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
