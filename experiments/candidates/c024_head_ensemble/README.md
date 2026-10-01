# C024 — C022 with a 2-head ensemble (ours v1 + x138 s075)  (FINAL PICK)

**Status**: Submitted 2026-09-25 14:46 KST, ref `56541277`: public **0.954** (best, ties C023). Recommended final pick together with C023. HANDOFF sections 22.4 and 23.

- Notebook: `biohub-c024-head-ensemble.ipynb` (sha256 `5d871311c3af9351...`)
- Kaggle kernel: `taeyangg4/biohub-c024-head-ensemble` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: Cell 4: both heads are mounted (SHA-pinned) and the embedded `v1284_coordinate_refinement.py` averages their bounded shifts (each < 2 um).
- Build / verify: `python src/build_head_ensemble.py notebook` (defaults = ours v1 + x138 s075). The main notebook in this folder is the **scored version pulled from Kaggle** on 2026-09-26 (`scored_v1/`, identical copy); `rebuild_20260925_generalised_builder.ipynb` is the later rebuild with the N-head builder (cells 0 and 4 differ in the mount-helper text only; same logic). C029 was built from that rebuild.
- Evidence: Held-out 12 0.9593 (6bba 0.9579, 44b6 0.9571). On the LB the mean of the two heads equals x138 alone. T4 visible-4 0.93147 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
