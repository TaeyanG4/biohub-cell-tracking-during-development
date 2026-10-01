# C027 — C022 with a 3-head ensemble (ours v1 + x138 s075 + v4)

**Status**: Submitted 2026-09-26 09:32 KST as an LB probe, ref `56564591`: public **0.953** (-0.001 vs C024). HANDOFF sections 22.4 and 23.

- Notebook: `biohub-c027-head-ensemble3.ipynb` (sha256 `8ca3909fde8c95f1...`)
- Kaggle kernel: `taeyangg4/biohub-c027-head-ensemble3` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`, `taeyangg4/biohub-v1284-head-v4`
- Change: Cell 4: three SHA-pinned heads, mean of the bounded shifts.
- Build / verify: `python src/build_head_ensemble.py notebook --head <ours v1> --head <x138 s075> --head <v4> --label C027 --slug biohub-c027-head-ensemble3 --dir experiments/candidates/c027_head_ensemble3`
- Evidence: Held-out 12 0.9543 (6bba 0.9505, 44b6 0.9592). T4 visible-4 0.93218 = local. Adding v4 hurts on the LB as locally.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
