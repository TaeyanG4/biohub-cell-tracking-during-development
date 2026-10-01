# C028 — C023 + C025's settings

**Status**: Submitted 2026-09-26 09:13 KST, ref `56564110`: public **0.951** (-0.003 vs C023). First submit on 2026-09-25 21:10 KST failed on the shared daily quota. HANDOFF sections 22.5 and 23.

- Notebook: `biohub-c028-x138head-pp-settings.ipynb` (sha256 `f522b951b5cecc1c...`)
- Kaggle kernel: `taeyangg4/biohub-c028-x138head-pp-settings` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`
- Change: Cell 0/1: readmit 3, gap-close 4 (drift guard updated), linefit 0.6 on top of C023.
- Build / verify: `python src/build_env_variant_candidate.py --base <C023 nb> --base-label C023 --label C028 --slug biohub-c028-x138head-pp-settings --dir experiments/candidates/c028_x138head_pp_settings --set ...` (same three settings as C025)
- Evidence: On the x138-head held-out graphs the settings gave +0.0045 locally; the LB says -0.003. T4 visible-4 0.92975 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
