# C029 — C024 + C025's settings

**Status**: Submitted 2026-09-26 09:31 KST, ref `56564563`: public **0.950** (-0.004 vs C024). HANDOFF sections 22.5 and 23.

- Notebook: `biohub-c029-ensemble-pp-settings.ipynb` (sha256 `8e103d39f025badd...`)
- Kaggle kernel: `taeyangg4/biohub-c029-ensemble-pp-settings` (private, NvidiaTeslaT4, internet off); datasets: `anvithpothula/biohub-v1284-head-s075`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: Cell 0/1: readmit 3, gap-close 4 (drift guard updated), linefit 0.6 on top of C024.
- Build / verify: `python src/build_env_variant_candidate.py --base <C024 nb> --base-label C024 --label C029 --slug biohub-c029-ensemble-pp-settings --dir experiments/candidates/c029_ensemble_pp_settings --set ...`
- Evidence: Best local number of all candidates (held-out 12 0.9646) and the worst LB of the family: the clearest case of local post-processing gains not transferring. T4 visible-4 0.93431 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
