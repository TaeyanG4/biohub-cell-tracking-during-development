# C025 — C022 + readmit 3 um / gap-close 4 um / linefit 0.6

**Status**: Submitted 2026-09-25 17:48 KST, ref `56546101`: public **0.952** (-0.001 vs C022). The settings do not transfer. HANDOFF sections 22.4 and 22.6.

- Notebook: `biohub-c025-readmit3-gap4-linefit06.ipynb` (sha256 `d036511c94580e13...`)
- Kaggle kernel: `taeyangg4/biohub-c025-readmit3-gap4-linefit06` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: Cell 0 only: `BIOHUB_READMIT_RADIUS_UM` 3, `BIOHUB_GAP_CLOSE_UM` 4.0 (cell-1 drift guard updated), `BIOHUB_OUTPUT_LINEFIT_WEIGHT` 0.6.
- Build / verify: `python src/build_env_variant_candidate.py --base <C022 nb> --base-label C022 --label C025 --slug biohub-c025-readmit3-gap4-linefit06 --dir experiments/candidates/c025_pp_settings --set BIOHUB_READMIT_RADIUS_UM=3 --set BIOHUB_GAP_CLOSE_UM=4.0 --set BIOHUB_OUTPUT_LINEFIT_WEIGHT=0.6`
- Evidence: Best of a 24-variant sweep + additivity check: 97 movies +0.0018, positive in every group (held-out 12 +0.0028, confirm-10 +0.0014, batches +0.0018). Lost 0.001 on the LB: winner's curse on in-sample movies. T4 visible-4 0.94186 = local.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
