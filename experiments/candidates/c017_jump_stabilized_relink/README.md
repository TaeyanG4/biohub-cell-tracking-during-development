# C017 — C012 + jump-stabilized motion relink (4 um)

**Status**: Submitted 2026-09-24 07:55 UTC, ref `56515876`: public **0.952** (= C012). HANDOFF section 22.1.

- Notebook: `biohub-c017-jump-stabilized-relink.ipynb` (sha256 `9b2ada599121b59a...`)
- Kaggle kernel: `taeyangg4/biohub-c017-jump-stabilized-relink` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-c012-v1284-head`
- Change: Cell 5: `motion_relink_edges` is wrapped; frame pairs whose ILP links move coherently by >= `BIOHUB_STAB_MIN_UM` (4.0) are relinked on jump-free coordinates (cumulative jump sum; geometry only). Same code as the harness option `--stabilize-relink`.
- Build / verify: `python src/build_c017_candidate.py --min-um 4.0`; `python src/verify_c017.py --replay` (text diff vs C012 + notebook == harness on the held-out 12).
- Evidence: Held-out 12 0.9470 -> 0.9583, held-out 6bba 0.9510 -> 0.9608, confirm-10 +0.0066, 97 movies +0.0071 (52 better / 9 worse). T4 visible-4 0.93852 = local. The 6bba proxy predicted ~0.955; the hidden test has fewer acquisition jumps than the proxy movies.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
