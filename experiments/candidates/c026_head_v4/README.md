# C026 — C022 with head v4 (trained on 187 captured movies)

**Status**: Built, never pushed (worst local end-to-end head). HANDOFF section 22.4.

- Notebook: `biohub-c026-head-v4.ipynb` (sha256 `6ffa6d61b1dd8c36...`)
- Kaggle kernel: `taeyangg4/biohub-c026-head-v4` (private, NvidiaTeslaT4, internet off); datasets: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `taeyangg4/biohub-v1284-head-v4`
- Change: Cell 4: head v4 from the private dataset `taeyangg4/biohub-v1284-head-v4` (`v1284_head_v4.pt`, sha bda96832...).
- Build / verify: `python src/build_head_ensemble.py notebook --head taeyangg4/biohub-v1284-head-v4=<sha>:v1284_head_v4.pt --label C026 --slug biohub-c026-head-v4 --dir experiments/candidates/c026_head_v4`
- Evidence: Best pair-level localisation (held-out pairs 1.223 um vs v1 1.255) but held-out 12 end-to-end 0.9465 vs v1 0.9592: larger shifts move the transformer's feature lookups off the detector peaks.

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
