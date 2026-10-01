# C031 — appearance-based division classifier (study, no notebook)

**Status**: NO-GO. HANDOFF section 24.

- Change: 3D CNN on 3-frame (8 x 32 x 32 voxel) crops around GT nodes: 151 GT divisions vs 7,960 one-child GT nodes (`crops/`, `extract.log`, `train_cv.log`, `division_cnn.pt` / `.json`).
- Build / verify: `python src/division_crops_extract.py --out experiments/candidates/c031_division_cnn/crops`; `python src/division_cnn_train.py --crops experiments/candidates/c031_division_cnn/crops --out experiments/candidates/c031_division_cnn/division_cnn.pt`
- Evidence: Movie-grouped 5-fold OOF: precision 0.58 at recall 0.5 (bar 0.6), 0.71 at recall 0.16, on a 1:53 sample. At the pipeline's ~2000:1 candidate base rate this is ~6 % precision; train labels are also noisy (forum thread 742942).

Written 2026-09-26 from `HANDOFF.md` and `experiments/submission_log.csv`; those two files stay the source of truth.
