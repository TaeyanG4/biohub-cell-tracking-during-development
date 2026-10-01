# C056 complementary-detector guarded readmission on C023

Follow-up of C055 (`experiments/candidates/c055_guarded_readmit/REVIEW.md`).
C055 showed that V12's 0.94-0.965 primary-detector range holds almost no
annotated missed cells (6 of 213 on 22 movies), while 101 missed cells have no
primary peak at all and the 2026-09-28 priority review found StrongUNet peaks
at 16 of 21 such cells on the visible movies.  This study feeds the unchanged
V12 guard (step <= 3 um, residual <= 1.5 um, separation >= 1.5 um, margin 0.5,
bridges, budget min(100, 0.2%)) with StrongUNet peaks at the fixed published
threshold p >= 0.5, replacing C023's own readmission (off in the new arms).

Arms: `control`, `readmit_off` (must equal C055's rows exactly),
`strongunet_guarded`.  Peaks: `peaks/<stem>_peaks.parquet` cached with the
existing `src/cache_strongunet_gpu_peaks.py` model on `data/train` movies.
Diagnostic ceiling: `diagnose.json` (missed GT nodes with a StrongUNet peak).
Gate and outputs: see the driver docstring; REVIEW.md after completion.
No Kaggle operation is performed by this queue.
