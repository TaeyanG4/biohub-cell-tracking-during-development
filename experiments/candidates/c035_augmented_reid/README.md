# C035: GT temporal supervision and modest augmentation for appearance ReID

Registered 2026-09-27 after the user's request to try promising augmentation approaches one by one. C034 did not fit any classifier because reliable negatives were scarce; this is a new training construction, not another run of that classifier. C023/C024 and final picks remain unchanged.

## Fixed first study

- `python src/reid_augmented_local.py run` uses the global CUDA Python on the local RTX 4070 Ti SUPER. Hidden background process, exclusive lock, six-hour execution budget, hashed sources and resumable phase artifacts. No Kaggle operations in this program.
- Exclude **all 97 official evaluation movie IDs** from new training, leaving 44 movies from 44b6 and 58 from 6bba before eligibility filtering. Choose 12 per embryo with at least 120 eligible unique links in a deterministic SHA256 order, based only on GT availability. Save the full inventory and selection in `plan.json` before training.
- Positives are actual consecutive GT links with unique child/parent. Negatives are the nearest distinct annotated nucleus at the **child's frame**, 6–40 um from the true child. Exclude division neighbourhoods (two lineage steps), gaps and near-coincident labels. Maximum 192 sampled links/movie, equal movie sampling in training. Unknown C034 targets are never negative training examples.
- Reuse `division_crops_extract.crop_at`: 8x32x32 voxel single-frame crops, movie intensity quantiles, clipping [0,3], original z/y/x coordinates. Decode each required frame once. Voxel scale is (1.625, 0.40625, 0.40625) um. No coordinate, distance, time, embryo ID or geometry feature enters the CNN.
- Reuse `DivisionCNN`'s width-8 backbone with a normalized 32-dimensional embedding. Triplet cosine loss, AdamW, 1,200 steps, batch 48, fixed seed and final checkpoint. No target-embryo model selection.
- Two fixed arms: **basic** = shared XY flips/transposition; **weak_aug** = the same plus independent +/-1.5 xy voxel / +/-0.3 z voxel shifts, gain 0.85–1.15, background +/-0.025, Gaussian noise up to 0.015 and slight lateral smoothing. No z/XY swaps, z flip, aggressive deformation or synthetic negative labels.
- Train both arms on 44b6 and test 6bba; train both on 6bba and test 44b6. Four small fits total. FP32, TF32 off, no AMP. This policy does not prove numerical equivalence to T4.

## Checks and diagnostic

Actual GT crops from both embryos must pass gradient/update/reload checks. For all 22 evaluation movies, import C034's unchanged Recorder and replay C023 with the existing official namespace. Reproduce stored control scores/counts and **exactly match C034 sources, targets, costs, labels and selections**. Get crops from Recorder's original pre-stabilization coordinates, including readmitted nodes; do not index edge caches by these IDs.

Use the opposite-embryo model to score the **existing real candidate groups**. Also report untrained downsampled patch NCC as a diagnostic control. No graph edges change here. Unknown targets remain unknown: a retrieval miss is not a claim every unmatched target is biologically false.

Fixed conservative proposal: change only an existing selection, appearance gain >=0.10, top-versus-second cosine margin >=0.05, C023 cost increase <=1.0. All candidates already pass the original gates. **Compute gate per learned arm:** at least five net top-rank recoveries vs C023 cost argmin AND five net conservative recoveries vs actual C023 selection in **each embryo direction**, with positive conservative net in at least two movies per direction. No threshold sweeps after seeing results. A positive augmented-vs-basic comparison is needed before attributing an effect to augmentation rather than the new supervision.

The gate only warrants graph-integration review. Assignment competition, later postprocessing and official metric effects are untested by ranking. Use the existing `eval_pp_variants_local.py` for restricted degree-consistent integration, exact off-control checks and official 12+10, then matched extension75. Any potential notebook must execute the same crop/model/stage code, pass actual local replay and T4 visible4 checks before submission.

## Follow-up

Read compact `status.json`; while healthy/running, one status check and finish without waiting. Do not modify hashed sources or start another copy. On failure, inspect the specific traceback and preserve provenance before fixing/restarting. `analysis.json`, `RESULTS.md`, `movie_diagnostics.csv` and model logs are authoritative when complete. Examine both embryo directions, movie concentration, negative-distance distribution, and both fixed arms. If unsupported, close honestly; no AUC or ranking number substitutes for official graph gain.

All public detectors saw both embryos, and GT-labelled negatives may be farther/easier than real ambiguous detections. Even cross-embryo success for the new CNN cannot establish hidden-test benefit. Keep C032/C033 closed and C034's original insufficient-label conclusion intact.

Standing user authorization includes worthwhile pushes, T4 checks and exact-version submissions. Inspect ledger/quota first, avoid duplicate or unsupported submissions and blind 400 retries. No LB polling. The finite `biohub-t4` heartbeat follows this study and stops when closed or verified candidates are submitted; no active model-turn waiting.
