# C034 appearance ReID diagnostic

**Closed 2026-09-27: insufficient negative labels, hypothesis untested.** All 22 extractions/control checks passed. Of 4,928 pairs, 2,302 are positive, 16 known negative (44b6: 6, 6bba: 10) and 2,610 unknown. Both cross-embryo fits were skipped below the fixed 20-per-class minimum. No learned ReID model, AUC result, graph-score change or submission exists. See `FINAL_REVIEW.md` and `decision.json`. Launch/design notes below are historical; do not restart solely to consume slots.

User requested this next experiment after C032/C033 were closed. Base is unchanged C023 with x138 head and the same fresh FP32 control caches from the completed 97-movie batch. This is a local diagnostic; no new submission notebook or model is ready.

`src/reid_probe_local.py smoke|run|analyse` reuses the already-pulled arnav170 public descriptor functions and the existing `eval_pp_variants_local` namespace, graph loaders and official scorer. It does not execute the public notebook's training, relink, submission or division code.

## What is recorded

The C023 relink's actual tight/relaxed candidate costs, raw/motion distances, learned probabilities, ranks and final assigned target are recorded without changing them. Seed assignments are ignored. The final relink invocation is retained because C023 can relink again after readmitting detections. Original image coordinates are captured before global stabilization; shifted geometry never determines image crop locations.

Only sources with a GT-matched, consecutive, uniquely annotated successor are eligible; division and immediately adjacent GT events are excluded. The successor must be in the actual gated candidate list and at least one competitor must exist. Matching uses C023's existing bipartite function at 7 um and its integer output-coordinate convention. Positive = matched annotated successor. Negative = a different GT-matched target. Unmatched targets have label -1 and are excluded from fitting. They remain in ranking diagnostics. Candidate sets are not expanded to make classification artificially easy.

Public 7x25x25-voxel descriptors are computed on original frames. Geometry-only and geometry+appearance models use the same candidate sets and fixed HistGradientBoosting recipe. Appearance uses intensity/shape/radial/size differences and cosine features; public context and DeepCenter features are excluded. DeepCenter is still used by C023's unchanged baseline post-processing.

## Validation and compute allocation

- Real smoke on short inputs from both embryos must preserve the exact final nodes, edges and stats. Passed before the full run. An initial recording-state bug (retaining candidates across internal relink invocations) was fixed before the full run; failed smoke and original source retained under `logs/`.
- Every full-movie extraction must reproduce the saved C023 official edge/division counts, adjusted-edge score and node/edge counts. Capture is not allowed to change output.
- Pilot uses the same heldout12+confirm10 movies, with all data from one embryo used to train and the other to test, then reversed. Descriptor scaling is fitted only on each training embryo. There is no random node/frame split, hyperparameter sweep or full-data fitted model submitted.
- Two diagnostic comparisons: appearance vs geometry annotated-successor agreement, and a fixed conservative proposed-change rule against C023 (top probability >=0.8, top-second gap >=0.2). The compute gate requires >=5 net gains in both comparisons in each cross-embryo direction. It decides whether an actual graph-level trial is worth implementing, not whether to submit.
- AUC is secondary. Per-source ranking ignores global assignment competition, node pruning, division and later edge restore. **These counts are not official-score gains.** Capture-time matching can differ from final matching. Base detectors were trained on both embryos; this is not independent hidden-test validation.

## Artifacts and continuation

`status.json`, `smoke_status.json`, `pairs/<stem>.npz`, per-movie `logs/`, `analysis.json`, `RESULTS.md`. Source hashes and exclusive lock prevent an unnoticed source change or duplicate extraction. Finite start-movie budget: 6 hours. Do not edit the script or replay harness while running.

After completion, review known negative sample counts and cross-embryo results. If the gate fails, close the study and stop its follow-up; do not relabel unknown targets as negative or tune thresholds to manufacture a gain. If it passes, implement a degree-consistent restricted relink using the same feature code and existing replay harness, keep genuinely held-out movies out of fitting, then run official 12+10 and prospective expansion before considering Kaggle. Existing user authorization permits worthwhile candidates to be pushed/verified/submitted; it does not justify skipping local/T4 checks. Preserve C023/C024 and user final picks. No automatic submit is performed by this script.
