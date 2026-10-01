# Candidates C012 / C013: x138 pipeline + our own V1284-compatible head

## Why

`thtennant/biohub-frontier947-readmit-v1` is byte-identical to `anvithpothula/biohub-x138` except that it lacks the V1284 block, and it scored **0.946** public (C011 is the same pipeline). x138 scored **0.953**, so its gain is the private coordinate head (+0.007). The head's weights are private; its code is public (`v1284_coordinate_refinement.py`, embedded in x138 cell 4): a `Linear(224,32) → SiLU → Linear(32,3)` over frozen primary-UNet features (centre + 6 neighbour differences) with a bounded (< 2 µm) shift applied to every first-seen fused detection.

## How the head was rebuilt

1. **Capture** (`src/v1284_capture_local.py`): the exact patched predict script from our C011 Kaggle run (`tmp/c011_output/tracking_repo`), same public pilkwang weights (SHA256-checked), `V1284_MODE='capture'`, on local train movies. Per movie, detections are matched one-to-one to GT (Hungarian, ≤ 4 µm) and only matched pairs are kept (`pairs/<stem>.npz`). The 12 movies used for end-to-end checks (`heldout_stems.txt` = C004's 8 validator movies + 4 visible) are never captured.
2. **Train** (`src/v1284_head_train.py`): same architecture and `bounded()` output as x138's module, standardised features (`mean`, `scale` saved with the weights, `scale > 0`), movie-grouped 5-fold CV.
3. **End-to-end** (`src/run_kaggle_predict_local.py` → `src/eval_pp_variants_local.py --round-coords`): full inference + ILP locally on the 12 held-out movies with the head, then x138's post-processing (readmit + gap filler active) and the official formula on coordinates rounded exactly like the submission writer.

## Local ↔ Kaggle T4 fidelity (the C008 concern)

- Local zero-mode run vs our C011 T4 run on the 4 visible movies: 99.97 % of detections identical, edge-probability |Δ| median 3e-5 (p99 1.7e-3); visible-4 official score 0.9056 locally and 0.9056 for the T4 submission.
- Local convolutions default to TF32 on the 4070 Ti (the T4 runs plain FP32): feature |Δ| TF32 vs FP32 median 2.6e-4 feature-std, head-shift |Δ| max 0.00057 µm (0.0014 voxel).
- The head is trained on features of the frozen public backbone; nothing is blended into the backbone's logits (unlike C008).

## Results (head v1 = 30 captured movies, 14,084 pairs)

Movie-grouped CV, distance of detection to GT on held-out movies: 1.713 → 1.313 µm (−23.3 %; 44b6 −9.1 %, 6bba −25.5 %; 0/5 folds worse). x138's author reported −10.4 % for their head.

End-to-end on the 12 held-out movies (score = adjusted edge Jaccard + 0.1 × division Jaccard; rounded coordinates):

| variant | all 12 | val8 | vis4 | Δ adj-edge 44b6 | Δ adj-edge 6bba |
|---|---:|---:|---:|---:|---:|
| zero (= C011, ≈ LB 0.946) | 0.9367 | 0.9480 | 0.9056 | – | – |
| **C012**: head v1, candidate (x138's path) | **0.9470** | 0.9539 | 0.9252 | −0.0038 | +0.0116 |
| **C013**: head v1 output layer ×0.5, candidate | 0.9425 | 0.9502 | 0.9175 | +0.0013 | +0.0070 |
| head v1, output-only coordinates (not shipped) | 0.9427 | 0.9510 | 0.9165 | +0.0041 | +0.0054 |
| head v2 (71 movies: 52 × 44b6 + 19 × 6bba, 24,778 pairs; not shipped) | 0.9434 | 0.9516 | 0.9135 | +0.0025 | +0.0078 |

Head v2 CV: 1.619 → 1.301 µm (−19.6 %; 44b6 −13.9 % vs −9.1 % for v1, 6bba −24.7 %). More 44b6 data removes v1's 44b6 edge loss but gives back part of the 6bba gain; v1 remains the best total locally. Which variant the LB prefers is the open question the C012/C013 scores answer.

Readmit ablation (x138's readmit stage, local e2e): removing it scores 0.9478 with head v1 (0.9480 with the gap filler off too) and 0.9404 head-less (+0.0037), matching thtennant's LB (gapfill 0.947 -> +readmit 0.946). A next candidate should drop readmit.

With the head, fewer edges pass the 0.48 candidate threshold (edge model sees sub-voxel lookups) and more ILP-discarded detections are readmitted; association in output-only mode is identical to zero mode (same ILP candidate graph) and still gains +0.0060.

## Kaggle

- Private dataset `taeyangg4/biohub-c012-v1284-head`: `v1284_head.pt` (head v1, sha256 `9d3484f7…`), `v1284_head_a05.pt` (×0.5, `209bc591…`), training report.
- Kernels (private, T4): `taeyangg4/biohub-c012-v1284-head-v1`, `taeyangg4/biohub-c013-v1284-head-a05` (pushed 2026-09-23). Notebooks pin the head SHA256 and fail loudly on a mount or checksum mismatch.
- Submitted on the user's instruction 2026-09-23 05:16 UTC: C012 ref 56483602, C013 ref 56483608 (LB pending at time of writing). Both kernel runs loaded the pinned head; T4 refined coordinates match the local runs within 0.003 um and the visible-4 official scores equal the local predictions (0.9252 / 0.9175).

Build / verify:

```
python src/build_c012_candidate.py --candidate c012 --head experiments/candidates/c012_v1284_head/dataset/v1284_head.pt --dataset-file v1284_head.pt
python src/verify_c012.py --candidate c012 --head experiments/candidates/c012_v1284_head/dataset/v1284_head.pt
```
