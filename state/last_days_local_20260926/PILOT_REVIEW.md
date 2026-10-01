# C032/C033 pilot review — 2026-09-26

All 23 pilot jobs completed with return code 0, including actual temporal-control inference, C033 embedded-notebook equivalence, eight 12/10-movie inference runs and their official replay. The queue stopped before any extension inference with `expected 75 distinct extension movies`: the driver read each comma-separated C016 list as one line. This was a queue input-format bug, not a model/inference failure.

## Scores

Size-weighted official local metric; all deltas against fresh C023 with the same FP32/math-SDPA policy and x138 head. These are two training embryos, not independent hidden embryos. Confirm10 contains only 6bba. Summary CSVs and per-movie rows are in `replay/`.

| Arm | Heldout12 score | Delta | Confirm10 score | Delta |
|---|---:|---:|---:|---:|
| C023 control | 0.955162 | — | 0.937308 | — |
| mean_det | 0.955836 | +0.000674 | 0.940256 | +0.002948 |
| future_det | 0.956464 | +0.001303 | 0.938445 | +0.001138 |
| mean_det_head | 0.957472 | +0.002310 | 0.936549 | -0.000759 |
| structured_raw | 0.954105 | -0.001057 | 0.936995 | -0.000313 |
| structured_stabilized | 0.954105 | -0.001057 | 0.936995 | -0.000313 |

Future_det confirm10 adjusted-edge delta is +0.003189, partly offset by division FP increasing from 4 to 6 (TP=2, FN=7 unchanged). All other score deltas equal their adjusted-edge deltas; division counts do not change.

| Temporal arm | Heldout 44b6 delta | Heldout 6bba delta | Edge wins/losses heldout12 | Edge wins/losses confirm10 |
|---|---:|---:|---:|---:|
| mean_det | +0.001162 | +0.000554 | 6 / 6 | 6 / 4 |
| future_det | +0.006102 | +0.000065 | 8 / 4 | 6 / 4 |
| mean_det_head | +0.007500 | +0.000975 | 6 / 6 | 5 / 5 |

Material regressions: future_det loses 0.061629 adjusted-edge on 44b6_0b24845f; mean_det loses 0.016815 on 6bba_3db54e20. Positive aggregate scores do not mean uniformly safer tracking. Final node-count deltas (heldout/confirm): mean_det +1589/+328; future_det -62/-239; mean_det_head +1643/+327.

## Numerical control

Four movies shared with existing T4 C023 cache have identical detection counts and 100% matched detections. Coordinate difference p99 is 6.20e-6 um; edge-probability difference p99 is 7.75e-7, fraction above 0.01 is zero. The fresh control's visible4 official score is 0.9333514696, matching the historical C023 T4 score to reported precision. This is strong baseline agreement, not bitwise equivalence or a T4 execution check of C032.

Local visible4 expected scores from this run: mean_det 0.936878; future_det 0.933084; mean_det_head 0.935551. Any new Kaggle run must be checked against its own expected local result and resource limits; no Kaggle operations were performed here.

## Reviewed continuation

- Original automatic compute gate selects future_det only.
- Add mean_det with the explicit driver option `--extra-extension-arm mean_det`: both pilot score and adjusted edge improve, and no extra division FP appears. This is a post-review compute-allocation decision, not a prespecified gate pass. No model settings are tuned.
- Do not extend mean_det_head or either structured mode.
- Reuse the 23 completed pilot jobs; run a fresh C023 control, future_det and mean_det on 75 additional unique movies. All 97 movie IDs are distinct, and extension GT exists for every ID.
- Original failed state, driver and launcher logs are retained in `review_before_resume_20260926/`. Only the queue driver changes; all previously tracked model/build/inference/scoring source hashes match the original run. `resume_review.json` records the exact hash migration. The driver now handles comma/whitespace/BOM lists, validates disjointness and records input hashes; invalid/duplicate/overlapping lists were tested.
- C023/C024 remain the scored anchors. Read status.json to determine current completion; this report describes the completed pilot only. User handles submission; no uploads, pushes or submissions.
