# Validation: what we measured, and where it misled us

Local validation decided almost every step of this project, so its limits explain most of the final result. This page records the protocol as it was used, the three corrections we made during the competition, and how well it predicted the leaderboards in the end.

## The local protocol

| Set | Movies | Purpose |
|---|---|---|
| held-out 12 | 6 x `44b6`, 6 x `6bba` | first screen; also used to fit the 6bba proxy below |
| confirm 10 | 10 x `6bba` | second screen on movies our own coordinate head had not seen |
| extension 75 | division-rich movies | larger check before a candidate became a submission |
| visible 4 | the 4 test movies Kaggle shows during the run (copies of training movies) | equality check between the local run and the Kaggle T4 run |

Lists: [`heldout_stems.txt`](../experiments/candidates/c012_v1284_head/heldout_stems.txt), [`confirm_stems.txt`](../experiments/candidates/c012_v1284_head/confirm_stems.txt), `experiments/candidates/c016_division_scorer/stems_b0{0,1,2}.txt`.

Every number was produced by replaying the candidate notebook's own code (local inference with `src/run_kaggle_predict_local.py`, post-processing with `src/eval_pp_variants_local.py`) and scoring with the organisers' metric code pinned to commit `075fc5f5`.

## Correction 1 (known from day one, under-weighted): the detector had seen every movie

The public detector, DeepCenter and node Transformer were trained on all 199 training movies (their `split_manifest.json` lists all of them). Any local movie is therefore in-sample for the detector: detections are cleaner than on a new embryo, and later stages look better than they will be on the test set. Other competitors reported the same effect in the forum, including sign flips between local and leaderboard results.

We knew this from 2026-09-24 and responded by raising the bar (local gains below about 0.005 treated as unproven) instead of building out-of-fold detector predictions. That kept us from shipping noise, but it also stopped small real gains from accumulating, and it left the downstream components tuned on detections that were better than the test-time ones.

## Correction 2 (2026-09-28): movie IDs are not independent samples

The 199 movies are crops of only two embryos (71 `44b6`, 128 `6bba`). A pixel audit found byte-identical image blocks between movies we had treated as "training" and "evaluation" for the appearance models (for example `44b6_7e557709` vs `44b6_12dfb391`, 4,608 voxels identical at two time points). From C048 on, every newly fitted component was trained on one embryo and evaluated only on the other. Earlier pooled local gains (C041-C046) were re-labelled as fit-domain evidence.

## Correction 3 (2026-09-29): our scorer was a replica, not the official code

Historical columns named `official_*` came from an internal replica of the metric. Re-scoring the same saved C023 graphs on 22 movies with the organisers' code gave 0.943113 instead of 0.945986 (divisions 4/5/20 instead of 5/6/19 TP/FP/FN). The difference is not rounding. All 388 saved graphs of C023/C052/C053/C054 on 97 movies were re-scored with the actual code ([`state/validation_audit_20260929/REVIEW.md`](../state/validation_audit_20260929/REVIEW.md), in Korean); candidate rankings did not change, but absolute local numbers before 2026-09-29 should be read as approximate.

## How well local numbers predicted the leaderboards

On 2026-09-24 a linear fit on six submissions matched the public board closely (`public = 0.298 x local_6bba + 0.668`, Pearson 0.994). It kept working for public ties, but neither it nor any other local measure predicted the private board:

![public vs private](figures/public_vs_private.png)

- Public 0.954 entries landed anywhere from 0.917 to 0.923 on the private board; public 0.952 C012 landed at 0.924.
- The cluster split is by coordinate head: every submission built on x138's public head (orange) sits at 0.917-0.919 private; every submission on our own head or the two-head mean (blue) sits at 0.920-0.924.
- The public test embryo is much denser than the private one (median detections per frame roughly 160-260 vs 60-100, as probed by the 3rd-place team). Choices tuned on public score therefore drifted toward the dense-embryo regime.

## What a sound protocol would have looked like

1. Out-of-fold upstream predictions: retrain (or at least fine-tune) the detector per fold so later stages are tuned on test-like detections.
2. Whole-embryo outer folds for every learned component from the start, reporting both directions separately.
3. The organisers' scorer from the first day, with per-term loss decomposition (edges, node term, divisions).
4. Leaderboard reads used as single-variable measurements with a pre-registered expected score, not as a ranking of near-ties.
5. Final picks from two structurally different families, not two public ties from the same family.
