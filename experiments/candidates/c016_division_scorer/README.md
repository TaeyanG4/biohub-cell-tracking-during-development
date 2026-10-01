# C016 — C012 + learned division scorer

Status: **NO-GO (2026-09-24 12:10 KST)** — the learned scorer reaches only 6–11 % per-parent precision at 25 % recall (pre-registered criterion: ≥ 35 %); nothing uploaded or pushed. See HANDOFF section 20 for the full record.
Source of truth for decisions: `HANDOFF.md` section 20.

## Why
C012 (x138 pipeline + own V1284 head v1) scores 0.952 on the LB. Its remaining systematic loss is
divisions: on the 12 held-out movies 14 of 24 GT divisions are predicted as a single continuing track
("no_fork"), typically because one daughter sits 1–2 µm from the parent (linked) and the other 7–14 µm
away (linked elsewhere by post-processing or left orphan). x138's rule-based safe-division stage
(parent ≤ 9 µm, sister ≤ 14 µm, symmetry ≤ 0.6, **orphan D2 only**, DeepCenter veto) cannot accept them.
The division term is 0.1 × micro Jaccard, so recovering half of them without extra false forks is worth
roughly +0.005 LB.

## What changes vs C012 (one change)
Cell 5 of the notebook: `add_safe_divisions_postlink` is replaced by a scored stage
(`src/division_scorer_stage.py`, embedded verbatim). Candidates are (P, D1, D2) triples — P at t with
exactly one child D1, D2 another node at t+1 within 14 µm (sister ≤ 16 µm); D2 may be free or already
parented by Q. A small torch model scores 40 features; accepted candidates add the edge P→D2 (removing
Q→D2 only above a stricter threshold and never when the ILP gave Q→D2 ≥ 0.95). x138's rule remains the
fallback if the scorer or the low-detection dump is missing. Everything else (cells 1–3, 6–11, every
environment setting) is byte-identical to C012; `src/verify_c016.py` enforces that.

## Files
- `batch_00..07.txt`, `train_stems_ordered.txt`, `gt_division_counts.json` — the 177 non-evaluation train
  movies ordered by GT division count (127 GT divisions in batches 00–02 = 75 movies).
- `e2e/head_v1_bNN/` — local C012-configuration inference (head v1, candidate mode) for each batch:
  `predictions/*.geff` (ILP graphs) + `edge_cache/*.npz` (low-detection dumps).
- `candidates_bNN.csv`, `candidates_heldout12.csv`, `candidates_confirm10.csv` — labelled candidates
  (`src/division_candidates_local.py`); labels follow the official `score_divisions()` rules.
- `smoke/` — throwaway tooling tests (timing), never used for decisions.
- `dataset/` — the scorer file uploaded as the private dataset `taeyangg4/biohub-division-scorer`.

## Protocol (overfitting guards)
1. Train only on batches 00–02 (+ confirm-10 if positives are scarce); the held-out 12 are never trained on.
2. Movie-grouped CV; prefer the simplest model (logistic regression) unless the MLP is clearly better OOF.
3. Go/no-go: OOF precision ≥ 0.35 on labelled candidates (baseline held-out division counts are
   TP 3 / FP 2 / FN 12, so the metric break-even is ≈ 0.15) at a recall that recovers ≥ 3 held-out
   divisions; then end-to-end 6bba proxy on the held-out 12 (`python src/verify_c016.py --scorer <pt> --replay`)
   must beat C012 by ≥ 0.005 local on most 6bba movies that contain divisions.
4. If it fails, C016 is not submitted; C012 stays the best entry.

## Results
- Strict-positive MLP (v3 features, 60 training movies, movie-grouped CV, per-parent): OOF 17 TP / 142 FP at 0.9 (10.7 %); held-out 12: 2 TP / 30 FP (6 %). Logistic and all-kinds MLP: no better. Top-20 OOF parents: 9 TP / 11 FP (45 %) — an ultra-conservative threshold is worth ~+0.0002 LB, inside the noise.
- Root cause: base rate (~2.5 recoverable divisions vs ~5,000 candidate parents per movie) and the absence of a learned P→D2 edge probability (the ILP never proposes the far daughter).
- Kept for reuse: `src/division_scorer_stage.py` (48 features), the metric-faithful labeller, per-parent evaluation, candidate CSVs (`candidates_*.csv`, 97 movies).
