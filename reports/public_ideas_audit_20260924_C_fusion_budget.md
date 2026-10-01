# Public notebook audit C: geometric-fusion forks, DAE, node budget (2026-09-24)

Scope: 14 public notebooks assigned in batch C. This was read-only: only notebook source was pulled (`kaggle kernels pull`). Nothing was pushed or submitted, and no datasets or weights were downloaded.
Method: I extracted all code and markdown cells and diffed them cell by cell against three references:
- `biohub-x138` (0.953, our base)
- `amanatar_geometric_fusion` (0.948)
- `biohub-harmonic-fusion` (HF = flexonafft/biohub-harmonic-fusion, 0.947)

I also diffed every `os.environ[...]` config against HF and x138.

LB values come from the radar DB (`public_score` = current version, `best_public_score` = best across versions).

Pulled sources are under `state/notebook_radar/pulled/<author>__<slug>/`. I also pulled `ghazarosbarseghyan91/biohub-dae-alpha-0-17` as evidence for #7.

## Summary

| # | ref | LB | parent | class | one-line verdict |
|---|---|---:|---|---|---|
| 1 | kksky9k/geofusion-det955-20260922 | 0.948 | (amanatar?) | INACCESSIBLE | 403 again on 2026-09-24; title = geometric fusion + det 0.955 (a det-threshold knob we already tried) |
| 2 | thedyingkai/biohub-geofusion-948 | - | (amanatar?) | INACCESSIBLE | 403, gone from author listing |
| 3 | thedyingkai/biohub-geofusion-pin-nomnn | - | (amanatar?) | INACCESSIBLE | 403; title suggests mutual-NN gate off (looser division) |
| 4 | chenwensheng/bh948-amangeo | - | (amanatar?) | INACCESSIBLE | 403 |
| 5 | chenwensheng/bh948-amanhack | - | (amanatar hack?) | INACCESSIBLE | 403; name suggests a fork of amanatar's "metric-hack" notebooks (likely EXPLOIT, unverified) |
| 6 | crystalbaby/biohub-geometric-fusion | - | amanatar | TRIVIAL-FORK | amanatar verbatim; only the config-guard cell is re-worded |
| 7 | ghazarosbarseghyan91/biohub-dae-alpha-0-15 | 0.946 | older HF profile | NEW-MODEL | per-video denoising-AE pre-filter blended into the detector input; LB vs alpha is non-monotone, which looks like noise |
| 8 | ghazarosghazaros/biohub-dae-self-distill-repeat | 0.937 | #7 | NEW-MODEL (negative) | peak-sharpening "self-distill" of detection logits; -0.009 vs #7 |
| 9 | lonnieqin/biohub-gap2-joint-node-budget | 0.915 | yusuketogashi "Biohub 159B/162" line | NEW-POSTPROCESS + NEW-MODEL | 4 mechanisms we never tried; all flat on LB in their own lineage |
| 10 | busyaprime/biohub-what-one-link-node-and-division-are-worth | - | ? | INACCESSIBLE | 403; title = metric marginal-value analysis, not a method |
| 11 | howonkang/biohub-0947-short5-prepp-r1 | 0.945 | HF | NEW-POSTPROCESS (negative) + TRIED knob | removes <=5-node components before post-processing; -0.002 vs HF |
| 12 | haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m | 0.944 | HF | TRIED | mutual-best rank bonus + density-group relink + DivNet gate |
| 13 | kunaldesale2408/biohub-cell-tracking | 0.942 | amanatar | TRIED (division-loosening family) | ungated second-pass division repair; -0.006 vs amanatar |
| 14 | evgendvorkin/biohub-0-947-lb-proxy-score-0-9490 | 0.947 | HF | TRIVIAL-FORK | HF verbatim (whitespace only); PROXY_SCORE = HF's own val8 validator (details below) |

Bottom line: batch C has no idea with LB evidence of stacking on x138. The untried mechanisms (#9's lookahead, local ranker, JS-reliability TTA; #7's DAE) have either flat or noisy evidence. The ranked list is at the end.

---

## Per-notebook detail

### 1. kksky9k/geofusion-det955-20260922 | LB 0.948 | INACCESSIBLE
- `kaggle kernels pull` returns 403 twice today (and on 09-23). It is not listed under the author (`kernels list --user kksky9k` gives "Not found"), and no public copy turns up under the search terms "geofusion" or "det955".
- The radar last saw it on 2026-09-23 02:17 UTC. Its tags come from the title only.
- Title-only guess, as already recorded in HANDOFF section 17: amanatar geometric fusion with `BIOHUB_DET_THRESHOLD` 0.955.
  - Its 0.948 equals amanatar's own 0.948, so it shows no gain.
  - Detection thresholds 0.95/0.96/0.965 are in our tried list.

### 2-4. thedyingkai/biohub-geofusion-948, thedyingkai/biohub-geofusion-pin-nomnn, chenwensheng/bh948-amangeo | no LB | INACCESSIBLE
- All return 403 and are gone from the authors' public listings. The chenwensheng listing now shows only three July-August notebooks, and thedyingkai gives "Not found". All were last seen on 09-23 and none was ever scored.
- Titles suggest amanatar geometric-fusion copies. "pin-nomnn" probably means a pinned PPSWEEP config with `SAFE_DIV_REQUIRE_MUTUAL_NN=0`, i.e. a looser division gate, which belongs to a family we already rejected. This is unverified.

### 5. chenwensheng/bh948-amanhack | no LB | INACCESSIBLE
- 403. The name points at amanatar's "metric-hack" notebooks (`amanatar/biohub-metric-hack-last-call` 0.947, `amanatar/improved-metric-hack-last-call` 0.885), so it is probably an EXPLOIT fork. I could not verify this.

### 6. crystalbaby/biohub-geometric-fusion | no LB | TRIVIAL-FORK
- Diff vs amanatar: 20 lines removed, 42 added, all in code cell 1 (the config guard). Variables were renamed (`_guard_*` to `_cfg_*`) and the messages re-worded.
- Every other cell, all env vars and all attached datasets are identical to amanatar.
- amanatar's content is HF plus the following, all already rejected by us (replay +0.0003, conflicts with readmit):
  - `BIOHUB_PPSWEEP_EXTENDED=1`
  - `BIOHUB_PPSWEEP_PREFIX_GUARD=1`
  - `prune_weak_leaf_nodes` (inert at `BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB=0.0`)
  - `division_recoverability_audit`

### 7. ghazarosbarseghyan91/biohub-dae-alpha-0-15 | LB 0.946 | NEW-MODEL (weak, noisy evidence)
**Core method.** A small 3D conv denoising autoencoder is trained from scratch on each test video and blended into the image before the frozen UNet+transformer sees it. Everything else is an older harmonic-fusion profile that the author calls "verified 0.940".

**What the DAE is applied to.** The normalised input windows `imgs = ((frames - q_low)/(q_high - q_low)).clamp(0)`, inside `predict_video` just before `model.encode(imgs)`. The same `imgs` tensor then feeds the secondary seed and all 8 detection-TTA views, so both seeds see the blended input.
- Architecture: `Conv3d(1,8,3)-ReLU-Conv3d(8,4,3)-ReLU-Conv3d(4,8,3)-ReLU-Conv3d(8,1,3)`, no down-sampling.
- Training (per video):
  - `BIOHUB_DAE_TRAIN_FRAMES=8` uniformly spaced frames
  - `BIOHUB_DAE_TRAIN_STEPS=30` Adam steps at lr 1e-3
  - input = clean + N(0, `BIOHUB_DAE_NOISE_SIGMA=0.05`), MSE loss to the clean frame
- There is no `torch.manual_seed` anywhere, and our local `predict_unet_transformer.py` has no seeding either. The DAE is therefore a different random filter on every run.

**What alpha is.** The blend weight in `imgs <- imgs + alpha*(DAE(imgs) - imgs)`, set by `BIOHUB_DAE_ALPHA=0.15`. After 30 steps from random init the DAE is barely trained, so the blend is effectively a mild contrast reduction plus a smoothed component.

**Other deltas vs HF/x138.** These are a regression to an older profile:

| setting | this notebook | HF / x138 |
|---|---|---|
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.30 | 0.15 |
| `GAP_CLOSE_UM` | 5.8 | 5.0 |
| `MOTION_RELINK_LEARNED_BONUS` | 1.35 | 1.0 |
| `MOTION_RELINK_VELOCITY_WEIGHT` | 0.75 | not set |
| primary/secondary edge-feature TTA | removed | on |
| DeepCenter TTA | removed | on |
| PPSWEEP | removed | on |
| validator movies per prefix | 2 | 4 |

**Evidence (author, LB).** Only alpha changes between versions:

| alpha | LB |
|---|---:|
| 0.10 | 0.942 |
| 0.15 | 0.946 |
| 0.17 | 0.935 (pulled sibling `biohub-dae-alpha-0-17`, alpha is the only diff) |
| 0.20 | 0.938 (stated in the 0.17 notebook) |

Losing 0.011 for a 0.02 change in alpha, on an unseeded stochastic filter, is most consistent with run-to-run noise around a ~0.940 base. Even the best draw is below HF's 0.947.

**Relevance to x138.** It would also change the frozen features the V1284 head reads, so the head might need re-capture and re-training.

### 8. ghazarosghazaros/biohub-dae-self-distill-repeat | LB 0.937 | NEW-MODEL (negative)
**Core method.** Notebook #7 (DAE alpha 0.15) plus a label-free "peak-EM self-distill" of the detection logits. The name is misleading: no student model is trained.

**How the self-distillation works** (`_biohub_self_distill_logits`). It is applied per frame to the final `det_logits`, after 8-view TTA and the dual-seed blend. Each round:
1. Find 3x3x3 local maxima with value >= `cfg.det_threshold`. This compares a logit to 0.965, while `_detect_cells_pooled` treats 0.965 as a sigmoid probability, so the effective cut is roughly p >= 0.72.
2. Keep peaks above the 40th percentile of peak values (`BIOHUB_SELF_DISTILL_KEEP_Q=0.40`).
3. Rasterise those peaks and blur with a separable [1,2,1]/4 kernel in z, y and x.
4. Rescale so the maximum equals the logit maximum.
5. Blend: `det <- 0.9*det + 0.1*blur` (`BIOHUB_SELF_DISTILL_ALPHA=0.10`).

This runs for `BIOHUB_SELF_DISTILL_ROUNDS=2` rounds. A frame fails open (keeps the original logits) if the detection count drops below 90%.

**Evidence.**
- LB 0.937 vs 0.946 for its parent (-0.009).
- The author's earlier edge-probability self-distill was flat (0.940).

### 9. lonnieqin/biohub-gap2-joint-node-budget | LB 0.915 | NEW-POSTPROCESS + NEW-MODEL (all flat in-lineage)
**Core method.** This is a different public lineage: yusuketogashi "Biohub 138 -> 154 -> 159B -> 162", clean LB 0.915.
- Its linker stack is not in x138: a 4-view JS-reliability edge-probability TTA and pilkwang's 22-feature local association ranker inside the Hungarian relink.
- It adds a three-frame forward-acceleration lookahead (from "Biohub 162").
- This notebook's own change: gap-2 recovery (`recover_strict_gap2`) and single-frame gap closing now draw from one shared synthetic-node budget.

**Mechanisms not in x138 or HF** (grep: zero hits in x138, HF, amanatar and our `src/`):

1. **Forward-acceleration lookahead**, in `motion_relink_edges`/`assign_pass`.
   - For candidate i->j it finds `r = min_k ||(x_k - x_j) - (x_j - x_i)||` over next-frame nodes k with step <= relaxed gate.
   - Relink cost gets `-= MAX_BONUS * max(0, 1 - r/MAX_ACCEL)`.
   - Settings: `BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD=1`, `..._MAX_ACCEL_UM=4.0`, `..._MAX_BONUS=0.20`.
   - x138's relink is strictly t->t+1 (neighbour-flow predictor) and never looks at t+2.
2. **pilkwang local association ranker** (public dataset `pilkwang/biohub-local-association-ranker-unet300-v1`, 22 features).
   - Features: edge_prob, in/out degrees, 7-um densities, raw/motion distance, motion gain, distance rank, candidate count, dz/dy/dx and their absolute values, velocity, frame sizes, t_norm.
   - It is used in `full_motion_assignment` mode: relink cost = motion + 0.05*raw - BONUS*(0.85*ranker + 0.15*edge_prob).
   - Settings: `BIOHUB_LOCAL_RANKER_FULL_WEIGHT=0.85`, `..._PRIMARY_RETAIN_WEIGHT=0.15`.
   - There is also an unused `low_margin_top2_rescue` mode: margin 0.35 um, minimum advantage 0.15, maximum bonus 0.20.
3. **JS-reliability log-pool edge TTA**: `BIOHUB_EDGE_TTA_MODE=js_reliability_log_pool`, `BIOHUB_EDGE_TTA_VIEWS=4`.
   - Views: identity, flip-x, flip-y, transpose. Each view gets its own UNet encode and its own harmonic forward/reverse softmax.
   - Each view is weighted per target by `1/(1 + JS/median JS)` to the 4-view mean, then combined with a log-opinion pool and rescaled to the identity view's logit centre/scale.
   - This is probability-level TTA. x138 averages 8-view UNet features and predicts once.
4. **Shared synthetic-node budget**: `BIOHUB_SHARED_SYNTHETIC_NODE_BUDGET_FRAC=0.05`, `_ABS=2000`. It replaces the independent caps of `close_single_frame_gaps` and `recover_strict_gap2` with one decreasing counter, aimed at the node-count term `1 - 0.1*(T_pred - T_true)/T_true`.

**Older or inferior settings.** These are not ideas worth taking:
- `DET_THRESHOLD` 0.96875
- ILP disappearance 1.5
- safe division 4.66/8.5/7.65 um
- DeepCenter off
- `SECONDARY_DETECTION_WEIGHT` 0.475 (swept publicly: 0.475/0.70/0.80/0.85)
- no edge-feature TTA

**Evidence.** Everything is flat on LB:
- 159B = 0.915
- 162 (+lookahead) = 0.915 (yusuketogashi, same lineage)
- this notebook (+gap2 shared budget) = 0.915

The lineage has no isolated measurement of the ranker or of JS-TTA.

### 10. busyaprime/biohub-what-one-link-node-and-division-are-worth | no LB | INACCESSIBLE
- 403, and not in the author's listing (only `biohub-0-942-lb-one-knob-past-the-public-line` remains).
- The title suggests an analysis of the metric's marginal values, not a method.

### 11. howonkang/biohub-0947-short5-prepp-r1 | LB 0.945 | NEW-POSTPROCESS (negative) + TRIED knob
**Core method.** HF (0.947) plus two changes:
1. `apply_short5_raw`, called in `write_test_submission` before `filter_output_graph`. It drops every weakly connected component with <= 5 nodes from the raw ILP graph before relink, gap closing and safe division run.
   - This is not `OUTPUT_MIN_TRACK_LEN`, which runs at the end and spares division components.
   - The validator and PPSWEEP never applied it, so its "proven" claim is not measured in-notebook.
2. `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` 0.25 -> 0.20.

**Evidence.**
- LB 0.945 vs HF 0.947.
- We already swept DeepCenter safe-div 0.20 in our harness (`reports/pp_replay/e2e_head_v1_ppknobs*`, `dcsafediv020`). It gave +0.0045 on all12, driven entirely by 44b6 divisions (1/0/3 -> 2/1/2 TP/FP/FN), and was unchanged on the 6bba proxy (0.95098 vs 0.95098; HANDOFF section 19).
- So the -0.002 is most likely SHORT5's. Removing fragments before relink or gap-fill can re-join them is plausibly harmful.

### 12. haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m | LB 0.944 | TRIED
- HF plus the following:
  - Relative-rank / mutual-best logit bonus before the edge softmax: beta = 0.12, `+beta` for column-best, `+0.5*beta` for row-best, `+0.5*beta` for mutual.
  - `DENSITY_GROUP_OVERRIDES` (low/middle/high by average nodes per frame, below 120 / below 400 / above) with per-group relink tight/relaxed/velocity/bonus.
  - DivNet 3D-CNN veto on output divisions (`giorgosi/biohub-divnet-v2`, p < 0.50).
  - `MOTION_RELINK_TIGHT_UM=5.5`, validator off.
  - CUDNN workspace cap and threaded post-processing (speed only).
- Every mechanism is in our tried list: mutual-best, DivNet gate, density-adaptive association.
- The title claims 0.951+; the LB is 0.944.

### 13. kunaldesale2408/biohub-cell-tracking | LB 0.942 | TRIED (division-loosening family)
**Core method.** amanatar geometric fusion plus `add_second_pass_division_repair` (`BIOHUB_SECOND_PASS_DIV_REPAIR=1`). It runs after the first-pass safe division and before `output_division_geometry_filter`.
- Eligible parents have exactly one child within min(existing-child cap, `DIV_PARENT_MAX_UM`).
- It adds an orphan (in-degree 0) second daughter that meets:
  - parent distance <= min(`SAFE_DIV_MAX_UM`, `DIV_PARENT_MAX_UM`)
  - sister distance <= min(`SAFE_DIV_SISTER_MAX_UM`, `DIV_SISTER_MAX_UM`)
- Scoring is `d_parent + 0.3*d_sister + 2*asymmetry`, with at most 2 per frame and a global cap of 1% of edges (x138's cap is 0.375%).
- It has no mutual-NN, divergence, symmetry-tau or DeepCenter gate.
- It attaches `anvithpothula/biohub-v1284-head-s075`, but the code never references it, so the head is inert.

**Evidence.**
- LB 0.942 vs amanatar 0.948 (-0.006).
- This matches our finding that division candidates beyond x138's rule have precision of 6-23%, well below the roughly 35% needed.

### 14. evgendvorkin/biohub-0-947-lb-proxy-score-0-9490 | LB 0.947 | TRIVIAL-FORK
**Code.** Identical to HF (flexonafft 0.947): the only changes are the removed module docstring and blank lines. Same env vars, same datasets. The only added content is a Russian/English markdown walkthrough and an evolution table from v10 to v31.

**How PROXY_SCORE is computed.** It is HF's in-notebook validator, code cells 7-9:
1. Take the train stems not present in `TEST_DIR`, grouped by embryo prefix (44b6 / 6bba).
2. Rank stems whose GT `.geff` contains a division first, then alphabetically, and keep `BIOHUB_VALIDATOR_N_PER_TYPE=4` per prefix. These are exactly our harness's `val8` stems.
3. Re-run the same inference on them, then `filter_output_graph`, with `TEST_DIR` temporarily pointed at `TRAIN_DIR` so DeepCenter reads the train frames.
4. Match predicted nodes to GT per frame with Hungarian matching within 7 um (physical scale z 1.625, y/x 0.40625).
5. Per movie, compute adjusted edge Jaccard = `J * (1 - 0.1*(T_pred - T_true)/T_true)`, with `T_true` = `estimated_number_of_nodes` from the GT geff.
6. `aggregate_official`: the mean of per-movie adjusted Jaccard weighted by (TP+FP+FN), plus 0.1 x the pooled division Jaccard (division TP/FP/FN summed over the 8 movies).

**Reliability as an LB predictor.**
- 0.9490 is the proxy of their v29 (LB 0.944). The code is v31 (proxy 0.9511, LB 0.947).
- In-sample: x138 itself notes that the pilkwang models saw these train movies.
- Across their 12-version table, proxy vs LB has Spearman 0.81, but only 7 of 11 consecutive steps move in the same direction. At v31's division TP/FP/FN of 3/1/9, one division moves the proxy by about 0.0077.
- HANDOFF section 18 already found that our 6bba-subgroup proxy predicts LB much better (Pearson 0.994) than any all-movie proxy.

---

## Headroom check (why association-level ideas are capped)

The C012 configuration on the local harness (`reports/pp_replay/e2e_head_v1_ppknobs_summary.csv`, `c012_as_is`):

| subset | recovered GT edges | fragmented | lost to detection | wrong-association |
|---|---:|---:|---:|---:|
| all12 | 7,610 | 189 | 79 | 1 |
| 6bba | 6,120 | 129 | 61 | 1 |

- **Wrong associations are about zero.** Linking between matched cells is essentially solved, and on GT positions our own geometry ranking already has nearest-parent top-1 of 0.988 (6bba) / 0.997 (44b6).
- **Our own learned geometry-swap ranker lost on the 44b6 holdout** (net -6; `reports/research_20260914/fulltrain_geometry_swap_ranker.json`).
- **The only edge mass left for a better linker is "fragmented" edges.** That is about 2% of 6bba GT edges (129 of roughly 6,300).
- **Upper bound for the lookahead, ranker and JS-TTA:** even fixing half of them would be about +0.01 on 6bba adjusted Jaccard, which is roughly +0.003 LB via HANDOFF's `LB = 0.298 x local_6bba + 0.668`.
- **Most local deltas will not be decidable.** A local 6bba difference below ~0.007 cannot be resolved against the LB.

## Ranked untried ideas that could stack on x138 (0.953)

| rank | idea (source) | type | effort | local harness? | prior evidence / expected |
|---:|---|---|---|---|---|
| 1 | Forward-acceleration (t+2) continuation bonus in x138's relink cost (lonnieqin / yusuketogashi "Biohub 162") | post-processing | ~2 h to patch a C012-derived notebook's `assign_pass` behind new globals (e.g. `FORWARD_LOOKAHEAD_MAX_BONUS`, `_MAX_ACCEL_UM`); 6 variants x ~1.5 min | YES (pure post-processing on cached e2e graphs; rank on 6bba) | Flat in its own lineage (0.915 -> 0.915); capped by 129 fragmented 6bba edges; expect <= +0.001 LB. Variant: compute the residual against x138's flow-predicted step instead of the raw step. Do a 30-min triage first: classify the fragmented edges (track end vs wrong target) to see whether target choice causes any of them. |
| 2 | pilkwang 22-feature local association ranker as relink evidence (0.85 ranker + 0.15 edge_prob), or its `low_margin_top2_rescue` mode (lonnieqin / yusuketogashi) | post-processing with a learned model | 4-5 h (port loader and features from lonnieqin cell 3 and `_ranker_context`, merge with the flow relink) | YES once the public dataset `pilkwang/biohub-local-association-ranker-unet300-v1` is downloaded (not done: needs your OK) | No isolated LB evidence (0.915 lineage). Likely trained on train movies, so optimistic on our 12 movies. Our geometry ranker failed out-of-fold. Expect about 0. |
| 3 | JS-reliability log-opinion-pool edge-probability TTA (4 views), replacing or on top of x138's feature TTA (lonnieqin "Biohub 154") | inference | ~4 h to port into the C012 `tracking_repo` patch chain, plus 2 re-inferences x ~25 min (replace / stack) | Only after re-inference (`src/run_kaggle_predict_local.py`, then harness) | No isolated evidence. Changes edge probabilities feeding the ILP (fragmentation), not the near-zero wrong associations. Small. |
| 4 | DAE input pre-filter, seeded, alpha 0.10-0.15 (ghazarosbarseghyan91) | inference | 1 h dev + at least 3 seeds x 2 alphas x ~25 min; may need V1284 re-capture and head retrain (+4 h) | Only after re-inference | LB non-monotone in alpha (0.942 / 0.946 / 0.935 / 0.938) on a 0.940 base without edge-feature TTA; likely noise. Low priority. |
| 5 | SHORT5: drop <=5-node raw components before post-processing (howonkang) | post-processing | 0.5 h | YES | LB -0.002 vs HF, and its other half (DeepCenter 0.20) is neutral on 6bba locally. Expect a loss; only a quick confirmation run. |

Not recommended, each with negative or flat LB evidence or already covered:
- peak self-distill of detection logits (-0.009)
- ungated second-pass division repair (-0.006; division-precision problem already quantified)
- shared synthetic-node budget (flat 0.915 -> 0.915; the node-count term was closed on 09-24)
- the haideptry items (tried)
- the six inaccessible notebooks: no code. #1 is probably a det-threshold knob; #5 is probably an exploit.

## Files
- Pulled sources: `state/notebook_radar/pulled/{crystalbaby__biohub-geometric-fusion, ghazarosbarseghyan91__biohub-dae-alpha-0-15, ghazarosbarseghyan91__biohub-dae-alpha-0-17, ghazarosghazaros__biohub-dae-self-distill-repeat, lonnieqin__biohub-gap2-joint-node-budget, howonkang__biohub-0947-short5-prepp-r1, haideptry__biohub-0-951-sota-deepcenter-fast-ilp-19m, kunaldesale2408__biohub-cell-tracking, evgendvorkin__biohub-0-947-lb-proxy-score-0-9490}/`
- Code locations cited:
  - lookahead: lonnieqin code cell 6 (`forward_acceleration_lookahead` / `_bonus`) and cell 8 (`motion_relink_edges`)
  - ranker: cell 3
  - JS-TTA: cell 7
  - shared budget: cell 8 (`close_single_frame_gaps`, `recover_strict_gap2`, `filter_output_graph`)
