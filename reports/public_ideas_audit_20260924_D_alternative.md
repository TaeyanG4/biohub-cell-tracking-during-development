# Public notebook audit D: alternative trackers, classical methods, older lineages, training kernels (2026-09-24)

Scope: the 14 assigned entries (18 kernels), mostly outside the x138 fork family. Each source was pulled with `kaggle kernels pull` into `state/notebook_radar/pulled/<author>__<slug>/`, and the markdown and code cells were read. No kernels were pushed, nothing was submitted, and no datasets or weights were downloaded. Reference stack: `state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb` (public 0.953). Already-tried list: `reports/tried_methods_20260924.md`.

Classes: TRIED / NEW-POSTPROCESS / NEW-MODEL / NEW-DATA / EXPLOIT / TRIVIAL-FORK / WEAK.

## Bottom line

- Nothing here matches x138. The best entry is josephadamski V1329 at 0.942, and it comes from the x138 lineage itself. Every tracker that doesn't use the pilkwang models scores between 0.10 and 0.917.
- Three components are missing from x138 and cheap enough to test locally:
  1. **V1057 ILP-edge reconciliation** (josephadamski).
  2. **pilkwang's public 22-feature local-association ranker** inside the Hungarian relink cost (yusuketogashi).
  3. **JS-reliability log-pool edge TTA** (yusuketogashi; an inference-level change).

  None of the three has a public LB change measured in isolation. Treat them as low-prior probes, not as expected gains.
- Most of what this batch offers is **negative evidence**, and it supports decisions we already made:
  - Every public detector replacement or fine-tune scores below the pilkwang weights on the LB:
    - josephadamski adapted centered-window detector: 0.942, even with a V1284 head
    - yongjilyu own hi-res ensemble: 0.888
    - tangai1 all-199 "native SSL" model: 0.714 LB, 0.728 on the visible 4 even though they are in its training set

    This matches our own C008 (0.926).
  - Removing all forks costs about 0.03 (alfonso: 0.942 -> 0.907).
  - Zero-shot Trackastra is not a better linker (0.616).
- No notebook in this batch uses pseudo-labels or external zebrafish data. The only external asset is Trackastra's Cell Tracking Challenge (CTC) checkpoint.

## Summary table

| # | ref | public LB | class | one line |
|---|---|---:|---|---|
| 1 | jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans | 0.616 | WEAK | DoG detection + watershed masks + zero-shot Trackastra ("ctc") in greedy mode |
| 2 | yusuketogashi/no-hack-biohub-cell-another-approch-3rd | 0.915 | NEW-POSTPROCESS (+ inference TTA) | older pilkwang dual-seed line + 22-feature association ranker in the relink cost, JS-reliability edge TTA, forward-acceleration lookahead |
| 3 | reyhanksatria/graph-patches-for-cell-tracking-0-917-lb | 0.917 | TRIED | xiaoleilian 3-model UNet + Hungarian; its post-link patches are x138's ancestors |
| 4 | praxel/biohub-0-902-motion-division-calibration | 0.901 | TRIED | single-seed pilkwang 50ep + D4 detection TTA; every function exists in x138 |
| 5a | tangai1/biohub-b02-all199-production-20260915-packed | none | WEAK | contains no method code: a hard-coded visible-4 CSV. It scores 0.728 locally on the visible 4. |
| 5b | tangai1/biohub-b02-dynamic-production-20260915 | 0.714 | WEAK (failed NEW-DATA) | loader for a private all-199 "native SSL" runtime |
| 6 | yongjilyu/biohub-ct-sp402 | 0.888 | INACCESSIBLE (403) | the same author's biohub-ct-sub (0.880) was read instead: own 3-model hi-res ensemble + ILP, val 0.877, WEAK |
| 7 | binasalama/biohub-learned-unet-transformer-ilp-gap-recovery | 0.887 | TRIVIAL-FORK | single-seed pilkwang 50ep with retuned constants |
| 8 | alfonso1799/biohub-top-3-push-v50-streamlined-sota | 0.907 | WEAK (negative evidence) | josephadamski V1329 with the head off + every fork linearised + at most 2 hand-gated divisions |
| 9 | josephadamski91/biohub-v1329-v1327-original-guard | 0.942 | NEW-POSTPROCESS (V1057) + NEW-MODEL (adapted detector, negative) | x138-lineage code with a V1284 head, + a fine-tuned detector using a 3-frame window centred on t + ILP-edge reconciliation |
| 10 | denpugovkin/why-this-tracker-parks-an-ancestor-outside-the-vol | 0.885 | EXPLOIT | ancestor "stem" node parked outside the volume |
| 11 | sujalsuyash/liquidtracker-core-biohub-lnn | none | WEAK | toy liquid neural network trained on GT coordinates; no inference |
| 12 | fabriciodasilva/biohub-dodecatiad-cell-tracking | 0.827 | WEAK | CPU-only DoG + Hungarian + gap close |
| 13 | junaid512 0.471 / sarveshchhetri 0.432 / tobimichigan 0.100 | | WEAK | classical baselines and a candidate classifier |
| 14 | seshurajup/lb-0-857-best-rule-base-v14 | 0.857 | TRIED (ancestor) | rule-based DoG + two-pass Hungarian; x138's post-processing constants come from here |

## Per-notebook notes

### 1. jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans (0.616; current version 0.601)
- **Method.**
  - Detection: multi-scale DoG (sigma 1.0/1.8/3.0 isotropic voxels after an XY/4 block mean), intensity centre-of-mass refinement, 4 um physical NMS.
  - Masks: marker watershed seeded at the peaks (or fixed balls).
  - Linking: `Trackastra.from_pretrained("ctc")` in greedy mode, then isolated nodes are pruned.
  - Validation: only a proxy (0.5 node-F1 + 0.4 edge-F1 + 0.1), not the official metric.
  - The closing markdown proposes fine-tuning Trackastra on the train geffs; this was never done.
- **Distinct from x138:**
  - classical detector
  - external pretrained linker (Trackastra, trained on CTC 2D+3D data), which handles divisions natively
  - no competition-specific training
- **Evidence:** LB 0.616. The same author's DoG-detection EDA notebook (not audited) is listed at 0.728, so zero-shot Trackastra did not improve on his earlier pipeline.
- **Class: WEAK.** HANDOFF already deprioritised zero-shot Trackastra, and this result agrees. Only a speculative follow-up remains (idea 7 in the ranked list).

### 2. yusuketogashi/no-hack-biohub-cell-another-approch-3rd (0.915)
- **Method ("Biohub 162"): the older pilkwang lineage.**
  - Models: support-pack 50ep primary + seed314159 secondary.
  - Detection: a "near-balanced" logit mix with secondary weight 0.475 (x138 uses 0.80), threshold 0.96875.
  - Edge TTA over 4 views with Jensen-Shannon reliability log-pooling ("Biohub 154").
  - ILP: edge -1, appearance 0, disappearance 1.5, division 1.0.
  - Hungarian motion relink: tight 6 / relaxed 10 um, velocity weight 0.5. Its cost is
    `motion_dist + 0.05*raw_dist - 1.0*(0.85*ranker_prob + 0.15*transformer_prob)`.
    `ranker_prob` comes from a public 22-feature MLP (`pilkwang/biohub-local-association-ranker-unet300-v1`, 20 KB checkpoint + `history.json` / `model_info.json`).
  - Forward-acceleration lookahead: the cost drops by up to 0.20 when the target has a continuation at t+2 whose velocity change is below 4 um.
  - Rest of the chain: density-adaptive gap closing, tight safe division (4.66 / 8.5 / 7.65 um), no DeepCenter, min track length 6, linefit smoothing.
- **Ranker features.** All can be computed from the ILP graph and node positions, so our harness can produce them:
  - ILP edge probability and a has-edge flag
  - in/out degree of source and target
  - number of neighbours within 7 um, frame node counts
  - distance rank and candidate count inside the gate
  - dz/dy/dx, raw and motion-predicted distance, motion gain, velocity
  - the target's best outgoing probability, source-has-previous, target-has-next, t/T
- **Distinct from x138.** x138's relink cost is `motion_dist(flow prior) - 1.0*ILP_prob` (`BIOHUB_MOTION_RELINK_FLOW_RAW_COST = 0`). It has:
  - no ranker
  - no lookahead
  - edge TTA that averages UNet features across views, not a prediction-level JS pool
- **Evidence.**
  - Parent notebook 159B (ranker + JS-TTA): public 0.915. Notebook 162 adds the lookahead and stays at 0.915.
  - No local CV of any component; the audit cell only quotes "Biohub 138 fixed-4 = 0.8879".
  - Neither the ranker nor the JS-TTA was ever scored on its own.
  - The ranker was trained on the edge probabilities of a different model ("unet300"), so its probability input is shifted when fed x138's fused probabilities.
- **Class: NEW-POSTPROCESS** (ranker, lookahead) plus an inference-level change (JS-TTA). These are ideas 2, 3 and 4 below.

### 3. reyhanksatria/graph-patches-for-cell-tracking-0-917-lb (0.917)
- **Method.**
  - Detection: xiaoleilian UNet3D models (2 base models, one with top-hat preprocessing, + a top-hat b32 model), each with 4-way flip TTA, heatmaps averaged; seed threshold 0.15; centre-of-mass refinement; 4 um NMS.
  - Linking: Hungarian with a velocity-predicted position (tight 6 / loose 10 um) plus a detection-confidence similarity cost `2*|logit(s_i) - logit(s_j)|`.
  - Snap-only 1-frame gap closing: it only uses unused low-score candidates (0.05-0.15).
  - Short-track filter 6, linefit 0.8.
  - Safe division: parent 12 / sister 15 / child 10 um, divergence 2.25, frame cap 0.0076, global cap 0.00375, mutual-nearest-neighbour check.
- **Distinct from x138:** different detector family, no learned edge model, no ILP, and the confidence-similarity cost term.
- **Evidence:** LB 0.917. The author then moved to the 0.947 pilkwang/harmonic lineage (our B0).
- **Class: TRIED.** x138 already has the same gap snapping (`GAPFILL_ALLOW_SYNTHETIC=0`), identical division caps and divergence (0.0076 / 0.00375 / 2.25; the envelope is wider here, a rejected family), the short-track filter and linefit.
  - Only leftover: the confidence-similarity cost term (idea 6, low prior). The cached ILP geffs don't store node detection scores, so testing it needs the low-detection dump or re-inference.

### 4. praxel/biohub-0-902-motion-division-calibration (0.901)
- **Method.**
  - Single-seed pilkwang 50ep + ILP + motion relink (learned-probability bonus 1.0) + gap close + min track length 6 + tight safe division (4.66 / 8.5).
  - A predictor patch adds 8-view D4 detection TTA (flips + rot90 + transpose).
  - The DeepCenter branch is removed; the author found it inert.
- **Distinct:** none. The D4 patch is verbatim in x138 cell 4 ("400ep spatial D4-style"), and every function exists in x138.
- **Evidence:** LB 0.902 (submission 54633411, July).
- **Class: TRIED.**

### 5. tangai1 (packed: no score; dynamic: 0.714)
- **5a (packed script).** Contains no method: it decodes an LZMA/base64 payload into a fixed `submission.csv` for the four visible test movies, checked against a SHA-256. It cannot score on the hidden test.
  - I decoded it into the scratchpad (SHA matched) and scored it with `src/evaluate_local.py --gt-dir data/visible_gt/train`:

    | metric | value |
    |---|---:|
    | total | 0.7280 |
    | adjusted edge Jaccard | 0.7196 (raw 0.7387) |
    | divisions tp / fp / fn | 1 / 9 / 2 |
    | node recall | 0.971 |
    | predicted nodes | 177,758 |

  - Adjusted edge Jaccard per movie: 44b6_0113de3b 0.864, 44b6_0b24845f 0.617, 6bba_05b6850b 0.674, 6bba_05db0fb1 0.752.
  - Same four movies for comparison: C011 (x138 without head) 0.9056, C012 0.9252. The B02 model was trained on all 199 movies, so these four are in its training set.
- **5b (dynamic script).** A 9 KB loader that mounts a private runtime:
  - code: `experiments/20260914_v6_b02_biohub_native_ssl/production_infer.py`, plus a "B01 long-window temporal" module and "C46 production"
  - weights: `b02_{primary,secondary}_production_all199.pth`

  It then audits the CSV. The method itself is not public.
- **Distinct:** own detector/linker trained on all 199 movies, with self-supervised ("native SSL") pretraining and a long temporal window.
- **Evidence:** LB 0.714; 0.728 on the visible 4 (in-sample). The author's older c35 fallback-detection kernel scored 0.934 (not audited).
- **Class: WEAK** (a failed NEW-DATA attempt). Nothing to transplant.

### 6. yongjilyu/biohub-ct-sp402 (0.888): INACCESSIBLE
- The pull returned 403 twice, and the kernel is no longer in the author's public list. Radar: best 0.888, latest 0.863, last run 2026-09-20.
- **Proxy: `yongjilyu/biohub-ct-sub`** (0.880, 2026-09-18), pulled to `state/notebook_radar/pulled/yongjilyu__biohub-ct-sub/`.
  - Code comes from a private pack `yongjilyu/biohub-ct-inference-pack` (own `celltrack` package).
  - Input is downsampled only 2x in XY (0.8125 um vs pilkwang's 1.625 um).
  - 3-model "hires" ensemble with logit averaging.
  - Detection threshold 0.881 (= sigmoid 2), edge threshold 0.40, max link 8 um.
  - ILP appearance 1.0, disappearance 1.0; no graph repair.
  - Docstring: "Val 0.8766 (19-video fold)", about 4.5 h for about 199 hidden movies.
- **Class: WEAK / NEW-MODEL,** not transplantable: code and weights are private, and the score is below pilkwang.

### 7. binasalama/biohub-learned-unet-transformer-ilp-gap-recovery (0.887)
- **Method.** Single-seed pilkwang 50ep (TemporalUNet3D + node cross-attention transformer) + ILP + deterministic repair, with retuned constants:

  | constant | value |
  |---|---|
  | detection threshold | 0.985 |
  | velocity weight lambda_v | 0.52 |
  | learned bonus beta | 0.78 |
  | relink tight / relaxed | 6.2 / 10.4 um |
  | gap gate | 5.75 um |
  | gap2 | 9.7 / 4.05 um, cap 0.0032 |
  | linefit weight | 0.76 |

  The markdown restates pilkwang's training objective: weighted BCE for detection, and a focal edge loss with softmax over parents, restricted to rows/columns that touch an annotated edge.
- **Distinct:** none; all 45 functions exist in x138.
- **Class: TRIVIAL-FORK.**

### 8. alfonso1799/biohub-top-3-push-v50-streamlined-sota (0.907)
- **Cell 1** runs a gzip/base64-embedded copy of josephadamski V1329 (the diff shows only path changes, `V1284_MODE='zero'`, and hash guards removed).
- **Cell 2** "linearises" every fork (keeps the nearer child). Then, only on movies with at least 30,000 nodes, it adds at most two divisions per movie through a strict rule. All of these must hold:
  - parent-child distance <= 8.5 um
  - sister distance 8.5-13.5 um
  - D2 is an orphan and the mutual nearest neighbour of D1
  - angle between P->D1 and P->D2 >= 140 deg
  - parent within 3.2 um of the daughters' midpoint
  - daughters separated by >= 9.6 um in xy and <= 4 um in z
  - both daughter tracks >= 12 frames
  - divergence >= 1.2 um at t+2
  - 20-pixel border margin
- **Evidence.**
  - The title claims 0.9605. That is a visible-4 number tuned on one visible event (P=20025 -> D=20865 in 6bba_05db0fb1).
  - LB is 0.907, against 0.942 for the unmodified V1329.
  - The head is worth about 0.007 in x138, so the fork purge costs roughly 0.03 on the hidden test.
- **Class: WEAK** (overfit to the visible 4).
  - Negative evidence: forks from the ILP and rule stages carry real division TPs on the hidden set; do not purge them.
  - The strict rule is a hand-written version of the C016 ultra-conservative top-k, which we measured at about +0.0007 local per movie.

### 9. josephadamski91/biohub-v1329-v1327-original-guard (0.942)
- **Base.** Reyhan's 0.946 harmonic-fusion code:
  - dual seed, bidirectional harmonic fusion 0.15
  - edge-feature TTA
  - DeepCenter gates
  - 90 % candidate-retention guard

  It also has the V1284 frozen-feature coordinate head in 'candidate' mode (trial head sha 5c1f83c4...). The module text is 0.90 similar to x138's (same origin), so the head is **TRIED** (C012).
- **(a) V1327 adapted detector.**
  - Training: a copy of the primary model fine-tuned for "256 released-data centered-W3 updates", i.e. a 3-frame window t-1, t, t+1, with the transformer and BN frozen.
  - Use: its centre logits **replace** the primary detection map frame by frame, after mean/std calibration (scale clamp 0.5-2). Edge tensors are untouched.
  - V1329 measures the retention guard against the untouched primary D4 map.
- **(b) V1057 reconciliation**, run after all post-processing:
  - Every ILP edge with p >= 0.30 that is not in the final graph is put back. In practice that means all ILP edges, since candidates need p > 0.48 to enter the ILP.
  - Conflicts are resolved by descending probability, with each source and target used at most once.
  - It never touches a node with two children or a daughter of one.
  - The relinked edges it displaces are dropped. No nodes are added or removed.
- **Distinct from x138: (a) and (b); x138 contains neither** (no `V1327`, `adapted_detector` or `V1057` anywhere). x138's `filter_output_graph` replaces the whole ILP edge set with the Hungarian relink output (`edges = motion_edges`); ILP edges only enter through the `-1.0*p` cost bonus.
- **Evidence.**
  - V1327 and V1329 both score 0.942.
  - The attribution string in the notebook traces the Reyhan chain: 0.933 fixed-90 dual-seed -> 0.934 harmonic -> 0.939 wider divisions -> 0.941 repair-threshold adaptation -> 0.946 edge-feature TTA.
  - No public score separates V1057 from the adapted detector. With a head, the combined result is 0.011 below x138 and 0.004 below the head-less lineage, so at least one extra hurts.
  - The detector swap is the likelier culprit: it changes every node, and every detector replacement so far has lost (C008 0.926).
- **Class:** the adapted detector is **NEW-MODEL with negative evidence; do not transplant**. V1057 is **NEW-POSTPROCESS, untested and cheap to test** (idea 1).

### 10. denpugovkin/why-this-tracker-parks-an-ancestor-outside-the-vol (0.885)
- **EXPLOIT:** every component root is connected to an ancestor at t=-1000, (-10000, -10000, -10000), plus five synthetic mitoses, so that all tracks share one lineage for the division term.
- Underlying tracker:
  - single-seed pilkwang 50ep
  - ILP on strong edges (p >= 0.50) plus each target's top-2 parents (p >= 0.25), 10 um cap
  - 1-frame gap close
  - short-track filter: minimum length 4, but keeps division components and any component that starts at t <= 2 or ends at t >= T-3
- The author's "biohub-0-885-without-the-fake-lineage-hub" (radar) scores the same 0.885, so the stem gains nothing on the patched metric.
- Only leftover: the time-boundary exemption in the short-track filter; x138 has none (idea 5).

### 11. sujalsuyash/liquidtracker-core-biohub-lnn (no score)
- A liquid time-constant RNN edge scorer trained for 8 epochs on GT node coordinates only. The coordinates are z-scored per frame, which throws away absolute position.
- No detector, no inference, no submission. **WEAK.**

### 12. fabriciodasilva/biohub-dodecatiad-cell-tracking (0.827)
- CPU only:
  - multi-scale DoG (1.5/4.0 and 2.2/5.5 um) at XY/4, with local-maximum picking and background-subtracted centroid refinement
  - distance-only Hungarian linking (8 um), no divisions
  - 1-frame gap close (6 um), isolated-node pruning
- This is roughly the ceiling for a classical pipeline. **WEAK.**

### 13. Newest (skimmed)
- junaid512 0.471: classical DoG + ILP/Hungarian with an automatic fallback to trained weights that were never attached. Its markdown correctly says detection recall is the ceiling.
- sarveshchhetri 0.432: Gaussian-filter peaks + Hungarian + a nearest-parent division pass.
- tobimichigan 0.100: DoG/watershed candidates + an SE-attention 3D CNN / RandomForest soft-vote candidate classifier + Hungarian.
- All **WEAK.**

### 14. seshurajup/lb-0-857-best-rule-base-v14 (0.857)
- isakatsuyoshi's rule-based baseline with pilkwang's config:
  - DoG (1.5/4.0 + 2.2/5.5 um) at XY/2, centroid refinement
  - two-pass Hungarian with velocity (tight 6 / loose 8)
  - 1-frame gap close, isolated-node pruning, minimum track length 4
  - linefit 0.8 / window 2, gap2 recovery 10.2 / 4.4 um
  - no divisions
- x138's `GAP2_MAX_TOTAL_UM` / `GAP2_MAX_STEP_UM` (10.2 / 4.4) and linefit (0.8 / 2) are identical: x138's post-processing chain started here.
- Last run 2026-07-03, before the 17 July metric patch. The "rule-based tracker was 7th" claim refers to that early leaderboard.
- **TRIED (ancestor).**

## Transplantable untried ideas (ranked)

**Check first (about 1 h, small script on a harness replay of C012).** On the 12 held-out movies, count:
- final edges that differ from the ILP edges
- edge FPs whose two endpoints are both GT-matched (wrong-partner errors)

Ideas 1, 2, 4 and 6 only change which partner a node is linked to. They cannot recover missed detections, so their ceiling is set by these two counts. If wrong-partner errors are a small share of the 6bba edge errors, skip them.

| rank | idea | source | change to x138 | effort | testability | evidence / prior |
|---:|---|---|---|---|---|---|
| 1 | V1057 ILP-edge reconciliation | josephadamski V1329 | after `filter_output_graph`, put back the ILP edges (p >= tau) that the relink displaced; unique source/target; forks and daughters untouched; sweep tau = 0.30 (published) / 0.7 / 0.9 / 0.95 | 1.5-2 h (graft modelled on `install_ilp_fork_reinjection`; `load_raw_graph` already keeps `edge_prob`) | high: post-processing only; cached ILP graphs of the held-out 12 + confirm 10; rank by the 6bba proxy | shipped in a 0.942 notebook, but mixed with the detector swap; no isolated number. Prior low-moderate: the relink (plus its flow prior, +0.0077 on replay) is x138's main edge gain, and the reconciliation partly reverses it. High tau is the plausible regime. |
| 2 | 22-feature local-association ranker in the relink cost | yusuketogashi 159B/162 (`pilkwang/biohub-local-association-ranker-unet300-v1`) | `cost = motion - 1.0*(0.85*ranker + 0.15*p)` in `motion_relink_edges.assign_pass`, keeping x138's flow-prior prediction | 3-4 h + a 19 KB dataset download (team decision; I did not download it) | high: harness; every feature comes from the ILP graph and positions | lineage peaked at 0.915, no isolated delta; ranker trained on another model's probabilities. Variant: retrain the same MLP on relink candidates from our ~85 local e2e movies (6-8 h), with the same in-sample edge-probability caveat that limited C016 |
| 3 | JS-reliability log-pool edge TTA | yusuketogashi (Biohub 154) | replace feature-averaged edge TTA: per-view harmonic probabilities (identity / flip_x / flip_y / transpose), per-target view weights from JS distance to the 4-view consensus, log opinion pool, rescaled to the identity view's logit statistics | 4-6 h + about 40 min GPU for 12 movies | medium: `src/run_kaggle_predict_local.py` (reproduces T4), then the harness | edge TTA itself was worth +0.005 LB in the Reyhan chain (0.941 -> 0.946); pooled vs averaged never compared. Prior low-moderate; inference changes carry fidelity risk |
| 4 | forward-acceleration lookahead bonus in the relink | yusuketogashi 162 | subtract `0.2*max(0, 1 - accel_residual/4 um)` when the target has a t+2 continuation | 1-2 h | high: harness | its own LB stayed at 0.915; overlaps x138's neighbourhood-flow prior. Low |
| 5 | short-track filter that exempts time-boundary fragments | denpugovkin (legitimate part) | keep components shorter than 6 nodes that start at t <= 2 or end at t >= T-3 | about 1 h | high: harness | no evidence; short tracks match GT at 0.66 % (node-budget analysis), so likely noise. Low |
| 6 | detection-confidence similarity term in the relink cost | reyhanksatria 0.917 | `+ w*|logit(s_i) - logit(s_j)|` | 2-3 h (needs per-node detection scores from the low-detection dump or re-inference) | medium | comes from a Hungarian pipeline without learned edges; largely redundant with transformer probabilities. Low |
| 7 | Trackastra P->D2 association as evidence for far daughters | jirkaborovec | an extra feature/veto for C016-type (P, D1, D2) candidates; the CTC checkpoint includes 3D nuclear embryo data | 1-2 days (masks from x138 detections, isotropic resampling) | low-medium | zero-shot linking was weak (0.616); speculative; not before 2026-09-29 |

## Not transplantable (negative evidence to keep)

- **Detector replacement or fine-tuning.** No public or own detector replacement has beaten the pilkwang weights on the LB:

  | attempt | public LB | note |
  |---|---:|---|
  | josephadamski adapted centered-window detector | 0.942 | with a V1284 head |
  | yongjilyu own hi-res ensemble | 0.888 | |
  | tangai1 all-199 SSL | 0.714 | 0.728 on the visible 4, which are in its training set |
  | our C008 | 0.926 | |

  "Train on all 199 movies" is what the public pilkwang weights already do (split manifest).
- **Purging forks or ultra-strict division rules:** alfonso lost about 0.03 on the hidden test.
- **Classical detectors (DoG / Gaussian peaks):** 0.10-0.86; nothing to reuse.
- **The exploit** (denpugovkin stem): excluded by policy, and it measured 0.885 with or without the stem.

## Files
- Pulled sources: `state/notebook_radar/pulled/{jirkaborovec__biohub-celltrack-dog-trackastra-graph-trans, yusuketogashi__no-hack-biohub-cell-another-approch-3rd, reyhanksatria__graph-patches-for-cell-tracking-0-917-lb, praxel__biohub-0-902-motion-division-calibration, tangai1__biohub-b02-all199-production-20260915-packed, tangai1__biohub-b02-dynamic-production-20260915, yongjilyu__biohub-ct-sub, binasalama__biohub-learned-unet-transformer-ilp-gap-recovery, alfonso1799__biohub-top-3-push-v50-streamlined-sota, josephadamski91__biohub-v1329-v1327-original-guard, denpugovkin__why-this-tracker-parks-an-ancestor-outside-the-vol, sujalsuyash__liquidtracker-core-biohub-lnn, fabriciodasilva__biohub-dodecatiad-cell-tracking, junaid512__biohub-cell-tracking-strategic-submission, sarveshchhetri__robust-3d-cell-tracking, tobimichigan__robust-cell-tracking-during-development-via-chan, seshurajup__lb-0-857-best-rule-base-v14}/`
- Key code locations:
  - V1057 reconciliation: `_v1057_reconcile_in_memory` in the single code cell of the josephadamski notebook
  - ranker loader and features: cells 6 and 15 of the yusuketogashi notebook (`_ranker_context`, `_ranker_aliases`, `motion_relink_edges.assign_pass`)
  - JS-TTA: cell 13 of the same notebook ("Biohub 154" block)
  - lookahead: cell 11 of the same notebook
