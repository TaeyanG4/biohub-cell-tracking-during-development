# Untried methods from all public notebooks and discussions (2026-09-24)

Sources: notebook radar refreshed 2026-09-24 06:05 UTC (269 public notebooks; 55 distinct-looking ones pulled and read by four parallel audits - `reports/public_ideas_audit_20260924_{A_learned,B_association,C_fusion_budget,D_alternative}.md`), and the whole discussion forum (~116 topics, ~80 technical threads read - `reports/untried_ideas_discussions_20260924.md`). Baseline of comparison: our tried list `reports/tried_methods_20260924.md`, current best C012 (public 0.952).

## Headline: one forum lead turned into a measured gain - jump-stabilized relink
- Lead: hengck23 "beware of jumps in ground truth track" (724283) and g john rao (729082): consecutive volumes that are byte-identical, and sudden whole-field translations.
- Measured on all 199 train movies (`src/frame_motion_audit.py`, `reports/frame_motion_audit.csv`): 947 frozen pairs (4.8 % of all frame pairs, in 114 of the 128 6bba movies, none in 44b6); the pair after a frozen pair carries a double step (GT median 3.28 um vs 1.82 um); 1,718 pairs (11.8 %) move coherently by >= 3 um (median 4.3 um, up to 58 um) in 137 movies of both embryos.
- Where C012 loses edges (22 movies, 16,551 GT edges): normal pairs 3.1 % missed, after-frozen pairs 7.9 %, coherent jumps 5-8 um 19.9 %, > 8 um 47.5 %. The raw ILP misses only 11.1 % / 27.5 % on those jumps - x138's distance-gated Hungarian relink (tight 5.5 um, seed flow from same-pair tight matches) breaks links the transformer had right.
- Fix (post-processing only, `src/eval_pp_variants_local.py --stabilize-relink`): per frame pair, the global shift is the component-wise median displacement of the ILP's own links (p >= 0.5, >= 8 links); pairs with |shift| >= STAB_MIN_UM are jumps; node positions are shifted by the cumulative jump sum only for the relink (geometry only, image lookups untouched), distances are recomputed on the original coordinates afterwards. On 97 movies the estimator's error vs the GT shift is ~1 um up to 8 um jumps; at 3 um it flags 884 of 1,239 true jumps with 40 false flags.
- Result (official metric, head v1 = C012 configuration):

| set | variant | total | d total | adj edge | divisions | movies better / worse |
|---|---|---|---|---|---|---|
| held-out 12 (all) | as is | 0.9470 | | 0.9293 | 3/2/12 | |
| | stab 3.0 | 0.9586 | **+0.0117** | 0.9410 | 3/2/12 | 8 / 3 (worst -0.0023) |
| held-out 6bba (LB proxy) | as is | 0.9510 -> pred LB 0.9514 | | | | |
| | stab 3.0 | 0.9613 -> pred LB 0.9545 | **+0.0103** | | | 5 / 1 |
| held-out 44b6 | stab 3.0 | 0.9467 | +0.0170 | | | 3 / 2 (both -0.0000) |
| confirm 10 (6bba, unseen by the head) | as is | 0.9404 | | 0.9204 | 3/6/6 | |
| | stab 3.0 | 0.9480 | **+0.0077** | 0.9266 | 3/5/6 | 6 / 1 (-0.0001) |

  Thresholds 2.0 / 2.5 / 3.0 / 4.0 um all give +0.010 +- 0.0005 on the held-out 6bba total (not a tuned optimum). Biggest movies: 44b6_0113de3b 0.885 -> 0.960, 44b6_267148e4 0.772 -> 0.846, 6bba_3db54e20 0.830 -> 0.868, 6bba_07e24132 0.846 -> 0.871, 6bba_05db0fb1 0.883 -> 0.907. A 75-movie robustness run (division-rich train batches) is in `reports/pp_replay/stabilize_head_v1_b0{0,1,2}.csv`.
- Why it should transfer: the error is an acquisition artefact of the shared instrument/protocol (host 724386), independent of detector memorisation; the fix only touches pairs whose own ILP links show a coherent shift. Predicted LB (6bba formula) 0.952 -> ~0.955; the formula's precision is about +-0.002.

## Ranked untried ideas (everything else)
| rank | idea | source | evidence | effort | local test |
|---|---|---|---|---|---|
| 1 | **Jump-stabilized relink** (above) - build C017 = C012 + this stage | forum 724283 / 729082 + our audit | +0.0103 held-out 6bba, +0.0077 confirm-10, +0.017 44b6 | 2-3 h build + verify | done |
| 2 | Extend stabilization to the gap-closing stages (single-frame gap, gap-2, low-detection filler also match across jump pairs) and a double-step motion prior for the pair after a frozen frame (after-frozen pairs still miss 7.9 %) | our audit | after-frozen pairs = 13.9 % of all misses | 3-4 h | harness |
| 3 | ILP-edge restore after post-processing (put back ILP links with p >= 0.3 that the relink replaced; forks untouched) | josephadamski V1057 (audit D) | overlaps with 1 (ILP was right on jumps); no isolated LB | 1.5-2 h | harness |
| 4 | Two-sided structured re-assignment (18 features incl. the child's next step, ILP membership, localisation shift; published weights) as the last stage | indarkarhana structured-trajectory (audit B) | author: gains on 10- and 8-movie local sets; LB ~ neutral vs base | 3 h (+4-6 h refit) | harness |
| 5 | One-frame lookahead bonus in the relink (candidates that continue smoothly to t+2) | lonnieqin / yusuketogashi (audits C, D) | LB flat in its own lineage | 2 h | harness |
| 6 | Appearance re-ID term in the relink cost (28-d cube descriptor + boosted trees, cost -= 8 P) | arnav170 reid3s (audit B) | +0.0031 adj out-of-fold, LB = base 0.947 | 8-10 h | harness |
| 7 | Nucleus-size division features (a dividing nucleus loses ~27 % volume by +3 frames, brightness flat) | zhincez notebook (audit A), Tim Krige 732474 | division scorer failed on base rate (6-23 % precision); needs +12 pp | 1-2 h AUC check, 5-8 h full | candidate CSVs + frames |
| 8 | Image-based motion prior (3D optical flow / learned motion field) instead of detection-based neighbourhood flow | hengck23 723655, Luis Rosar 734093 (-45 % motion error) | no LB evidence | 6-10 h | harness |
| 9 | ~~JS-consensus-weighted edge TTA over flipped views~~ **TESTED 2026-09-25, NEGATIVE**: 8-view / 8-view+mean / exact 4-view recipe all lower the adjusted edge score (held-out 12: 0.9416 -> 0.9393-0.9399; 6bba -0.004 to -0.005; 7-10 of 12 movies worse). Tool `src/build_tta_pool_repo.py`; HANDOFF section 22 | yusuketogashi (audits C, D) | closed | done | done |
| 10 | Fine-tune only the edge transformer (UNet frozen), oversampling division windows | noisyislands transformer-finetune (audit A) | no published result; base model saw every train movie | 16-24 h + GPU | local inference |
| 11 | Out-of-sample validation on external embryos (public Ultrack embryo `2024_03_22_dorado`, Zebrahub + `*_tracks.csv`, linajea 160328, CTC) | 7th place (741386), 1st place comment, host 734330 | the only honest check of corrective stages and model changes | GB-scale downloads (needs your OK) + 1-2 days | new |
| 12 | Synthetic CC0 dataset (165k divisions) / FOCUS-3D dense pseudo-labels / SuperGlue-style linker | José Freitas 732103, hengck23 738217 | +0.012-0.018 for a from-scratch 0.939 pipeline (hikaggler); 1st place: no gain | weeks | retraining |

Needs your approval before downloading: `hengck23/hengck23-cell-point-detector-demo` (43 MB, license unknown), `pilkwang/biohub-local-association-ranker-unet300-v1` (19 KB), `indarkarhana/biohub-trajectory-motion-runtime-v1` (code only in the dataset), `bhpepper/biohub-synthetic-5fold-ensemble-v1`, and any external embryo data (idea 11).

## Checked and not useful
Logit transforms (hard-negative margin = temperature, rank bonus, mutual-best: 8 LB results 0.943-0.945 on 3 bases), 035 leaf-prune + velocity, DAE per-video denoising (0.935-0.946, unseeded noise), dropping small components, XGBoost / TabPFN / MLP division or linker toys (weaker than our closed scorer), detector replacements (fine-tuned 3-frame detector 0.942 even with a V1284 head, own ensembles 0.888, all-199 self-supervised 0.714), zero-shot Trackastra 0.616, rule-based / classical trackers 0.43-0.86, HOCT from ball masks, CoTracker (2D), metric exploits (fake hubs, ancestors outside the volume, negative-time nodes). Removing forks entirely costs 0.035 on the LB (alfonso V1329 0.942 -> 0.907), so the division term is real on the hidden test.

2026-09-25 update: `kunaldesale2408/biohub-cell-tracking (0.953)` is a whitespace-only copy of x138 (identical code, same datasets incl. the x138 public head) - nothing new. Idea 2 (gap-closing stabilization) dropped on evidence (no pure breaks left, `reports/gap_headroom_c022.txt`); idea 3 became C021/C022 (+0.001 LB); idea 9 tested negative (above).

2026-09-25: `amanatar/optimized-biohub-max-score` (user request) = x138 + an in-kernel validator-selected 'metric-aligned' PP layer + runtime governor; its latest version errored (best score 0.910 from an old V1). Its weak-edge output filter measured -0.004 on 22 movies on top of C022; the other levers were already measured negative (leaf/node pruning, division relaxations). Nothing to adopt.

2026-09-26 refresh: radar 271 notebooks, nothing new beyond amanatar (reviewed); raunakdey07/biohub-harmonic-fusion-v3 (0.953) is x138 minus comments. New threads 743222 (imissher: 8 LB experiments all negative, local/LB sign flips), 742266 (Justin CH123: 10 single-knob changes lost, node-count tracking), 742942 (Quantizr: GT label errors) read - they confirm our closures; no new technique.

2026-09-27 user-screenshot refresh: four of five sources unchanged; amanatar optimized latestV6 is new. Its primary-preserving calibrated-logit max fusion is a distinct untested lead, while the latest score is absent and best.953 belongs to inaccessibleV4. Details and branch-risk audit: `reports/public_notebook_refresh_20260927.md`. Do not apply the old no-new-code conclusion toV6. Active C038 queue remains untouched.
