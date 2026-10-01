# Closed-arms ledger (lens: CLOSED ARMS), 2026-09-29, read-only

Sources: HANDOFF.md sections 18/21/25-43 (lines 787-921, 1042-1591), AGENTS.md top bullets,
experiments/candidates/c03x..c056/{FINAL_REVIEW,PILOT_REVIEW,REVIEW}.md + decision.json,
state/research_closure_20260928/REVIEW.md, state/priority_review_20260928/REVIEW.md,
state/split_audit_20260928/REVIEW.md, state/c055_diagnostics_20260929/*.csv, experiments/submission_log.csv.

Folds legend: pooled = no fitted component (fold irrelevant); XE = whole-embryo opposite fold (44b6<->6bba);
FIT = fixed model that saw the evaluated embryo (fit-domain evidence only).

| Arm | Mechanism | Closed on | Folds | Key number | Closure type | Reason class |
|---|---|---|---|---|---|---|
| C032 mean_det / future_det | temporal-context detection fusion | 97 | pooled | all97 -0.000395 / -0.000818; ext75 -0.00114 / -0.00161; adj-edge negative in BOTH embryos | evidence | negative local |
| C032 mean_det_head | + head on mean context | 22 | pooled | confirm10 -0.000759 | evidence (22 only) | negative local |
| C033 raw / stabilized | public 18-weight structured assignment | 22 | pooled (fixed public weights) | heldout -0.001057, confirm -0.000313 (identical both modes) | evidence (22 only, no refit) | negative local |
| C034 appearance ReID (HGB) | geometry vs geometry+28-d appearance | 22 | XE | 16 known negatives (6/10) < 20/class -> fits skipped | NOT evidence | insufficient supervision |
| C035 augmented ReID CNN | GT-supervised crops, 24 outside movies | 22 diag | XE | conservative net +1 (44b6) / -3 (6bba); negatives median 23.75/26.68 um, only 150/69 <=12 um | evidence, weak (far negatives) | negative diag |
| C036 local 3D NCC registration | candidate-independent motion | 22 diag | pooled | fixes/harms 1/0 (44b6), 0/0 (6bba); 1218/2302 (52.9%) boundary-rejected | evidence + coverage limit | negative diag |
| C037 Transformer finetune (early150/late600/blend25) | production-input attention/pair MLP | 22 | XE | +0.000357 / +0.000148 / -0.000276; all lose 6bba and confirm10; 0 division targets | not closed -> C040/C041 | superseded |
| C038 appearance / agreement stage | post-restore edge edits | 97 | XE | all97 +0.000012 / -0.0000006; ext75 -0.000085 / -0.000034 | evidence | not transferring |
| C039 primary_max / calibrated_max / public_division | Amanatar V6 components on C023 | 22 | pooled | -0.002212 / -0.002212 / -0.006702; div FP 6->31; negative all splits+embryos | evidence (22 only, large uniform) | negative local |
| C040 late600 (+appearance) | XE Transformer on 97 | 97 | XE | late600 all97 +0.001054 / edge +0.000546; appearance increment on ext75 -0.000033 | positive component -> C041 | -- |
| C041/C042/C043 fixed param-mean Transformer (+appearance) | pooled deployment | 22 local + LB | FIT | local22 +0.001202 / +0.001702; LB 0.954 / 0.954 (tie with C023) | LB: no displayed gain | not transferring (at display precision) |
| C044/C045 (C024 + fixed Transformer[+appearance]) | pooled deployment on 2-head base | 22 local + LB | FIT | local22 +0.001495 / +0.001938 vs C024; LB 0.953 / 0.953 (-0.001) | LB negative | not transferring |
| C046 fixed appearance-only on C023 | pooled | 97 local; T4 verified | FIT | all97 +0.000093; 6bba -0.0000015; ext75 6bba -0.000057; 9 wins/8 losses/80 ties | not submitted | insufficient evidence (fit-domain, tiny) |
| C047 hard-example appearance | 29 outside movies w/ 6-12 um competitors | stopped before fit | -- | pixel-identical overlap found (3 pairs, 4608 voxels 100%) | budget/design (split) | replaced by C048 |
| C048 whole-embryo appearance | all 190 eligible movies, XE | 97 | XE | all97 +0.0000388 (44b6 +0.000919, 6bba -0.0000612); ext75 -0.0000158; 11 W / 9 L | evidence | not transferring |
| C049 image division CNN (C031 corrected) | 3-frame crop CNN | GT-centre samples | XE | P@fixed .5 = .156/.245, R .04/.5; P@R>=.5 = .040/.302; AP .079/.298 | evidence (one dir. at break-even bar) | precision below break-even |
| C050 candidate division learner (C016 corrected) | 40-feature MLP @0.9 | 97 (27/70) | XE | TP/FP/FN 6/154/70 and 1/16/17 -> P .0375/.0588 | evidence | precision below break-even |
| C051 boundary-support registration | C036 boundary recentering | 22 diag | pooled | 484 new trusted groups; 0 new recovery; 1 lost match (to unknown) | evidence | negative diag |
| C052 division-supervised Transformer (XE) | 300 ord + 300 div steps | 97 | XE | all97 +0.001297 / edge +0.001227; edge TP +80 FP -28 FN -80; division TP net 0; usable daughter targets 39/246 | open (component positive) | -- |
| C053 fixed param-mean of C052 | deployment | 97 local + LB pending | FIT | +0.000516 / edge +0.000585; -0.000781 vs C052 XE; confirm10 -0.0000126; sub 56643044 | pending LB | -- |
| C054 fixed raw-logit output-mean of C052 | deployment | 97 local + LB pending | FIT | +0.000681 vs C023; +0.000165 vs C053; 29 W / 68 L; sub 56648523 | pending LB | -- |
| C055 V12 guarded endpoint readmission | replaces C023 readmit | 97 | pooled | v12 - readmit_off = -0.000357 (22), negative 72/75 ext movies; 44b6(27) -0.002840 vs control; only 6/213 missed GT nodes in 0.94-0.965 | evidence | negative local (node-term) |
| C056 StrongUNet guarded readmission | complementary detector candidates | 22 (gate) | pooled | -0.000339 vs readmit_off; 44b6 -0.001573 vs control; 56 nodes, 0 recovered; ceiling: p>=0.5 covers 40/213 nodes at 37.5% node burden | evidence + ceiling analysis | negative local |
| readmit_off (C015 lineage) | node-budget lever | 97 + LB | pooled | local all97 +0.000431 but 44b6 -0.002695 / 6bba +0.000792; LB C015 0.951 < C012 0.952 | LB negative | not transferring / sign flips by embryo |
| Node-budget trimming (sec 21) | remove low-match node groups | 22 + 97 analysis | pooled | best group (readmitted) +0.0010 6bba / -0.0006 44b6; no group below break-even | evidence | negative local |
| ILP division weight 0.7 / 0.5, fork re-injection (sec 21) | P->D2 prob as fork evidence | 12 | pooled | 578 / 1838 forks vs 15 GT; precision <= 2.6%; every reinjection variant < baseline | evidence | precision below break-even |
| OUTPUT_MIN_TRACK_LEN screen (sec 19, 2026-09-24) | 5/7/8 vs 6 on C012 graphs | 22 | pooled | sel set 7/8 +0.0018/+0.0024; unseen10 7 -> 0.0000, 8 -> -0.0001, 5 -> +0.0014 (+21 TP); sign flips | evidence (old C012 graphs, pre-stabilization) | negative local (weak: never re-screened on C023) |
| Public notebook lane | V6 0.965+, V12, x138 copies | actual T4 outputs | -- | V6 visible4 0.925 with repair_fallback on all 4; V12 best 0.947 = V8; all >=0.950 public are x138 copies | evidence | not real |

## Diagnostics that bound reopening

- 213 GT nodes missed by C023 on 22 movies (379 GT edges): 101 no primary peak >=0.3 within 3 um (188 edges); 91 production peaks (>0.965) dropped:
  63 never selected by the ILP (107 GT edges; 61 of them 6bba), 27 deleted by the short-track filter + 1 at motion relink (48 GT edges); 6 in V12 range 0.94-0.965; 15 lower.
  Source: state/c055_diagnostics_20260929/{missed_gt_nodes_22_summary,dropped_detections_stages_22}.csv (recount in this session: 63/28 confirmed).
- Division term: 149/151 parents and 268/302 daughters exist as final nodes; 27 TP / 46 FP / 124 FN. One recovered fork ~ +0.0005 on the 97 total, one false fork ~ -0.00013; break-even precision ~0.2-0.3 (HANDOFF sec 43).
- Local -> LB calibration from closed arms: C025 +0.0018 (local) -> -0.001 LB; C044 +0.0015 (22) -> -0.001; C042 +0.00105 (97, XE component) -> tie 0.954. Local gains < ~0.005 remain unproven.

## Closed on weak grounds (candidates for reopening with a specific ingredient)

1. C034 appearance ReID: closed for lack of negatives, not by a measured negative. Ingredient: negative labels that do not require an annotated competitor (e.g. the other daughter of a resolved division as hard negative; or predicted-graph pseudo-negatives validated only on XE folds). C035/C048 partially covered this (up to 96 close 6-12 um triplets per movie in C048) and were still ~0 on 97, so prior is low.
2. C052 division supervision: usable daughter targets were only 39/246 (20/125 windows) under the C037 known-parent-competition label rule, and daughter-link survival was flat (6bba admission +2 -> final 22->22). "Division supervision does not recover forks" is NOT established; the ingredient is a label-construction audit that raises usable daughter targets toward the 285 detector-covered daughter links (priority_review table A) without unknown-as-negative, plus a survival-aware objective. Still bounded by the ~0.3 fork-precision bar.
3. OUTPUT_MIN_TRACK_LEN: screened only on C012 graphs (pre stabilization/restore) on 22 movies with sign flips; the 28-node/48-edge pool is now exactly measured. A 97-movie CPU-only harness replay on C023 caches (min_len 5) would settle it cheaply; LB history for aggressive rescue (C005/C006 min_len=3, 0.944) is negative, so expect at most +0.001 local.
4. C033 structured assignment: fixed public weights, 22 movies only; a refit under XE folds was never tried. Low prior (both modes identical -> the stage barely acts).
5. ILP node selection (63 never-selected production peaks, 107 GT edges = 28% of missed GT edges on 22 movies): the only pipeline-internal pool larger than readmission's. Never targeted directly (disappearance 3.0 lost -0.0065 on confirm10; candidate threshold 0.48 never swept for isolated peaks). Any change pays the linear node term for every non-annotated peak admitted; needs a GT-free selector, none available.

Not reopenable without new data: detector recall (101/213 missed nodes have no primary peak >=0.3; StrongUNet covers only 30 of those, at 37.5% node burden) and whole-pipeline independent validation (2 embryos, frozen public detector/head saw both).
