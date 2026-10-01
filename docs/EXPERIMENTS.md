# Experiment catalogue (C001-C070)

This page lists every candidate built for the Kaggle competition *Biohub - Cell Tracking During Development* (3D zebrafish nucleus tracking). The score is the node-adjusted edge Jaccard plus 0.1 x division Jaccard.

A **candidate** is a numbered folder under [`experiments/candidates/`](../experiments/candidates/). Most candidates hold a Kaggle notebook. Later ones (from about C032) are often local studies with a README, a review file (`FINAL_REVIEW.md` or `PILOT_REVIEW.md`) and a `decision.json`. Names follow `cNNN_<short_name>`. A few experiment work areas share a prefix with a candidate; they are folded into that candidate's row.

**Public** is the public leaderboard score, computed on embryo `fdad`. **Private** is the private leaderboard score, computed on embryo `ea36` and read after the competition closed. Both are shown only for submitted candidates.

**Local** numbers come from the 199 training movies. Those movies come from only two embryos (`44b6` and `6bba`), and the public detector models had already been trained on them. Local numbers are therefore in-sample and were optimistic. Most local results use 22 evaluation movies (held-out 12 + confirm 10) or 97 movies (those 22 plus 75 division-rich movies). Deltas are against C023 unless stated otherwise.

## Status legend

| Status | Meaning |
|---|---|
| Submitted | Sent to the competition and scored. Public and private scores are shown. |
| Built, not submitted | A notebook was built (some also ran on Kaggle) but was never scored. |
| Closed (study) | A local study or diagnostic. It ended without a submission. |
| Superseded | Replaced by a later candidate before it was evaluated or submitted. |

C059 is the only ID with no folder. It was reserved for a temporal division-feature study that was deferred and never run.

## Phase A: C001-C011, tuning the public baseline and moving to x138

B0 is Reyhan's public 0.947 notebook. x138 is `anvithpothula/biohub-x138`, a public 0.953 notebook built on the same detectors with a learned coordinate head (V1284), a flow motion prior, readmission and gap filling.

| ID | Idea | Status | Public | Private | Outcome / lesson |
|---|---|---|---|---|---|
| [C001](../experiments/candidates/c001_r3_temporal/) | Temporal-history association on B0 | Superseded | - | - | Byte-identical copy of B0; the planned changes were never added. |
| [C002](../experiments/candidates/c002_node_rescue/) | Rescue missed nodes from a second detector (StrongUNet) | Superseded | - | - | Only the peak cache was built. The idea was measured later in C056 and did not pay. |
| [C003](../experiments/candidates/c003_medal_frontier/) | B0 + relink gate 5.5 um + relaxed division geometry + wider validator grid | Submitted | 0.946 | 0.914 | Below B0. Division gains were measured on GT graphs, not on predicted graphs. |
| [C004](../experiments/candidates/c004_adaptive_lineage/) | B0 + wide division envelope + in-notebook validator | Submitted | 0.948 | 0.915 | The validator rejected the wide envelope; the scored run used B0's strict division rules + 5.2 um gate. |
| [C005](../experiments/candidates/c005_gold_fusion/) | C004 ideas + velocity relink + short-track rescue (min length 3) | Submitted | 0.944 | 0.914 | Short-track rescue and velocity weight 0.75 added noise. |
| [C006](../experiments/candidates/c006_consensus_fusion/) | C005 variant with gap-anchor tuning | Submitted | 0.944 | 0.914 | Same regression as C005. |
| [C007](../experiments/candidates/c007_champion_frontier/) | Bidirectional edge consensus + aggressive rescue | Submitted | 0.946 | 0.915 | Still below C004. |
| [C008](../experiments/candidates/c008_local_transformer_fusion/) | C004 + a UNet-Transformer we trained locally, fused as secondary model | Submitted | 0.926 | 0.914 | Large public drop; attributed to a feature-space mismatch with the public models. |
| [C009](../experiments/candidates/c009_boost_geometric_fusion/) | C008 + weak-leaf pruning + per-embryo validator guard | Built, not submitted | - | - | Ran on Kaggle; held after C008's result. |
| [C010](../experiments/candidates/c010_x138_flow_fusion/) | x138 + C004 wide division envelope, head disabled | Superseded | - | - | Local replay: wide envelope costs 0.0070 on x138 post-processing (division FP 3 to 16). |
| [C011](../experiments/candidates/c011_x138_zero/) | x138 with the private coordinate head set to pass-through | Built, not submitted | - | - | Equal to a public head-less twin (0.946). Showed that x138's gain came from its head. |

## Phase B: C012-C031, coordinate head, stabilised relink, ILP-edge restore, head ensembles, knob probes, division models

| ID | Idea | Status | Public | Private | Outcome / lesson |
|---|---|---|---|---|---|
| [C012](../experiments/candidates/c012_v1284_head/) | x138 + our own coordinate head (v1, 30 movies) | Submitted | 0.952 | 0.924 | Local +0.0103 on held-out 12 became +0.006 public. Tied for best private score. |
| [C013](../experiments/candidates/c013_v1284_head_a05/) | C012 with head shifts at half strength | Submitted | 0.948 | 0.921 | Kept about a third of C012's public gain. |
| [C014](../experiments/candidates/c014_v1284_head_noreadmit/) | Head v2 (71 movies, balanced) + readmission off | Submitted | 0.949 | 0.920 | Best local all-12 score, lower public. Public order matched the local 6bba subgroup, not the all-12 total. |
| [C015](../experiments/candidates/c015_v1284_head_v1_noreadmit/) | C012 with readmission off | Submitted | 0.951 | 0.923 | Confirm-10 said +0.0022; public said -0.001. |
| [C016](../experiments/candidates/c016_division_scorer/) | Learned division scorer after x138's rule | Closed (study) | - | - | Held-out precision 6-8.5% at threshold 0.9; break-even was about 15%. Not pushed. |
| [C017](../experiments/candidates/c017_jump_stabilized_relink/) | C012 + jump-stabilised relink (shift removed on pairs with >= 4 um global jumps) | Submitted | 0.952 | 0.924 | +0.0071 on 97 movies, no public gain. Work area `c017_ilp_division`: a lower ILP division weight made 578 forks for 15 GT divisions; closed. |
| [C018](../experiments/candidates/c018_frozen_catchup_prior/) | C017 + frozen-frame catch-up prior | Submitted | 0.952 | 0.924 | +0.0004 locally; submitted as a probe. |
| [C019](../experiments/candidates/c019_ilp_seeded_flow/) | C017 + relink flow seeded from the ILP's own links | Closed (study) | - | - | +0.0000 to +0.0004 locally; rejected. |
| [C020](../experiments/candidates/c020_stabilize_all_pairs/) | C012 + stabilisation on every frame pair | Submitted | 0.953 | 0.924 | +0.0088 on 97 movies. |
| [C021](../experiments/candidates/c021_ilp_edge_restore/) | C017 + restore ILP edges (p >= 0.7) displaced by the relink | Submitted | 0.953 | 0.923 | +0.0017 on 97 movies vs C017. |
| [C022](../experiments/candidates/c022_stabilize_all_restore/) | C020 + ILP-edge restore | Submitted | 0.953 | 0.923 | +0.0029 vs C017 on 97. Post-processing base for all later candidates. |
| [C023](../experiments/candidates/c023_x138_head_stabilize_restore/) | C022 with x138's public head instead of ours | Submitted | 0.954 | 0.917 | Locally worse than C022, best public. Lowest private of the 0.954 group. |
| [C024](../experiments/candidates/c024_head_ensemble/) | C022 with the mean shift of two heads (ours v1 + x138) | Submitted | 0.954 | 0.923 | Tied C023 on public, 0.006 higher on private. Work area `c024_tta_pool`: probability-level edge TTA lowered adjusted edge by ~0.002; closed. |
| [C025](../experiments/candidates/c025_pp_settings/) | C022 + readmit 3 um / gap-close 4 um / linefit 0.6 | Submitted | 0.952 | 0.923 | +0.0018 on 97 movies, -0.001 public. Work area `c025_fusion_weights`: fusion weights and ILP disappearance cost; gains flipped sign on confirm-10; closed. |
| [C026](../experiments/candidates/c026_head_v4/) | C022 with head v4 (187 movies) | Built, not submitted | - | - | Best pair-level error, worst local end-to-end (0.9465). |
| [C027](../experiments/candidates/c027_head_ensemble3/) | C022 with three heads (v1 + x138 + v4) | Submitted | 0.953 | 0.924 | Head v4 cost 0.001 public vs C024. |
| [C028](../experiments/candidates/c028_x138head_pp_settings/) | C023 + C025 settings | Submitted | 0.951 | 0.918 | Settings lost 0.003 public. |
| [C029](../experiments/candidates/c029_ensemble_pp_settings/) | C024 + C025 settings | Submitted | 0.950 | 0.921 | Best local held-out score (0.9646), lost 0.004 public. |
| [C030](../experiments/candidates/c030_adabn/) | C023 + AdaBN test-time BatchNorm adaptation | Built, not submitted | - | - | Locally 44b6 -0.019, 6bba +0.005; unstable by embryo. |
| [C031](../experiments/candidates/c031_division_cnn/) | 3D CNN on crops to recognise dividing nuclei | Closed (study) | - | - | Reported 0.58 precision at recall 0.5; C049 later showed that number was precision at recall 0.29. |

## Phase C: C032-C041, temporal context, structured assignment, appearance, registration, Transformer fine-tuning

| ID | Idea | Status | Public | Private | Outcome / lesson |
|---|---|---|---|---|---|
| [C032](../experiments/candidates/c032_temporal_context/) | Use the second temporal window's detection map (mean or future) | Closed (study) | - | - | 97 movies: mean -0.00039, future -0.00082; edge score lower in both embryos. |
| [C033](../experiments/candidates/c033_structured_trajectory/) | Public structured trajectory assignment after C023 | Closed (study) | - | - | -0.0011 held-out 12, -0.0003 confirm 10, raw and stabilised alike. |
| [C034](../experiments/candidates/c034_appearance_reid/) | Appearance re-identification of ambiguous links | Closed (study) | - | - | Only 16 known negative pairs in 22 movies; no model could be fitted. |
| [C035](../experiments/candidates/c035_augmented_reid/) | CNN appearance embedding with augmentation, trained on 24 other movies | Closed (study) | - | - | Net +1 (44b6) and -3 (6bba) against C023 choices; failed the gate. |
| [C036](../experiments/candidates/c036_local_registration/) | Local 3D image registration to predict motion | Closed (study) | - | - | 1 fix / 0 harms; 52.9% of groups rejected at the image boundary. |
| [C037](../experiments/candidates/c037_transformer_finetune/) | Fine-tune the edge Transformer on known links, cross-embryo folds | Closed (study) | - | - | 22 movies: +0.0004 / +0.0001 / -0.0003 for three arms; all lost on confirm 10 and 6bba. Models reused later. |
| [C038](../experiments/candidates/c038_complementary_fusion/) | GT-free appearance and motion-agreement relinking after C023 | Closed (study) | - | - | +0.0003 / +0.0001 on 22 movies; ~0 on 97 and negative on the extra 75. |
| [C039](../experiments/candidates/c039_public_v6_salvage/) | Components of a public notebook claiming 0.965+ | Closed (study) | - | - | Its actual run crashed into a fallback. Components on C023: -0.0022 and -0.0067. |
| [C040](../experiments/candidates/c040_late600_combo/) | C037's 600-step Transformer with and without appearance relinking | Closed (study) | - | - | 97 movies: +0.00105 and +0.00117, both embryos positive. Appearance increment negative on the extra 75. |
| [C041](../experiments/candidates/c041_fixed_models/) | One fixed model for all movies (Transformer parameter mean, appearance mean) | Closed (study) | - | - | 22 movies: Transformer +0.0012, both +0.0017. Became C042/C043. |

## Phase D: C042-C054, portable fixed-model submissions, whole-embryo split correction, division-supervised Transformer

In this phase a split audit showed that the 199 movies come from two embryos and that some sources overlap at the pixel level. From C048 on, new models were validated by training on one embryo and testing on the other.

| ID | Idea | Status | Public | Private | Outcome / lesson |
|---|---|---|---|---|---|
| [C042](../experiments/candidates/c042_fixed_transformer/) | C023 + C041 fixed Transformer | Submitted | 0.954 | 0.918 | Tied C023 on public. |
| [C043](../experiments/candidates/c043_fixed_transformer_appearance/) | C042 + fixed appearance relinking | Submitted | 0.954 | 0.918 | Same scores as C042. |
| [C044](../experiments/candidates/c044_fixed_ensemble_transformer/) | C024 + C041 fixed Transformer | Submitted | 0.953 | 0.923 | 22 movies +0.0015 vs C024; public -0.001. Study folder [`c044_c024_fixed_study`](../experiments/candidates/c044_c024_fixed_study/). |
| [C045](../experiments/candidates/c045_fixed_ensemble_transformer_appearance/) | C044 + fixed appearance relinking | Submitted | 0.953 | 0.923 | 22 movies +0.0019 vs C024; same scores as C044. |
| [C046](../experiments/candidates/c046_fixed_appearance/) | C023 + fixed appearance relinking only | Submitted | 0.954 | 0.917 | 97 movies +0.00009; 6bba slightly negative. Study folder [`c046_fixed_appearance_extension`](../experiments/candidates/c046_fixed_appearance_extension/). |
| [C047](../experiments/candidates/c047_hard_example_appearance/) | Appearance model with more close-competitor training data | Superseded | - | - | Stopped before fitting when the split audit found embryo overlap; replaced by C048. |
| [C048](../experiments/candidates/c048_embryo_holdout_appearance/) | Appearance model trained only on the opposite embryo | Closed (study) | - | - | 97 movies +0.00004; 44b6 up, 6bba down. Not promoted. |
| [C049](../experiments/candidates/c049_division_embryo_audit/) | C031 division CNN re-run with whole-embryo folds | Closed (study) | - | - | Precision 0.04 and 0.30 at recall >= 0.5. Corrected C031's metric. |
| [C050](../experiments/candidates/c050_division_candidate_audit/) | C016 division scorer re-run with whole-embryo folds | Closed (study) | - | - | Precision 0.038 and 0.059; far below break-even. |
| [C051](../experiments/candidates/c051_boundary_registration/) | C036 registration with recentred windows at image borders | Closed (study) | - | - | Coverage 41.6% to 62.6%, but 0 new recoveries and 1 lost match. |
| [C052](../experiments/candidates/c052_division_transformer/) | Fine-tune the Transformer with division windows added, opposite-embryo folds | Closed (study) | - | - | Positive on 97 movies in both embryos (+0.0018 with the organiser's evaluator). Division TP unchanged. Source of C053/C054/C065-C070. |
| [C053](../experiments/candidates/c053_fixed_division_transformer/) | Parameter mean of the two C052 models, on C023 | Submitted | 0.954 | 0.919 | 97 movies +0.0011 (fit-domain); kept less of C052's gain. |
| [C054](../experiments/candidates/c054_output_ensemble/) | Mean of the two C052 models' raw logits, on C023 | Submitted | 0.953 | 0.919 | 97 movies +0.0013 (fit-domain). |

## Phase E: C055-C070, readmission, localisation studies, pooled and single-source Transformers, final combinations

A diagnostic before this phase found that 1,047 of 1,385 missed links (both ends present) touch a node more than 3.5 um from its GT position. C058-C064 tried to fix that localisation error.

| ID | Idea | Status | Public | Private | Outcome / lesson |
|---|---|---|---|---|---|
| [C055](../experiments/candidates/c055_guarded_readmit/) | Motion-guarded endpoint readmission (from Lineage Forge V12) | Closed (study) | - | - | Own effect -0.00036 on 22 movies; 44b6 -0.0028 on 97. Added nodes recovered almost no GT edges. |
| [C056](../experiments/candidates/c056_complementary_readmit/) | Same guard with StrongUNet peaks as candidates | Closed (study) | - | - | Failed its gate (-0.00034 vs readmission off); no annotated edge recovered. |
| [C057](../experiments/candidates/c057_conditional_parent/) | C052 with an extra loss on competing parent links | Closed (study) | - | - | 22 movies +0.00085 vs C023 but -0.00063 vs C052; lost on 44b6. |
| [C058](../experiments/candidates/c058_raw_localizer/) | Raw-image CNN to move node coordinates on a frozen graph | Closed (study) | - | - | 22 movies -0.0062; mean residual 1.77 to 2.13 um. |
| C059 | Temporal division features | Not run | - | - | Deferred; no folder. |
| [C060](../experiments/candidates/c060_spatial_track_localizer/) | Three-frame spatial localiser with strict ownership guards | Closed (study) | - | - | Score unchanged on all 22 movies; good points slightly worse in both embryos. |
| [C061](../experiments/candidates/c061_direct_axial/) | Direct z (axial) recentering trained on GT points | Closed (study) | - | - | Opposite-embryo 3D error 1.78 to 2.78 um; 20 of 22 movies worse. |
| [C062](../experiments/candidates/c062_tracklet_appearance/) | Three-frame tracklet appearance vs single frame | Closed (study) | - | - | Inconclusive: no known mistakes for it to fix in either embryo. |
| [C063](../experiments/candidates/c063_frozen_spatial/) | Small head on frozen production UNet features to refine positions | Closed (study) | - | - | Pooled 3D error 1.78 to 1.96 um; good points worse in both embryos. |
| [C064](../experiments/candidates/c064_deepcenter_axial/) | z refinement from the frozen DeepCenter field | Closed (study) | - | - | 3D error 2.74 to 3.41 um (44b6) and 2.97 to 3.83 um (6bba). |
| [C065](../experiments/candidates/c065_pooled_transformer/) | One Transformer fitted directly on both embryos' C052 data | Submitted | 0.953 | 0.918 | 97 movies +0.0011 (fit-domain); below C052 opposite folds. |
| [C066](../experiments/candidates/c066_expanded_ordinary/) | C052 recipe with 87 more movies of ordinary links | Closed (study) | - | - | Opposite-embryo 97 +0.0014 vs C023 but -0.0004 vs C052; production fit not run. |
| [C067](../experiments/candidates/c067_single_source_44b6/) | One C052 model (trained on 44b6) used for every movie | Submitted | 0.954 | 0.918 | 97 movies +0.0015; 6bba gain, 44b6 below C065. |
| [C068](../experiments/candidates/c068_single_source_6bba/) | One C052 model (trained on 6bba) used for every movie | Closed (study) | - | - | 97 movies +0.0011; only diagnostic proof, no time for full verification. |
| [C069](../experiments/candidates/c069_fixed_appearance/) | C067 + fixed appearance relinking | Submitted | 0.954 | 0.918 | 97 movies +0.0017, the highest local score; same scores as C067. |
| [C070](../experiments/candidates/c070_fixed_appearance/) | C065 + fixed appearance relinking | Submitted | 0.953 | 0.918 | 97 movies +0.0012; same private score as C065. |

## Patterns

- **Small local gains rarely transferred.** C025's settings gained +0.0018 on 97 movies and lost 0.001 on public; adding them to C023 and C024 (C028, C029) lost 0.003-0.004. C015's +0.0022 on confirm 10 became -0.001. The Transformer lane (C042-C070) gained +0.001 to +0.0017 locally and ended at 0.953-0.954 public and 0.917-0.919 private, close to C023 (0.954 / 0.917).
- **The x138 head won public but lost private.** Candidates using x138's head alone scored 0.917-0.919 private. Candidates using our v1 head, alone or in an ensemble, scored 0.920-0.924. C023 and C024 tied at 0.954 public but differed by 0.006 private. The best private score, 0.924, went to C012, C017, C018, C020 and C027.
- **Public differences in phase A did not show up privately.** C003-C008 ranged from 0.926 to 0.948 public but all scored 0.914-0.915 private. Stabilisation and ILP-edge restore (C017-C022) gained up to 0.001 public over C012 and changed private by 0 to -0.001.
- **Division models never reached the needed precision.** A new fork breaks even at roughly 0.15-0.3 precision, depending on the analysis. Held-out precision was 6-8.5% for the C016 scorer, 0.04-0.06 for its whole-embryo re-run (C050) and 0.04-0.30 for the division CNN (C049). Missed divisions are mostly fork-linking failures: the daughters exist as nodes, but the graph has no fork.
- **The whole-embryo split corrected earlier validation.** Movie-level cross-validation mixed the two embryos. Re-running with one embryo for training and the other for testing lowered earlier numbers (C031's 0.58 precision was really precision at recall 0.29). Results with models trained on both embryos are labelled fit-domain.
- **The local metric replica differed from the organiser's evaluator.** On the same 22 C023 graphs the replica gave 0.9460 and the organiser's code 0.9431, with different division counts. After an audit on the last day, C052-C054 were re-scored and C065-C070 were scored with the organiser's code.
- **Localisation studies did not reduce coordinate error on the opposite embryo.** C058, C060, C061, C063 and C064 all left mean error unchanged or worse, often for well-localised points too, although localisation error touches about three quarters of the missed links.
- **Pair-level accuracy did not predict the end-to-end score.** Head v4 (C026) had the best coordinate error and the worst local score. The x138 head was worse locally and better on public.
