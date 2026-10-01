# Public notebook audit A: learned models, divisions, fine-tuning (2026-09-24)

Scope: 12 public notebooks. For each one I checked what it does that is not already in our pipeline (the x138 / harmonic-fusion lineage, see `reports/tried_methods_20260924.md`). This was read-only. I pulled notebook source only (`kaggle kernels pull -m`). I did not download any kernel outputs, datasets or weights. For hengck23's dataset I only listed its files.

Method: I extracted the code and markdown cells (and any embedded images, which is where hengck23 put his result tables). I then diffed them against `state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb` and `kaggle_notebooks/flexonafft_harmonic_fusion/biohub-harmonic-fusion.ipynb`, with comments and blank lines removed first. The `.py` script kernels have no saved outputs. The `.ipynb` kernels were saved with outputs cleared, so the only evidence is what the authors wrote in markdown or in images.

Terms used below:
- **In-sample**: the public UNet and transformer were trained on all 199 train movies (HANDOFF section 20, `split_manifest.json`). Any local score from those models is therefore optimistic.
- **6bba proxy**: our local-to-LB fit, `LB ~ 0.298 x local 6bba total + 0.668`.
- **Per-parent precision**: out of the parents where a fork is accepted, the fraction that are true divisions. The C016 go/no-go bar was 35 %.

## Summary

| # | ref | public LB | class | one line |
|---|---|---|---|---|
| 1 | noisyislands/biohub-xgboost-division-events | none | TRIED | XGBoost on 13 GT-geometry fork features, with trivially easy random negatives. Weaker than our closed C016 scorer. |
| 2 | noisyislands/biohub-linker-association-mlp | none | TRIVIAL | Toy 6-feature distance MLP, 2 epochs on 4 movies. The author later says it was distance-inverted ("local S 0.32"). |
| 3 | noisyislands/biohub-linker-association-mlp-v2 | none | TRIVIAL | 8 motion/density features, GT-only. Already covered by our velocity plus flow-prior relink. |
| 4 | noisyislands/biohub-cell-tracking-tabpfn-3-5-events | none | TRIED | Same GT-only division features as #1, with TabPFN 3.5 as the classifier. |
| 5 | noisyislands/biohub-transformer-finetune | none | NEW-MODEL | Template: frozen UNet, fine-tune the transformer only, with synthetic duplicate/noise hard negatives. No reported result. |
| 6 | zhincez/a-dividing-nucleus-gets-smaller-not-dimmer | none | NEW-POSTPROCESS (feature idea) | Measured division signature: nucleus size drops from dt -2 to +5 while peak brightness stays flat. |
| 7 | xiaoleilian/biohub-ct-mix-divaug | 0.884 | EXPLOIT | Own weak 2-UNet heatmap detector with Hungarian linking, plus the fake hub/fork metric hack. |
| 8 | xiaoleilian/biohub-unet3d-v2models-training-code | none | NEW-MODEL (dominated) | Heatmap UNet3D trained from scratch. Val recall 0.58-0.67, far below the pilkwang detector. |
| 9 | hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp | none | NEW-MODEL + NEW-DATA | End-to-end detector and linker trained **without Kaggle labels** (FOCUS-3D pseudo-centroids with simulated motion). Title claims raw edge J 0.9 with no ILP. |
| 10 | hengck23/cell-point-detector | none | TRIED (partially, C002) | StrongUNet3D point detector. Recall >= 0.99 on sparse GT, but 1.05-2.4x over-count. We already have its weights. |
| 11 | gautiermarti/biohub-deepcenter-unet3d | 0.945 | TRIVIAL-FORK | HF fork with a few knobs changed. Its markdown proposes a "link-logit TTA" that was never implemented. |
| 12 | codezzzsleep/biohub-095-owned-validation | none | EXPLOIT | Simplified single-seed pilkwang pipeline plus the same fake hub/fork hack. |

All 12 pulls succeeded. They are in `state/notebook_radar/pulled/<author>__<slug>/`.

---

## Per-notebook detail

### 1. noisyislands/biohub-xgboost-division-events (script, no LB) - TRIED
- **Method:** A GBDT (`binary:logistic`, depth 4, 200 rounds, `scale_pos_weight = n_neg/n_pos`) on 13 features. They are `parent_daughter_dist_{min,max,sum,absdiff}_um`, `sister_dist_um`, `sister_symmetry`, `daughter_separation_per_frame_um`, `n_frames_before/after`, `is_first/last_frame`, and `local_density` / `parent_velocity_um`, which are always NaN in GT-only mode.
- **Labels:** Positives are the GT forks from all train GEFFs (at most 151 in 199 movies). Negatives are single-child GT parents paired with a random node from **the parent's own frame** (`by_t[t_split]`), 40 per parent, capped at 10k. The negatives are therefore geometrically impossible "daughters" and trivially separable.
- **Pipeline role:** Trains and saves `xgboost.json`. There is no inference integration and no evaluation on detector candidates. It only prints in-sample accuracy.
- **Versus ours:** C016 already used a strict superset (48 features including all of this geometry), realistic post-rule candidates, and labels that mirror `score_divisions()`. C016 reached 6-23 % per-parent precision and was closed. Nothing new here.

### 2. noisyislands/biohub-linker-association-mlp (script) - TRIVIAL
- **Method:** An `EdgeMLP` 6->16->16->1 on `[dist/10, dt, is_first, is_last, max(0, d-4)/10, n_competing_sources]`. Candidates are within 10 um, top-3 per target, built from GT nodes. The "production_sim" mode injects 15 % duplicates (<= 3 um) that **inherit positive GT edges**, plus random noise nodes. Training is 2 epochs on 4 movies. There is no integration.
- **Evidence:** The v2 docstring says v1 "failed as a ranker ... distance-INVERTED (P(9um) > P(2um)) ... local S 0.32".

### 3. noisyislands/biohub-linker-association-mlp-v2 (script) - TRIVIAL
- **Method:** An MLP 8->16->16->1 on `d/10`, cosine(step, previous velocity), `|step - vel|`, speed ratio, rank among sources, number of competitors, density within 10 um, and a track-start flag. It uses GT velocities, GT plus synthetic nodes, a random 20 % held-out movie split, and a "distance-slope gate" that rejects inverted models.
- **Evidence:** Only a val AUC gate (promote if > 0.7). No score is reported.
- **Versus ours:** x138's Hungarian relink already combines learned probability, a velocity prior and a neighbourhood-flow prior. This MLP is strictly weaker than the transformer.

### 4. noisyislands/biohub-cell-tracking-tabpfn-3-5-events (notebook) - TRIED
- **Method:** The same GT-only division rows as #1. Negatives here are single-child parents paired with a random non-child from frame t+1, anywhere in the frame, so still trivially easy. It uses 8 movies (`BIOHUB_MAX_MOVIES`) and a leave-one-movie-out outer fold. It fits `TabPFNClassifier` with the TabPFN 3.5 weights from `Prior-Labs/tabpfn_3_5` (downloaded from Hugging Face; 8 estimators), an affine calibration, and a LightGBM reference.
- **Evidence:** None (outputs cleared).
- **What is new:** Only the classifier. TabPFN is an in-context tabular model that is strong with ~100 positives. Our C016 bottleneck was the base rate (about 2.5 recoverable divisions per movie against about 5,000 candidate parents), not the classifier. See idea 7.

### 5. noisyislands/biohub-transformer-finetune (script) - NEW-MODEL
- **Method:** An architecture-parity copy of pilkwang's `UNetNodeTransformer`: TemporalUNet3D(1,32,[32,64,128]), detect head, and SimpleNodeTransformer(feat 64, hidden 128, 4 heads, 4 blocks, dropout 0.3). It loads `edge_predictor_best.pth` strictly and asserts 0 missing and 0 unexpected keys.
  - **Freezing:** The UNet and detect head are frozen and kept in eval mode, so BatchNorm stats do not drift. Only `transformer.*` is trained, including `pair_mlp`.
  - **Data:** 2-frame GT windows (t, t+1). Node coordinates are **GT positions, not detections**. UNet features are taken at those positions plus a 32-d sinusoidal position embedding. It uses at most 25 windows per movie, which are the **first** 25 frames. The movie split is a random 80/20 (seed 7).
  - **Hard negatives:** 15 % of frame-t nodes are duplicated (+-2 z, +-7 yx voxels). The duplicate rows **copy the source's positive targets**. There are also 2 random noise nodes per frame, forced active with target 0.
  - **Loss:** The original softmax-over-sources BCE with focal (1-p_t)^2 weighting, over the extended mask. AdamW lr 2e-5, 6 epochs (pilot 2, "full" 8). It checkpoints on val loss.
  - **Output:** `edge_predictor_ft_best.pth`. It is saved as a `{epoch, model_state, ...}` dict, so `model_state` must be extracted before it is truly drop-in.
- **Evidence:** None. There are no metrics in the source, no LB and no public follow-up kernel.
- **Assessment:**
  - Freezing the UNet avoids the failure that sank C008 (retrained UNet, feature-scale mismatch, LB 0.926). This is the only cheap route into our known gap, "fine-tune the edge transformer".
  - As written, the recipe has four problems. (a) Clean GT inputs do not reproduce the production detector distribution. (b) Duplicates are labelled positive, which teaches the model to link duplicates, while the forum says duplicates cost about 9 %. (c) The random movie split is meaningless, because the base model saw every train movie. (d) Taking only the first 25 frames biases the data toward early stages.
  - See idea 1 for a corrected version.

### 6. zhincez/a-dividing-nucleus-gets-smaller-not-dimmer (EDA) - NEW-POSTPROCESS (feature idea)
- **Method:** For each GT division (window of 6 frames, 60 division-richest films), the author compares against up to 3 **paired** ordinary tracks from the same film starting on the same frame. Three numbers are measured in a +-5 voxel box (11x11x11 voxels, about 17.9 x 4.5 x 4.5 um):
  - peak above background (p10 of the box), i.e. brightness;
  - half-max volume, the voxel count at or above bg + 0.5 x (peak - bg), i.e. size;
  - the naive mean.
  Each track is normalised to its own first frame, and the notebook reports median differences with 6,000-sample bootstrap intervals.
- **Evidence (author's markdown; numeric outputs were cleared):**
  - Volume is below the controls with intervals clearing zero at dt = -2 and -1 (the -1 bound only just) and at +1 to +5. The deepest point is about **-0.27 at +3**, and it recovers by +6.
  - Peak brightness clears zero only at the division frame itself. The daughters are as bright as the parent.
  - The mean falls from +1 to +6 only because of the size change. The author's first version, with a fixed-radius ball, read 0.84 and mistook it for dimming.
  - The notebook also confirms that there are 151 labelled divisions in 199 films.
- **Versus ours:** C016's appearance features are `int_*_peak` and `int_*_mean` in a 3x7x7 window (`src/division_scorer_stage.py`). They have no size or shape measure, and their mean is exactly the confounded quantity this notebook warns about. Nucleus size is on our "not tried" list.
- **How it would plug in:** Half-max volume of P(t-1), P(t), D1(t+1), D2(t+1), and D1/D2 at t+2, plus ratios. True daughters should both shrink about 15-27 %. In a false fork, D1 is a continuing cell (ratio about 1) and D2 is an unrelated cell. A second-moment elongation of P at t-1/t (the "condensed band" visible in the animation) is a natural companion. The notebook does not measure it.
- **Caveat:** The effect is a shift in medians with wide IQR bands. It is unlikely to have the 0.02 %-false-fork specificity C016 lacked on its own. See idea 2.

### 7. xiaoleilian/biohub-ct-mix-divaug (LB 0.884) - EXPLOIT
- **Pipeline:** Two UNet3D (base 24) heatmap models on an XY-pooled (x4) volume, one on raw input and one on tophat input (`grey_opening` 1x7x7). Their heatmaps are averaged. Peak threshold 0.15, 4 um physical NMS, intensity-weighted centroid refinement in the raw volume. Linking is a two-pass Hungarian (6 / 10 um gates, 0.5 x velocity prediction), followed by a short-track filter (4) and line-fit smoothing. The gap closing path is disabled (`GAP_DT=0`).
- **The last cell is an explicit metric exploit.** It adds a hub node at t=-1000 linked to up to 1,400 roots, plus 5 synthetic forks at negative times. The author states "Validated offline on host_metric replica: VAL 0.8388 -> 0.9203 (+0.0815)". The honest pipeline is therefore about 0.80.
- A useful side note in the code: "DoG-union U-Net measured 0.661 -- over-detection blows past T_true". This matches our node-count findings.
- Nothing to take. We exclude exploits on purpose.

### 8. xiaoleilian/biohub-unet3d-v2models-training-code - NEW-MODEL (dominated)
- **Training recipe for #7's detectors:**
  - UNet3D base 32 on the XY/4-pooled 64^3 grid.
  - Target: Gaussian stamps (sigma 1) with weighted BCE (positive weight 12, background below the q40 intensity weight 1, everything else 0.05).
  - Augmentation: flips and rot90 ("rich"), plus "bright" augmentation (intensity x U(0.5,1.3), low-frequency Gaussian background field, additive noise).
  - Tophat preprocessing for high-background movies, named in the code as 6bba_05db0fb1 and 44b6_0b24845f.
  - Embryo-grouped split: 12 val movies per embryo, and the 4 visible movies (TEST4) never trained on.
- **Evidence:** Best val recall (peak within 2 pooled voxels) was 0.579 (v2 tophat, OOM-stopped around epoch 8), 0.674 (v4 bright 40 ep) and 0.649 (seed 1). Their local TEST4 replicas were "classical 0.750, prior U-Net 0.643". A full v2 run takes about 13 h on an RTX 4090 and about 90 h on a T4.
- **Versus ours:** Far weaker than the pilkwang detector. C008 already showed that a locally trained detector does not transfer. The only reusable detail is the tophat/background augmentation, which matters only if we ever retrain the detector.

### 9. hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp - NEW-MODEL + NEW-DATA
- **Method:** `End2EndCellLinker` (`model_v12.py` in the public dataset `hengck23/hengck23-cell-point-detector-demo`, checkpoint `00000008.pth`, 43 MB; the model code is not in the notebook).
  - **Backbone:** a StrongUNet3D 3-level ResBlock UNet, channels (64,128,256), dropout 0.1, on the 64^3 XY/4-pooled volume. It outputs a node heatmap (peak kernel 3, threshold 0.2, at most 1,024 nodes) and sub-voxel `refine_zyx` / `refine_logit`.
  - **Linker:** a transformer over the pyramid node features (hidden 256, embed 128, 4 heads, 4 layers) that outputs pairwise `edge_logit` between t and t+1.
- **Inference:**
  - Frames are processed in 20-frame chunks under fp16 autocast. Duplicates are removed by greedy NMS at radius 3 pooled voxels (about 4.9 um), ordered by `refine_logit`.
  - The link score is `p_dst(j|i) x p_src(i|j)`: the forward softmax (with an explicit NO_LINK logit of 0) times the reverse softmax, renormalised. Candidates are kept up to 99 % cumulative mass.
  - Each source keeps its top-1 link, with destination conflicts resolved by the higher logit. There is **no ILP and no division output** (top-1 means no forks).
- **Training data (author's note):** "trained without Kaggle annotations. It just uses focus-3D instance segmentation centroids + augmentation to simulate cell movement. It only uses frame from t=0,5,10 ..., i.e. 20% of kaggle data."
- **Evidence:**
  - The title claims raw edge Jaccard 0.9 without ILP on the 11 `valid_id` train movies (the 4 visible movies plus 7 others), using `biohub_tracking.metrics.evaluate` at 7 um.
  - Detector table (from the notebook image, same 11 movies): recall 0.992-1.000 at p >= 0.2 and 0.944-1.000 at p >= 0.5, against sparse GT. Mean centre error 0.98-2.85 um. Predicted nodes are 1.05-2.36x the organiser's `estimated_number_of_nodes`.
  - Discussion threads 740145 and 738217 are referenced but I did not read them.
- **Why it matters:**
  - It is the only model in this batch that never saw Kaggle labels. Its local numbers are therefore honest on every train movie, unlike ours (in-sample).
  - It is an independent opinion whose errors should be less correlated with the pilkwang UNet's.
  - It is also the concrete form of the "external / pseudo-label data" gap, without us having to run FOCUS-3D (gated weights).
- **Limits:**
  - Its coordinates come from a 1.625 um grid, so only its link probabilities are useful to us, not its positions.
  - It cannot propose divisions.
  - A previous third-model tiebreak (andnyu synthetic edge) scored LB = base.
  - The dataset license is "unknown". Check the competition's external-data and model rules before any use.
  - Using it requires downloading the public dataset, which needs your approval.

### 10. hengck23/cell-point-detector - TRIED (partially, C002)
- **Method:** Inference only for the `StrongUNet3D3Level` detector (`model_v5.py`, `00000030.pth`). Max-pool peak picking, threshold 0.2, evaluation at p >= 0.5, GT matched within 7 um. It also has plotly visualisations.
- **Evidence:** The same image table as #9.
- **Our status:** We already hold these exact weights (`artifacts/hengck_point_detector/`). C002 cached 196,991 low-threshold peaks on the visible 4 movies, but the rescue analysis was never finished (`experiments/candidates/c002_node_rescue/README.md`).
- Given the node-budget results (added or readmitted nodes match GT only about 0.5 % of the time), adding its nodes is unattractive. Using it as independent **evidence** is the remaining angle (idea 6).

### 11. gautiermarti/biohub-deepcenter-unet3d (LB 0.945) - TRIVIAL-FORK
- **Code:** Harmonic-fusion code with config changes. The normalised diff against HF contains only the following substantive items.
  - **Config deltas versus x138:**

    | knob | gautiermarti | x138 |
    |---|---|---|
    | `GAP_CLOSE_UM` | 5.8 | 5.0 |
    | `DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.26 | 0.25 |
    | `DEEPCENTER_SAFE_DIV_VETO` | **0 (off)** | on |
    | `SECONDARY_EDGE_WEIGHT` | 0.20 | 0.15 |

    The secondary edge-feature TTA weight is 0.75 in both. The effective defaults differ, but both set it via env.
  - **DeepCenter:** heatmap TTA code removed (single view).
  - **Safe division rewritten:** a ball query of radius `SAFE_DIV_MAX_UM` around P, and the mutual-NN candidate must lie within the sister max. It also adds **a requirement that P has a predecessor** (`if source_id not in incoming: continue`), which x138 does not have. The `used_sources` guard and the veto counters were dropped.
  - **Removed:** the validator and ppsweep. It also has none of x138's additions (flow prior, readmit, gap filler, V1284 head).
- **Evidence (Russian markdown with version tables):**
  - Heuristic DoG and Hungarian: 0.808 -> 0.860.
  - HF fork from 0.923 to 0.934: proxy and LB moved in opposite directions in 4 of 8 steps. Tightening safe-division (12/15 -> 8/11 um plus "parent mid-track") gave "Div FP 2->1, LB +0.004 with no proxy change". Their reading is that false forks are punished more on the dense LB than on the sparse proxy.
  - v28 (safe-div 9/14, symmetry 0.6, DeepCenter epoch 2): 0.942.
  - v29/v30 (edge-feature TTA, tight 5.5, DC TTA): 0.945.
  - They quote an edge-feature TTA gain on 44b6_12dfb391 of 0.9115 -> 0.9256.
- **Unimplemented idea:** The v30 markdown plans "TTA-fusion for link logits: average the transformer's raw link logits over all 8 TTA views" (expected +0.002-0.003). The code only averages UNet **features** and then runs the transformer once, which is what x138 does. No notebook in this batch implements per-view transformer logits. See idea 4.

### 12. codezzzsleep/biohub-095-owned-validation - EXPLOIT
- **Pipeline:** A simplified single-seed pilkwang pipeline (hengck23-style).
  - Detection: 8-view D4 TTA on detection logits, threshold 0.97.
  - Transformer: fed the **identity-view features only** (`point_feature[0][:1]`), with `softmax(dim=0)` (over sources) and edge threshold 0.5.
  - ILP: edge -1, appearance 0, disappearance 1.4, division 1.0. No post-processing.
- The final cell appends the same hub plus 5 synthetic forks exploit as #7, taking the largest 1,400 components first. The title "095" presumably comes from that exploit. There is no public score in the radar.
- Nothing to take.

---

## Division-detection digest
- **Positives:** Nobody in this batch has more than the 151 GT divisions (199 movies, confirmed by zhincez and by our own count). The noisyislands trainers use the GT forks directly, which gives at most 151 positives. They avoid the base-rate problem only by using trivially easy random negatives: same-frame nodes (XGBoost) or any t+1 node (TabPFN). No notebook reports precision on detector-derived candidates. Our C016 negatives (real post-rule candidates, labels mirroring `score_divisions()`) are the harder and correct setting.
- **Division term in the scored notebooks:** In #7 and #12 it is gamed with fake forks. hengck23's linker produces no forks. gautiermarti reports that rule tightening, including "parent must have a predecessor", helped the LB (+0.004 in an older pipeline) while the proxy did not move.
- **Only new signal:** zhincez's nucleus size. It is a genuine per-cell division signature (about -15 % to -27 % volume from +1 to +3, brightness unchanged) that none of our 48 C016 features measures.

## Fine-tuning digest
- **noisyislands transformer fine-tune:** frozen UNet, transformer only, lr 2e-5, 6-8 epochs, random 80/20 movie split, GT windows, synthetic hard negatives. **No result published.**
- **xiaoleilian detector:** trained from scratch, embryo-grouped split with the 4 visible movies held out, 40-120 epochs. Val recall 0.58-0.67, far below pilkwang's.
- **hengck23 detector and linker:** trained on pseudo-labels (FOCUS-3D centroids, every 5th frame) with no Kaggle labels. He reports detector recall >= 0.99 and raw edge J 0.9 with no ILP.
- **Our own evidence on fine-tuning:** C008 (retrained UNet) lost on the LB because of feature-scale mismatch. The forum evidence (HANDOFF section 20: Pavel's edge-weight model +0.02-0.04 on folds and +0.000 LB, and hikaggler's division head +0.12 local divJ and +-0.001 LB) says that locally measured gains from learned components on in-sample movies mostly vanish on the LB.

---

## Ranked untried ideas

"Prior" is my estimate of the chance of a real LB gain. The team's own bar is at least +0.01 local on the 6bba proxy, confirmed on unseen movies. Most items below will probably not clear it. The list is ordered by expected value per hour given the 2026-09-29 deadline.

| rank | idea (source) | evidence behind it | effort | test locally? | prior | overfitting guard |
|---|---|---|---|---|---|---|
| 1 | **Nucleus size/shape features for the division scorer** (zhincez #6) | Paired-control bootstrap on 60 films: volume drops (about -0.27 at +3), brightness stays flat. Our C016 has intensity peak/mean only (3x7x7 window, confounded). | Pre-check 1-2 h: AUC of half-max volume ratios on the existing strict positives vs a negative sample from the candidate CSVs. Full 5-8 h: add features to `division_scorer_stage.py`, regenerate CSVs for 97 movies, retrain, `verify_c016.py --replay`. | Yes: candidate CSVs, trainer, harness replay | Low (needs a big jump from 6-23 % to >= 35 % per-parent precision; C016 failed on base rate) | Keep C016's pre-registered gate (>= 35 % OOF per-parent precision and >= 3 held-out TPs). Stop if the pre-check AUC is below about 0.85. |
| 2 | **Frozen-UNet transformer-only fine-tune, fixed recipe** (noisyislands #5; fills the "fine-tune transformer" gap and C016's "far-daughter P->D2" route) | No author result. Freezing the UNet removes C008's failure mode. Our diagnosis says 6 of 9 held-out misses need a model that gives far daughters P->D2 probability. | 16-24 h dev plus 4-8 h GPU. Reuse `src/train_local_unet.py` or the noisyislands script. Changes: freeze UNet and BN; train on **detector detections** matched to GT within 7 um (not GT positions); no positive labels on duplicates; oversample windows with GT divisions; exclude the held-out 12 and confirm-10 movies from fine-tuning. Evaluate with `src/run_kaggle_predict_local.py` using swapped weights (primary seed first). | Yes: local Kaggle-inference reproduction plus 6bba proxy | Low-moderate (highest ceiling; the base model is in-sample on every movie, and forum learned-model gains did not transfer) | Held-out and confirm movies never enter fine-tuning. Require majority-of-movies consistency. Watch division FP on annotated parents. |
| 3 | **hengck23 label-free linker as an independent third edge opinion** (#9) | Title claim: raw edge J 0.9 with no ILP on 11 train movies. Never saw Kaggle labels, so its local numbers are honest. Detector recall >= 0.99. | 8-12 h. Download the public dataset (`model_v12.py` plus `00000008.pth`, 43 MB; **needs your OK**; license "unknown"). Run it on the held-out 12 at 64^3. Match its nodes to ours within about 3 um. Add its `p_dst x p_src` as a relink-cost term or veto, and replay in the harness. | Yes: harness, if its probabilities are precomputed per node pair | Low (andnyu third-model tiebreak scored LB = base) | Fusion weight chosen on the held-out 12, confirmed on confirm-10. |
| 4 | **Transformer logit TTA over the 8 D4 views** (gautiermarti #11 markdown, never implemented anywhere) | Only the author's own expectation (+0.002-0.003). x138 averages features and runs the transformer once. | 4-8 h. Patch the predict loop to run the transformer per view with view-transformed coordinates and average the logits. The transformer cost is small next to the UNet. | Yes: local reproduction on the held-out 12 | Low (likely below the 0.01-local bar; ILP-input changes can flip single divisions, which is noise) | Same as above. Needs a 6bba proxy gain of at least 0.007 to be decidable. |
| 5 | **Two safe-division precision knobs** (gautiermarti #11): P must have a predecessor; DeepCenter safe-div veto off | gautiermarti's older pipeline: "parent mid-track" plus tighter radii gave Div FP 2->1, LB +0.004. The veto-off run scored 0.945 but is confounded with other knobs. | 1-2 h. The veto is a global (harness variant). The predecessor check needs a one-line patch in a test copy of the notebook. | Yes: harness on cached graphs | Very low (held-out division counts are about 3 TP / 2 FP; one event moves the 6bba total by about 0.008) | Only act if division FP drops without losing a TP on both the held-out 12 and confirm-10. |
| 6 | **Finish C002: StrongUNet peaks as independent evidence** (#10) | Recall >= 0.99 on sparse GT, detector independent of the pilkwang UNet, weights already local. | 4-6 h. Run it on the held-out 12 (only the visible 4 are cached). Recompute `det_t_at_d2` (our second-best division feature, AUC 0.83 after asymmetry at 0.86) and gap-fill confirmation from it instead of the same-UNet low-detection dump. | Yes | Low (improves one feature; node additions are unattractive per the node-budget analysis) | Movie-grouped CV as in C016. |
| 7 | **TabPFN as the division classifier** (#4) | None for this task. TabPFN is known to be strong with about 100 positives. | 2-3 h on the existing candidate CSVs (needs `pip install tabpfn` and Hugging Face weights; subsample negatives to fit the context size; check the license for Kaggle offline use). | Yes | Very low (the classifier is not the bottleneck) | C016 gate. |

Not worth pursuing:
- **noisyislands MLP linkers (#2, #3):** dominated by the transformer plus the flow-prior relink.
- **GT-only XGBoost (#1):** weaker than the closed C016.
- **xiaoleilian detector (#7, #8):** far below pilkwang's, and C008 already showed that local retraining does not transfer.
- **Fake hub/fork exploits (#7, #12):** excluded on purpose.
- **gautiermarti's other knobs:** gap 5.8 and secondary weight 0.20 were screened or are noise-level in our C015 screen.
- **hengck23's NO_LINK bidirectional product:** x138's harmonic fusion already does the bidirectional part.

Follow-ups that need your approval (not done here):
- Download `hengck23/hengck23-cell-point-detector-demo` (43 MB of new files) for idea 3.
- Optionally fetch the noisyislands kernel output logs. They would show whether the transformer fine-tune's val loss or AUC ever improved. Kernel outputs were out of scope for this audit.
