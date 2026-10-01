# Public-notebook audit B: association family (2026-09-24)

Scope: 12 assigned public notebooks, plus 4 arnav170 siblings and `yudaiyamauchi/lb-exploration-e-mutual-best` (the scored twin of E-association). Read-only: only notebook sources were pulled, into `state/notebook_radar/pulled/<author>__<slug>/`. Nothing was pushed, submitted or downloaded (no datasets, no weights).
Method: split each notebook into cells, hash every cell, and diff the cells that differ against x138 (`state/notebook_radar/pulled/biohub-x138`) and HF (`kaggle_notebooks/flexonafft_harmonic_fusion`). Then compare function and `BIOHUB_*` inventories, and read markdown and embedded payloads (newwang12 base64 runtime zip, indarkarhana overlay files).

Pull log: all 17 refs pulled OK. `seyitkaangunes/biohub-035-deconfounded-edge-stack` is byte-identical (SHA256 3cf813dd…) to the earlier radar copy. Older versions of `newwang12/biohub-v1-grouped` (v25–v32) return **403**, so its best-scoring (0.946) version is inaccessible. No notebook contains metric exploits: the `"t": -1` rows are the standard CSV edge placeholder, which x138 and HF also write.

## 0. Three facts that decide what can still help x138

1. **The admission gate makes logit tweaks weak.** The predict script (`tmp/c011_output/tracking_repo/scripts/predict_unet_transformer.py:849-889`) admits a candidate edge only if the per-target column softmax (over sources) gives p > 0.48. So a target has at most 2 admitted parents, and in practice about one. The local e2e cache shows 25,506 admitted edges for 26,533 detections (`experiments/candidates/c012_v1284_head/e2e/head_x138_s075/edge_cache/*.npz`, key `admitted`). The ILP therefore mainly decides drops and divisions, and x138's post-ILP motion relink re-decides association anyway. Every change to the logits before the softmax works as a per-target shift of the 0.48 admission gate.
2. **"Hard-negative margin" (yudai A/B) is only a temperature.** `raw - beta*(colmax - raw)` = `(1+beta)*raw - beta*colmax`, and `colmax` is constant within each softmax column. So the result is `softmax(raw/T)` with T = 1/(1+beta): A has T = 0.87 and B has T = 0.74. It is equivalent to the edge-threshold sweeps we already tried.
3. **Headroom for pure re-association is small.** On our C012 all-12 replay (`reports/pp_replay/e2e_head_v1_ppknobs_summary.csv`, `c012_as_is`) the GT edges break down as 7,610 recovered, 189 fragmented and 79 lost to detection. Only **1** predicted edge is a wrong association between two GT-matched nodes. Swap-fixing methods can only win on the 189 fragmented edges, and only where the predicted link was diverted to an unannotated or spurious node rather than broken. Each net fix (one FN and one FP removed) is worth about +0.00024 adjusted edge Jaccard (pooled estimate: 7,610 TP at J = 0.9293).

## 1. Per-notebook findings

| # | ref | LB | class | one-line verdict |
|---|---|---|---|---|
| 1 | yudaiyamauchi/lb-exploration-a-hard-negative-margin | 0.945 | TRIVIAL-FORK (= tried edge-threshold) | temperature 0.87 on the transformer logits |
| 2 | yudaiyamauchi/lb-exploration-b-hard-negative-strong | 0.944 | TRIVIAL-FORK (= tried) | temperature 0.74; LB falls further as the softmax gets sharper |
| 3 | yudaiyamauchi/lb-exploration-c-disagreement-adaptive | 0.946 | TRIVIAL-FORK (knob untried by us) | secondary-model `adaptive` blend at weight 0.30 |
| 4 | yudaiyamauchi/lb-exploration-d-relative-rank | 0.945 | TRIED (rank / mutual-best family) | bonus to the column-best / row-best edge, beta 0.12 |
| 5 | yudaiyamauchi/lb-exploration-e-mutual-best-association | none (re-run of E) | TRIED | code identical to `lb-exploration-e-mutual-best` (0.944) |
| 6 | andnyu/biohub-947-mutual-rank | 0.945 | TRIED | HF + the same rank bonus; −0.002 vs HF 0.947 |
| 7 | andnyu/biohub-synthetic-conditional-third-model | 0.946 | NEW-DATA / NEW-MODEL (partly TRIED) | swaps a few peaks for peaks from a detector trained on synthetic data, keeping node count fixed; edge part already tried |
| 8 | seyitkaangunes/biohub-035-deconfounded-edge-stack | 0.946 | TRIED | geometric fusion with weak-leaf prune 0.30 and velocity weight 0.25; −0.002 vs its 0.948 base |
| 9 | newwang12/biohub-v1-grouped | best 0.946 / current 0.940 | NEW-POSTPROCESS (no usable evidence) | per-20-frame "packet" classifier routes ILP and PP parameters; accessible version has empty overrides |
| 10 | indarkarhana/biohub-structured-trajectory-candidate | 0.946 | NEW-POSTPROCESS + NEW-MODEL | per-frame Hungarian re-assignment of 1:1 links with 18 features, using history *and future* motion; AR(2) endpoint repair |
| 11 | anhadmahajan06/biohub-track-your-cells | 0.946 | TRIVIAL-FORK | HF + in-kernel validator sweep over existing knobs |
| 12 | arnav170/biohub-reid3s (+reid3, retro, divdiag, steal2) | 0.947 (retro 0.946; others none) | NEW-MODEL (ReID) + NEW-POSTPROCESS (division variants) | appearance re-ID boosted-tree term in the relink cost; division steal/retro/NN-k |

### 1-5. yudaiyamauchi "LB exploration" A–E (one lineage)
- **Base:** "public0947-combined". This is tsukasamiyaji's 0.941 CONTROL (pilkwang_v2 lineage) plus the four HF config diffs: low_margin_consensus, DeepCenter TTA, primary/secondary edge-feature TTA (0.75), GAP_CLOSE 5.0, DC safe-div 0.25. It has no flow prior, readmit or gap filler. The base's own LB is not public.
- **What differs:** only code cells 0 and 7 differ between siblings. Cell 7 patches `predict_unet_transformer.py` at the anchor `raw = edge_logits_pair[0]`, which exists verbatim in our C011/x138 script. Env `BIOHUB_LB_SCORING_MODE` / `_BETA`:
  - A `hard_negative_margin`, beta 0.15, and B, beta 0.35: `raw -= beta*(colmax-raw)`, which is exactly the temperature change in §0.2.
  - D `relative_rank`, beta 0.12: +beta if the edge is the column (target) best, +0.5·beta if row (source) best, +0.5·beta if both.
  - E `mutual_best`, beta 0.25: D's bonus plus a −0.2·beta penalty on non-mutual edges. `e-mutual-best-association` has the same code (only `md_00` differs) and was re-run on 2026-09-23 without a score yet.
  - C: scoring mode `none`, plus `BIOHUB_SECONDARY_LINK_MODE=adaptive`, `SECONDARY_EDGE_WEIGHT=0.30`, `SECONDARY_LOW_MARGIN_MAX=0.50` (margin max is unused in adaptive mode). Adaptive sets the per-target secondary weight to clamp(w + margin_secondary − margin_primary, 0.15, 0.75), and to at least w when both models pick the same parent. x138 uses `low_margin_consensus`, w 0.15, margin 0.35.
- **Evidence:**
  - LB: A 0.945, B 0.944, C 0.946, D 0.945, E 0.944.
  - The author's markdown reports a 2-movie local A/B only for the base diffs, with no numbers for A–E.
  - The same rank bonus was submitted independently on other bases and was negative every time: andnyu on HF (0.945 vs 0.947), haideptry density-rank (0.944 vs density-adaptive 0.945), haideptry "0.951 SOTA" (0.944), haideptry mutual-best-density (0.943).
  - Per the md_01 diff table, 'adaptive' was CONTROL's own mode (it replaced low_margin_consensus at 0.940→0.941); both 0.947 forks kept low_margin_consensus. anhadmahajan06 calls adaptive "logit pollution".
- **Orthogonality to x138:** none. These act before the ILP (see §0.1) and are superseded by the relink.

### 6. andnyu/biohub-947-mutual-rank (0.945)
- **What differs:** HF plus one extra code block in cell 4, the D rank bonus with beta 0.12 (commented "copied from haideptry/biohub-0-948-sota-density-rank-2xt4-fast"). No other cell changes.
- **Evidence:** LB 0.945 vs HF 0.947 (−0.002).
- **Class:** TRIED (mutual-best / rank).

### 7. andnyu/biohub-synthetic-conditional-third-model (0.946)
- **Base:** haideptry `biohub-0-951-sota-deepcenter-fast-ilp-19m` (LB 0.944), i.e. HF with:
  - DivNet 3D mitosis veto (`BIOHUB_DIVNET_VERIFY=1`, `DIV_MIN_PROB=0.50`)
  - per-movie density-group relink overrides (groups by average nodes per frame <120 / <400 / ≥400; tight/relaxed/velocity/bonus per group)
  - tight 5.5, validator off
  - Both DivNet and density-adaptive association are already tried by us. andnyu **removed** haideptry's rank patch.
- **Added:** a third model, `bhpepper/biohub-synthetic-5fold-ensemble-v1` (`synthetic_5fold_swa.pth`, SHA 0eacacaf…), a synthetic-data SWA UNet+transformer.
  - (a) *Edge tiebreak*: blended only where the public top-2 margin is below 0.12 and the synthetic margin is larger by more than 0.02, with weight ≤ 0.25. This is the same code as `biohub-947-synthetic-edge`, which is TRIED (LB = base).
  - (b) **New: node-count-neutral detection swap** (`_budget_neutral_synthetic_swap`). Per frame, up to 0.5 % of base peaks are replaced. A candidate must be a synthetic-model peak (threshold det−0.03) that lies on a ≥4-frame synthetic tracklet (synthetic peaks at 0.90, greedy NN links ≤ 8.4 µm) and is more than 2.5 µm from every base peak. The public ensemble must also score it at ≥ det−0.03. It replaces the weakest base peak if its mean (public, synthetic) probability is at least 0.005 higher. Env: `BIOHUB_SYNTH_SWAP_FRAC=0.005`, `_NEAR_THRESHOLD_BAND=0.03`, `_SWAP_MIN_GAIN=0.005`, `_TRACKLET_MIN_LEN=4`, `_TRACKLET_THRESHOLD=0.90`, `_TRACKLET_MAX_STEP_UM=8.4`.
- **Evidence:** LB 0.946 vs andnyu's own density-adaptive reproduction 0.945 and haideptry density-adaptive 0.945. That is +0.001 at best, confounded by removing the rank patch. No local numbers.
- **Orthogonality to x138:** partial. x138 readmit and gap filler also recover sub-threshold peaks, but only from the same public detector. The swap brings a second, independently trained detector's opinion.

### 8. seyitkaangunes/biohub-035-deconfounded-edge-stack (0.946)
- **What differs:** every cell except cell 0 is byte-identical to `amanatar/biohub-geometric-fusion` (LB 0.948; also crystalbaby's copy). Cell 0 turns the validator off and hard-codes `MOTION_RELINK_TIGHT_UM=5.5`, `MOTION_RELINK_VELOCITY_WEIGHT=0.25` (default 0.5), `LEAF_PRUNE_MIN_EDGE_PROB=0.30`, `PPSWEEP_EXTENDED=0`, `PPSWEEP_PREFIX_GUARD=0`. It adds no new functions; `prune_weak_leaf_nodes` and `_prefix_guard_ok` come from amanatar.
- **Evidence:** 0.946 vs 0.948 for the validator-selected original, so "deconfounding" cost 0.002.
- **Class:** TRIED (weak-leaf prune, velocity weight).

### 9. newwang12/biohub-v1-grouped (best 0.946; current v33 0.940)
- **Core method** (embedded base64 runtime `biohub_{parameters,runtime,ilp,postprocess}.py`, SHA 6d3bd72c…):
  - Each movie is cut into 20-frame packets, and each packet gets 4 features from the standard PP output graph: nucleus XY half-max radius % change per link (`apparent_light_halfmax`, measured on the image), radius variance over 4 frames, nearest-neighbour distance, and neighbour-motion disagreement over 7-frame chains.
  - A quantile-normal plus linear-softmax classifier (`grouper_4feat_v7`) gives severity = p_mid + 2·p_high, cut into low / middle / high.
  - Optuna-tuned per-group overrides are then applied per frame to PP parameters (`FrameParameters(key, t)`) and to the ILP. `group_solve_ilp` rebuilds the tracksdata graph from exported pre-ILP candidates, with per-node appearance/disappearance/division weights and per-edge `ILP_EDGE_WEIGHT(t)·p`, and re-solves with SCIP.
  - The post-processing itself is HF's code; it adds no new functions.
- **What differs in the accessible version:** "ABASE_BOFF035" paired control: `GROUP_OVERRIDES` are all empty, `OUTPUT_GAP2_RECOVERY=False`, `DEEPCENTER_SAFE_DIV_THRESHOLD=0.35`. So the routing machinery runs but changes nothing.
- **Evidence:** the radar shows current 0.940 vs best 0.946. The best version, presumably with overrides, is 403-inaccessible, so no evidence for the routing is available.
- **Class:** NEW-POSTPROCESS, but related to the density-adaptive routing we already tried. The half-max radius function is a ready-made nucleus-size feature, a gap listed under "not tried".

### 10. indarkarhana/biohub-structured-trajectory-candidate (0.946)
- **Pipeline:** `portable-worker.py` (overlay embedded in the notebook) runs a pinned sjlee101 LF-DCTTA predictor and PP, then ILP and `filter_output_graph`. It then runs (1) `mixture.reconnect` and (2) `structured.refine`, and submits `repaired-prediction.json`.
  - (1) Per NOTICE.txt, "two source-trained robust AR(2) motion-support experts and a fixed reciprocal genuine-endpoint union". This reconnects track endpoints and was trained cross-embryo (`source-44b6-target-6bba.npz`, `source-6bba-target-44b6.npz`). **Its code lives only in the attached dataset `indarkarhana/biohub-trajectory-motion-runtime-v1` and was not inspected.**
  - (2) **Structured trajectory assignment**: fully visible in the overlay `structured-trajectory.py`, with weights in `structured-trajectory-model.json`. For every frame it takes the children that have exactly one consecutive parent; divisions and gap edges are protected and untouched. Candidate parents are the 16 nearest ILP-observed parents within 20 µm, plus the current and the ILP parent. Each pair gets 18 prediction-only features:
    - final and raw distances
    - residuals against half and full *history* velocity and half and full ***future*** velocity (the child's next step)
    - speeds and velocity disagreement, with known-flags
    - raw transformer probability and log-odds, plus a "was admitted" flag
    - an ILP-edge indicator
    - parent and child localization shift (final vs raw position)
  - A fixed linear score is used. The largest weights are initial_edge +1.22, prob_known +1.21, log-odds +0.60, prob +0.50 and velocity_disagreement +0.22; the distance and residual terms are −0.27 to −0.35. A per-frame Hungarian over the same children×parents set (capacity-preserving permutation) is accepted only if the total cost improves. Degrees, positions and divisions are unchanged.
- **Evidence:**
  - Author's NOTICE: the fixed model "improved independent ten-movie confirmation and separate eight-movie validation without movie regressions"; "0.9453907 local diagnostic is not an LB score".
  - LB 0.946 vs LF-DCTTA 0.947. That 0.947 was likely validator-selected (tight 5.5); this runtime executes "no proxy/GT/search cells", so the effective base is probably the validator-off 0.946 configuration, making the net LB effect about 0.
- **Orthogonality to x138:** partial. x138's relink is forward-greedy, frame by frame, with cost = flow-predicted motion − 1.0·prob. It never uses the child's future step, the ILP membership or localization shifts, and it runs *before* gap filling, readmit, divisions and linefit. The structured pass runs *after* them. Headroom is capped by §0.3.
- **Harness fit:** our e2e runs store exactly the inputs it needs, with no re-inference: `coords` (t,z,y,x per ILP node id) and `admitted` (src, tgt, prob, dist) in `edge_cache/<stem>.npz`, plus the ILP `predictions/<stem>.geff`.

### 11. anhadmahajan06/biohub-track-your-cells (0.946)
- **What differs:** HF plus `VALIDATOR_ENABLE=1`, `PPSWEEP_SELECT_MARGIN=0.0008`, explicit tight 5.5 / relaxed 10, and a 14-candidate sweep with greedy stacking and a division-FP guard. Candidates: tight 5.2/5.3/5.4, SAFE_DIV_MAX 8.0/8.5, DC div 0.35, DIVERGE 2.5, GAP2 step 3.5 / total 8.0, bonus 1.15, GAP_CLOSE 4.8. No new functions.
- **Evidence:** the markdown targets "0.948–0.950+", but the LB is 0.946. Earlier versions scored 0.940 and 0.943.
- **Class:** TRIVIAL-FORK. The knobs are covered by C004's validator combo (tight52) and our `variants_pp_knobs.json` replay.

### 12. arnav170 family: biohub-reid3s (0.947), reid3 (none), retro (0.946), divdiag (none), steal2 (none)
- **Code base:** all are HF with identical cell 0. New code sits in cells 2, 5, 9 and 10, with all new knobs defaulting to off:
  - **ReID appearance model** (`reid_descriptors_for_frame`, `reid_pair_features`, `REID_WEIGHT`, `REID_TIGHT_ONLY`):
    - Descriptor: a 28-d handcrafted vector from a 7×25×25-voxel cube (about ±4.9 µm). It covers intensity (6), second-moment shape (6), radial profile at 0–8 µm (8), context (depth, density, NN1, NN2), DeepCenter heat (1) and size (3).
    - Pair features: 6 block distances, 2 cosines, raw/motion distance, transformer prob, log volume and peak ratios, density, rank, gap and gated count.
    - Model: HistGradientBoosting (300 iterations, lr 0.05). Positives are GT links mapped to raw detections; negatives are all other raw detections inside the relaxed gate. Trained in-kernel on the validator's 8 train movies, leave-one-stem-out for validator scoring and on all 8 for test.
    - In `motion_relink_edges`: `cost -= REID_WEIGHT * P(same cell)` for every gated pair, not only transformer-admitted ones.
  - **Division variants in `add_safe_divisions_postlink`:** `SAFE_DIV_STEAL_OWNER_UM` / `_RATIO` / `_REATTACH` take a 2nd daughter already owned by another track; `SAFE_DIV_RETRO_MAX_BACK` looks for the division earlier along the owner chain; `SAFE_DIV_MUTUAL_NN_K`; `SAFE_DIV_RAW_SUCC_FALLBACK`.
  - **Unused hook:** `DEEPCENTER_NODE_VETO_THRESHOLD` (off in every sweep).
- **reid3s (0.947):** the sweep is skipped ("hidden-test rerun timed out at 12h") and the config is **forced** to `REID_WEIGHT=8, MOTION_RELINK_TIGHT_UM=5.5, MOTION_RELINK_RELAXED_UM=9.0`. The code comment says "measured in biohub-reid3 v1: adj_edge 0.9311 vs 0.9280 tight55" (+0.0031, out-of-fold by stem).
- **retro (0.946):** validator-selected retro/NN-k candidates. The author notes in reid3s: "12 holdout divisions are too noisy (nnk2 +0.0077 proxy -> -0.001 LB)", and switched to edge-first selection.
- **steal2, divdiag, reid3:** no LB.
- **Evidence summary:** ReID gave +0.003 adjusted edge Jaccard locally but no LB gain (0.947 = HF). Division variants were negative on LB.
- **Class:** NEW-MODEL (ReID) and NEW-POSTPROCESS (division variants).
- **Already tried by us:** all six trivial candidates in the retro/steal2/divdiag sweeps (gap45, relaxed9, bonus125, gap2step40, reuse28, dcgap035) are in our `reports/pp_replay/variants_pp_knobs.json`. On all 12 movies (C012 e2e) relaxed9 gave +0.00015 adjusted edge Jaccard and bonus125 −0.0003; the rest were about 0.

## 2. Ranked untried ideas that could stack on x138

| rank | idea (source) | why it might add to x138 | effort | local test |
|---|---|---|---|---|
| 0 | **Gate check (do first):** split our 189 "fragmented" GT edges into *broken* (source has no outgoing link) vs *diverted* (source linked to an unmatched node), per 44b6/6bba | Decides whether ideas 1–2 (diverted links) or 3 (broken links) have any headroom (§0.3) | 0.5 h | harness outputs only |
| 1 | **Structured two-sided re-assignment** (indarkarhana `structured-trajectory.py`, Apache-2.0, 18 published weights) as the last PP step | Uses future-step residuals, ILP membership and localization shift, which x138's forward flow relink lacks; runs after readmit, gap fill and linefit; preserves degrees and divisions | 3 h port with published weights; +4–6 h to refit the 18 weights with movie-grouped CV | **Yes, harness post-hook.** No re-inference: the e2e `edge_cache` already has `coords` + `admitted` |
| 2 | **Appearance ReID term in the relink cost** (arnav170 reid3s: 28-d cube descriptors + boosted-tree classifier, cost −= 8·P) | Appearance is a signal x138 does not use anywhere; complementary to the flow prior; author measured +0.0031 adjusted edge Jaccard out-of-fold | 8–10 h: port descriptors, build GT pair set, leave-one-movie-out training, harness sweep of weight 2/4/8 | **Yes, harness** (reads `data/train` frames and the DeepCenter gate). Caution: the transformer-probability feature is over-confident on in-sample train movies; train and score movie-grouped |
| 3 | **AR(2) endpoint-union repair** (indarkarhana `mixture.reconnect`) | Targets broken tracks, the larger error bucket, with learned cross-embryo motion experts | unknown: code is only in the dataset; inspecting it needs your OK to fetch that dataset's `.py` files | harness post-hook once the code is visible |
| 4 | **Node-count-neutral synthetic-detector swap** (andnyu) | A second, independently trained detector for near-threshold peaks; keeps node count fixed | 5–6 h plus downloading `bhpepper/biohub-synthetic-5fold-ensemble-v1` (needs approval) | **No:** ~25–40 min local re-inference (3rd model) |
| 5 | Secondary `adaptive` blend, w 0.30 (yudai C) | Cheap; env-only | 1 h | ~25 min re-inference: `run_kaggle_predict_local.py --env BIOHUB_SECONDARY_LINK_MODE=adaptive --env BIOHUB_SECONDARY_EDGE_WEIGHT=0.30` |
| 6 | Per-packet severity routing (newwang12), or just its half-max radius as a nucleus-size feature | Time-local parameter routing; the radius function could feed ReID or the (closed) division scorer | 10 h+ (routing needs retraining a classifier + Optuna) | PP part: harness; ILP part: offline ILP re-solve from `admitted` |

**Skip (evidence says no):**
- Temperature, rank and mutual-best logit transforms: 8 LB submissions on 3 different bases, all 0.943–0.945 (yudai A, B, D, E; andnyu on HF; three haideptry variants). None beat its base where the base score is known. They are also pre-ILP threshold shifts (§0.1–0.2).
- 035's leaf-prune and velocity 0.25: tried, and −0.002 on LB.
- anhadmahajan06's and arnav170's knob sweeps: already replayed.
- Division steal / retro / NN-k: −0.001 on LB after a +0.0077 proxy gain, and division headroom is closed per our division-scorer study.

Expected upside is small for every item. The best public result from any of these methods is 0.947, equal to the HF base. Idea 1 is the only one testable today without training or re-inference, so it is the cheapest to accept or reject. Idea 2 is the strongest independent local signal (+0.003 out-of-fold adjusted edge Jaccard), but it had no LB effect on HF.
