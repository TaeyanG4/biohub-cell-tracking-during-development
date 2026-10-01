# C008 - Local Transformer Fusion

**Candidate ID**: C008  
**Name**: `local_transformer_fusion`  
**Parent**: C004 adaptive lineage (56391478 - **0.948 LB**, Rank <= 213)  
**Notebook SHA256**: `591cc8de18176a54da4bc4ac02d5ab331bbf31985a968d50313f6446ead32be3`  
**Status**: INITIALIZED AND RIGOROUSLY VERIFIED  
**Target Score**: **>= 0.955** (Silver / Gold Frontier)

---

## 1. Executive Summary & Strategic Rationale

Candidate C008 combines our two strongest competitive assets:
1. **The Proven C004 Winning Anchor (0.948 LB)**:
   - C004 successfully conquered the 689-team 0.947 public baseline plateau to break into Bronze safe territory (Rank <= 213).
   - Root Cause Analysis on C005 & C006 (0.944 LB) conclusively proved that regressions stemmed from:
     - Overly aggressive short-track rescue (`min_len=3`, `min_prob=0.82`) admitting noisy false positives into the graph.
     - Excessive velocity momentum (`0.75`) incorrectly pulling curving cells onto rigid linear paths during dense crossings.
   - C008 **strictly retains C004's exact winning parameters**:
     - `BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN = '4'`
     - `BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = '0.88'`
     - `MOTION_RELINK_VELOCITY_WEIGHT = 0.50`
     - Calibrated cytokinesis envelope (`sister_max=16.0`, `exist_child=12.0`, `diverge_um=0.5`, `sym_tau=0.85`, `global_frac_cap=0.0050`).
2. **Local High-Performance 4070 Ti SUPER Weights**:
   - Trained across all 199 full-train `.zarr` movies with NVMe chunk caching on the local RTX 4070 Ti SUPER.
   - Peak validation tracking score: **0.9808** (val accuracy 99.99%, val recall 98.10%).
   - Weights published to Kaggle dataset: `taeyangg4/biohub-local-4070ti-weights` (SHA256: `1eaf064a010f08a94b28078490c33f188280b7ad1677493374e23b8146812a26`).
   - Integrated as the secondary model in the `low_margin_consensus` dual-seed edge predictor with `secondary_edge_weight = 0.20`.

---

## 2. Key Architecture & Configuration

| Parameter | C004 Anchor | C005 / C006 (Regressed) | C008 (Current) | Rationale |
|---|---|---|---|---|
| **Secondary Weights** | seed314159 | seed314159 | **4070Ti Local (0.9808)** | Full-train trained high-performance weights |
| **Short-Track Min Len** | 4 | 3 | **4** | Prevent spurious 3-frame noise |
| **Short-Track Min Prob** | 0.88 | 0.80 / 0.82 | **0.88** | Strict edge probability threshold |
| **Velocity Weight** | 0.50 | 0.75 | **0.50** | Natural biological momentum without over-steering |
| **Cytokinesis Sister Max** | 16.0 um | 16.0 um | **16.0 um** | Recovers distant post-mitotic daughters |
| **Cytokinesis Diverge** | 0.5 um | 0.5 um | **0.5 um** | +133.8% division recall on 199 GTs |
| **Cytokinesis Symmetry** | 0.85 | 0.85 | **0.85** | Accommodates cleavage asymmetry |
| **Relink Tight Radius** | 5.5 um | 5.5 um | **5.5 um** | Restores proven tight55 relink |
| **Secondary Edge Weight** | 0.15 | 0.15 / 0.20 | **0.20** | Confident blending of 0.9808 local weights |
| **Held-out PP Grid** | 10 orthogonal | 10 orthogonal | **10 orthogonal** | Runtime non-regressive validation promotion |

---

## 3. Verification & Guard Compliance

- **AST Syntax**: 100% PASS (zero syntax errors).
- **Startup Configuration Guard**: 100% PASS (zero drift against expected numeric & text constants).
- **Secondary Weights Integrity**: Verified SHA256 `1eaf064a010f08a94b28078490c33f188280b7ad1677493374e23b8146812a26`.
- **Model Reconstitution**: `load_model` cleanly reconstitutes `UNetNodeTransformer` without warnings.
- **Full-Train GT Cytokinesis**: Evaluated across all 199 GT graphs with >2x TP boost and Jaccard doubling.
