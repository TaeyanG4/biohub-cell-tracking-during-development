# Candidate C009: Boost Geometric Fusion

## Overview
- **Candidate ID**: C009
- **Name**: `boost_geometric_fusion`
- **Parent**: C008 Local Transformer Fusion / C004 Adaptive Lineage (0.948 LB anchor, ref 56391478)
- **Target Public LB**: **>= 0.955 (Silver/Gold Medal Frontier)**
- **Notebook SHA256**: `5feeb3fc796b7d51209f32bd6af8950c23939c55a947b02e88f8aaa09188401b`

## Integrated Enhancements
1. **Proven C004 Winning Anchor**:
   - `MOTION_RELINK_VELOCITY_WEIGHT = 0.50` (natural velocity momentum avoiding C005/C006/C007 over-steering)
   - `BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN = '4'`, `BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = '0.88'` (noise-free rescue)
   - Full-train cytokinesis envelope (`sister_max=16.0`, `exist=12.0`, `diverge=0.5`, `sym=0.85`, `cap=0.0050`)
   - `BIOHUB_MOTION_RELINK_TIGHT_UM = '5.5'`
2. **/boost Geometric Innovations (Aman Atar 0.948 LB Frontier)**:
   - **Weak-Leaf Terminal Pruning** (`prune_weak_leaf_nodes`): single-pass terminal leaf node pruning for edges with probability `< LEAF_PRUNE_MIN_EDGE_PROB` (sweeping `0.30` and `0.40`). Completely exempts cytokinesis/safe-division daughter edges (`edge_prob is None`).
   - **Per-Embryo Prefix Stability Guard** (`PPSWEEP_PREFIX_GUARD = 1`): rejects post-processing candidates that regress by > 0.001 on ANY embryo prefix (`44b6` or `6bba`), preventing validation set overfitting.
   - **Extended Orthogonal Validator Grid**: includes `leaf030`, `leaf040`, `t55_leaf030`, `div_envelope_wide`, `div_zero_diverge`, `tight52`.
3. **Local RTX 4070 Ti SUPER Weights**:
   - Checkpoint `best_local_unet_transformer.pth` (0.9808 val score) fused via dual-seed `low_margin_consensus` (`margin=0.35`, `weight=0.20`).
