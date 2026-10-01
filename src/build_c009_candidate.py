#!/usr/bin/env python3
"""Build Candidate C009: Boost Geometric Fusion.

Integrates:
1. Proven C004 Winning Anchor (0.948 LB, ref 56391478):
   - Natural velocity momentum 0.50
   - Conservative short-track filter min_len=4, prob=0.88
   - Full-train calibrated cytokinesis envelope (sister=16.0, exist=12.0, diverge=0.5, sym=0.85, cap=0.0050)
   - Tight relink 5.5
2. Newly Discovered /boost Geometric Innovations (Aman Atar 0.948 LB frontier):
   - Weak-leaf terminal pruning (prune_weak_leaf_nodes) to eliminate single-degree detector noise spikes
   - Per-embryo prefix stability guard (PPSWEEP_PREFIX_GUARD = 1) across held-out validation movies (44b6 / 6bba)
   - Resilient rank scan selection: first candidate passing all 3 guards ships (preventing runner-up blocking)
   - Extended orthogonal runtime validator grid with leaf030, leaf040, t55_leaf030, cytokinesis wide, and tight52
3. Local RTX 4070 Ti SUPER Weights Integration (from C008):
   - Dual-seed consensus blending with best_local_unet_transformer.pth (0.9808 val score)
   - Attached via Kaggle dataset taeyangg4/biohub-local-4070ti-weights
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C008_NOTEBOOK = REPO_ROOT / "experiments" / "candidates" / "c008_local_transformer_fusion" / "biohub-c008-local-transformer-fusion.ipynb"
C009_DIR = REPO_ROOT / "experiments" / "candidates" / "c009_boost_geometric_fusion"
C009_NOTEBOOK = C009_DIR / "biohub-c009-boost-geometric-fusion.ipynb"


def build_c009() -> None:
    print("Building Candidate C009: Boost Geometric Fusion...")
    C009_DIR.mkdir(parents=True, exist_ok=True)

    with open(C008_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Markdown Title Cell (Cell 0)
    nb["cells"][0]["source"] = [
        "# Biohub Cell Tracking: C009 Boost Geometric Fusion (0.955+ Frontier)\n",
        "\n",
        "Production-grade 3D cell tracking pipeline integrating:\n",
        "- **C004 Winning Anchor (0.948 LB, ref 56391478)**: Full-train calibrated cytokinesis envelope (`sister=16.0`, `exist=12.0`, `diverge=0.5`, `sym=0.85`, `cap=0.0050`), conservative short-track rescue (`min_len=4`, `prob=0.88`), natural velocity momentum (`0.50`), tight relink `5.5`.\n",
        "- **/boost Geometric Innovations (Aman Atar 0.948 LB)**: Weak-leaf terminal pruning (`prune_weak_leaf_nodes`, `LEAF_PRUNE_MIN_EDGE_PROB`) eliminating spurious detector spikes, per-embryo prefix stability guard (`PPSWEEP_PREFIX_GUARD=1`) preventing post-process overfit across 8 held-out validation movies (44b6/6bba), resilient multi-candidate guard scan, and extended orthogonal validator grid.\n",
        "- **Local RTX 4070 Ti SUPER Weights**: Dual-seed consensus blending with `best_local_unet_transformer.pth` (`0.9808` val tracking score) via low-margin consensus (`margin=0.35`, `weight=0.20`).\n"
    ]

    # 2. Update Code Cell (Cell 1)
    code = "".join(nb["cells"][1]["source"])

    # A. Update Preset and Score Axis
    code = code.replace(
        "BIOHUB_PRESET = 'c008_local_transformer_fusion'",
        "BIOHUB_PRESET = 'c009_boost_geometric_fusion'",
    )
    code = code.replace(
        "BIOHUB_SCORE_AXIS = '0.947 baseline -> 0.948 C004 adaptive lineage -> 0.950+ C008 local 4070Ti UNet transformer fusion (0.9808 val tracking score)'",
        "BIOHUB_SCORE_AXIS = '0.947 baseline -> 0.948 C004 anchor -> 0.955+ C009 boost geometric fusion (weak-leaf pruning + prefix-guard + local 4070Ti weights)'",
    )

    # B. Add Leaf Prune & Prefix Guard Environment Variables
    env_anchor = "os.environ['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] = '0.35'\n"
    c009_env_additions = (
        "os.environ['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] = '0.35'\n"
        "os.environ['BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB'] = '0.0'\n"
        "os.environ['BIOHUB_PPSWEEP_PREFIX_GUARD'] = '1'\n"
    )
    assert env_anchor in code, "env_anchor not found"
    code = code.replace(env_anchor, c009_env_additions, 1)

    # C. Update Expected Constants in Startup Guard
    old_exp_num = "'BIOHUB_SECONDARY_LOW_MARGIN_MAX': 0.35, 'BIOHUB_DUAL_SEED_EDGE_THRESHOLD': 0.48"
    new_exp_num = "'BIOHUB_SECONDARY_LOW_MARGIN_MAX': 0.35, 'BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB': 0.0, 'BIOHUB_DUAL_SEED_EDGE_THRESHOLD': 0.48"
    assert old_exp_num in code, "old_exp_num not found"
    code = code.replace(old_exp_num, new_exp_num, 1)

    old_exp_txt = "'BIOHUB_SECONDARY_LINK_MODE': 'low_margin_consensus'"
    new_exp_txt = "'BIOHUB_SECONDARY_LINK_MODE': 'low_margin_consensus', 'BIOHUB_PPSWEEP_PREFIX_GUARD': '1'"
    assert old_exp_txt in code, "old_exp_txt not found"
    code = code.replace(old_exp_txt, new_exp_txt, 1)

    # D. Define LEAF_PRUNE_MIN_EDGE_PROB and PPSWEEP_PREFIX_GUARD constants
    const_anchor = "SAFE_DIV_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_MAX_UM', '4.7'))\n"
    c009_const_additions = (
        "SAFE_DIV_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_MAX_UM', '4.7'))\n"
        "LEAF_PRUNE_MIN_EDGE_PROB = float(os.environ.get('BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB', '0.0'))\n"
        "PPSWEEP_PREFIX_GUARD = os.environ.get('BIOHUB_PPSWEEP_PREFIX_GUARD', '1') != '0'\n"
    )
    assert const_anchor in code, "const_anchor not found"
    code = code.replace(const_anchor, c009_const_additions, 1)

    # E. Define prune_weak_leaf_nodes function
    leaf_prune_func = '''
def prune_weak_leaf_nodes(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Remove terminal leaf nodes attached by a low-probability edge.

    A node is pruned only when ALL of the following hold:
      * it has no successor (out-degree 0) -- i.e. the tracker could not continue it;
      * it has exactly one incoming edge (never a division daughter ambiguity);
      * it does not sit on the final frame (terminal cells there are legitimate);
      * its single incoming edge HAS a learned probability and it is below LEAF_PRUNE_MIN_EDGE_PROB.
    Edges without a probability (edge_prob is None, e.g. safe-division additions)
    are exempt by construction, so division daughters added by add_safe_divisions_postlink
    can never be pruned here. Single pass, no cascade: conservative by design.
    """
    if LEAF_PRUNE_MIN_EDGE_PROB <= 0.0 or not nodes_by_id or not edges:
        return nodes_by_id, edges
    max_t = max(int(node["t"]) for node in nodes_by_id.values())
    out_degree: dict[int, int] = {}
    incoming: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        out_degree[source_id] = out_degree.get(source_id, 0) + 1
        incoming.setdefault(target_id, []).append(edge)
    drop: set[int] = set()
    for node_id, node in nodes_by_id.items():
        if out_degree.get(node_id, 0) != 0:
            continue
        if int(node["t"]) >= max_t:
            continue
        incoming_edges = incoming.get(node_id, [])
        if len(incoming_edges) != 1:
            continue
        prob = incoming_edges[0].get("edge_prob")
        if prob is None:
            continue
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(prob):
            continue
        if prob < LEAF_PRUNE_MIN_EDGE_PROB:
            drop.add(node_id)
    if not drop:
        return nodes_by_id, edges
    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id not in drop}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) not in drop and int(edge["target_id"]) not in drop
    ]
    stats["leaf_prune_nodes"] = len(drop)
    stats["leaf_prune_edges"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges
'''
    filter_func_anchor = "def filter_output_graph("
    assert filter_func_anchor in code, "filter_output_graph anchor not found"
    code = code.replace(filter_func_anchor, leaf_prune_func + "\n" + filter_func_anchor, 1)

    # F. Insert prune_weak_leaf_nodes call into filter_output_graph
    short_filter_call = "nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
    c009_filter_call = (
        "nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
        "    if LEAF_PRUNE_MIN_EDGE_PROB > 0.0:\n"
        "        nodes_by_id, edges = prune_weak_leaf_nodes(nodes_by_id, edges, stats)\n"
        "        if stats.get('leaf_prune_nodes'):\n"
        "            print(f'[{dataset}] after weak-leaf pruning: {len(nodes_by_id)} nodes, {len(edges)} edges (leaf_pruned = {stats[\"leaf_prune_nodes\"]})')\n"
    )
    assert short_filter_call in code, "short_filter_call not found in filter_output_graph"
    code = code.replace(short_filter_call, c009_filter_call, 1)

    # G. Add LEAF_PRUNE_MIN_EDGE_PROB to PP_SWEEP_KEYS
    sweep_keys_anchor = "PP_SWEEP_KEYS = ['SAFE_DIV_MAX_UM',"
    c009_sweep_keys = "PP_SWEEP_KEYS = ['LEAF_PRUNE_MIN_EDGE_PROB', 'SAFE_DIV_MAX_UM',"
    assert sweep_keys_anchor in code, "sweep_keys_anchor not found"
    code = code.replace(sweep_keys_anchor, c009_sweep_keys, 1)

    # H. Update score_validator_config to compute prefix_proxy
    summary_anchor = "summary['seconds'] = time.time() - t0\n"
    c009_prefix_summary = (
        "summary['seconds'] = time.time() - t0\n"
        "    prefix_rows: dict[str, list[dict]] = {}\n"
        "    for row in rows:\n"
        "        prefix_rows.setdefault(str(row['stem']).split('_')[0], []).append(row)\n"
        "    summary['prefix_proxy'] = {}\n"
        "    for prefix, prows in prefix_rows.items():\n"
        "        w = sum(r['weight'] for r in prows) or 1\n"
        "        p_tp = sum(r['div_tp'] for r in prows)\n"
        "        p_fp = sum(r['div_fp'] for r in prows)\n"
        "        p_fn = sum(r['div_fn'] for r in prows)\n"
        "        p_adj = sum(r['adjusted_edge_jaccard'] * r['weight'] for r in prows) / w\n"
        "        summary['prefix_proxy'][prefix] = (\n"
        "            p_adj + VALIDATOR_DIVISION_WEIGHT * edge_jaccard(p_tp, p_fp, p_fn)\n"
        "        )\n"
    )
    assert summary_anchor in code, "summary_anchor not found"
    code = code.replace(summary_anchor, c009_prefix_summary, 1)

    # I. Add _prefix_guard_ok and Update PP_CANDIDATES
    old_pp_candidates_section = code[code.find("PP_CANDIDATES: dict[str, dict] = {"):code.find("PP_SELECT_MARGIN = float(")]
    
    c009_pp_candidates_section = '''PP_CANDIDATES: dict[str, dict] = {
    'leaf030': {'LEAF_PRUNE_MIN_EDGE_PROB': 0.30},
    'leaf040': {'LEAF_PRUNE_MIN_EDGE_PROB': 0.40},
    't55_leaf030': {'MOTION_RELINK_TIGHT_UM': 5.5, 'LEAF_PRUNE_MIN_EDGE_PROB': 0.30},
    'div_envelope_wide': {'SAFE_DIV_MAX_UM': 11.0, 'SAFE_DIV_EXISTING_CHILD_MAX_UM': 12.0, 'SAFE_DIV_SISTER_MAX_UM': 16.0, 'SAFE_DIV_SISTER_SYMMETRY_TAU': 1.0, 'SAFE_DIV_DIVERGE_UM': 0.5},
    'div_zero_diverge': {'SAFE_DIV_DIVERGE_UM': 0.0},
    'div_base_strict': {'SAFE_DIV_MAX_UM': 9.0, 'SAFE_DIV_EXISTING_CHILD_MAX_UM': 10.0, 'SAFE_DIV_SISTER_MAX_UM': 14.0, 'SAFE_DIV_SISTER_SYMMETRY_TAU': 0.6, 'SAFE_DIV_DIVERGE_UM': 2.25, 'SAFE_DIV_GLOBAL_FRAC_CAP': 0.00375},
    'tight52': {'MOTION_RELINK_TIGHT_UM': 5.2},
    'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0},
    'gap45': {'GAP_CLOSE_UM': 4.5},
    'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}
}
'''
    assert len(old_pp_candidates_section) > 0, "old_pp_candidates_section not found"
    code = code.replace(old_pp_candidates_section, c009_pp_candidates_section, 1)

    # J. Update positive candidate filtering and selection loop with _prefix_guard_ok
    base_summary_anchor = "        base_summary = PP_RESULTS['base']\n"
    assert base_summary_anchor in code, "base_summary_anchor not found"
    
    c009_prefix_guard_func = (
        base_summary_anchor +
        "        def _prefix_guard_ok(summary: dict) -> bool:\n"
        "            if not PPSWEEP_PREFIX_GUARD:\n"
        "                return True\n"
        "            cand_pp = summary.get('prefix_proxy') or {}\n"
        "            base_pp = base_summary.get('prefix_proxy') or {}\n"
        "            for prefix, base_val in base_pp.items():\n"
        "                if cand_pp.get(prefix, float('-inf')) < base_val - 0.001:\n"
        "                    return False\n"
        "            return True\n"
    )
    code = code.replace(base_summary_anchor, c009_prefix_guard_func, 1)

    # Update positive list comprehension to include _prefix_guard_ok
    old_pos_pattern = "positive = [label for label, summary in PP_RESULTS.items() if label != 'base' and summary['proxy_score'] >= base_summary['proxy_score'] + 0.0005 and (summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS)]"
    new_pos_pattern = "positive = [label for label, summary in PP_RESULTS.items() if label != 'base' and summary['proxy_score'] >= base_summary['proxy_score'] + 0.0005 and (summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS) and _prefix_guard_ok(summary)]"
    assert old_pos_pattern in code, "old_pos_pattern not found"
    code = code.replace(old_pos_pattern, new_pos_pattern, 1)

    # Update selection loop to resiliently scan ranked candidates with _prefix_guard_ok
    old_sel_block = """        best_label, best_summary = ranked[0]

        if best_label != 'base' and best_summary['proxy_score'] >= base_summary['proxy_score'] + PP_SELECT_MARGIN and (best_summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS):
            selected_label = best_label
            selected_config = dict(PP_CANDIDATES[best_label])"""

    new_sel_block = """        selected_label = 'base'
        selected_config = {}
        for best_label, best_summary in ranked:
            if best_label == 'base':
                break
            if best_summary['proxy_score'] >= base_summary['proxy_score'] + PP_SELECT_MARGIN and (best_summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS) and _prefix_guard_ok(best_summary):
                selected_label = best_label
                selected_config = dict(PP_CANDIDATES[best_label])
                break"""
    assert old_sel_block in code, "old_sel_block not found"
    code = code.replace(old_sel_block, new_sel_block, 1)

    # K. Update Guard Report
    code = code.replace(
        "'experiment': 'c008_local_transformer_fusion_v1', 'status': 'verified_c008_local_fusion', 'parent_experiment': 'c004_adaptive_lineage_v1'",
        "'experiment': 'c009_boost_geometric_fusion_v1', 'status': 'verified_c009_boost_fusion', 'parent_experiment': 'c008_local_transformer_fusion_v1'",
    )

    # 3. Parse AST to verify 0 syntax errors
    print("Validating Python AST of generated C009 code...")
    ast.parse(code)
    print("AST verification: PASS (0 syntax errors).")

    # 4. Save C009 notebook
    nb["cells"][1]["source"] = [code]
    with open(C009_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"C009 notebook written to {C009_NOTEBOOK} (size: {C009_NOTEBOOK.stat().st_size} bytes)")

    # 5. Write kernel-metadata.json
    meta = {
        "id": "taeyangg4/biohub-c009-boost-geometric-fusion",
        "title": "biohub-c009-boost-geometric-fusion",
        "code_file": "biohub-c009-boost-geometric-fusion.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "taeyangg4/biohub-local-4070ti-weights"
        ],
        "kernel_sources": [],
        "competition_sources": [
            "biohub-cell-tracking-during-development"
        ],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"
    }
    meta_path = C009_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Kernel metadata written to {meta_path}")

    # 6. Write candidate.json
    nb_sha = hashlib.sha256(C009_NOTEBOOK.read_bytes()).hexdigest()
    cand = {
        "candidate_id": "C009",
        "name": "boost_geometric_fusion",
        "description": "C009 Boost Geometric Fusion: C004 winning anchor + local 4070Ti UNet transformer consensus + weak-leaf terminal pruning + per-embryo prefix stability guard.",
        "parent_id": "C008",
        "target_score": ">= 0.955 (Silver/Gold Frontier)",
        "candidate_sha256": nb_sha,
        "secondary_weights_sha256": "1eaf064a010f08a94b28078490c33f188280b7ad1677493374e23b8146812a26"
    }
    cand_path = C009_DIR / "candidate.json"
    cand_path.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    print(f"Candidate manifest written to {cand_path} (SHA256: {nb_sha[:16]}...)")

    # 7. Write README.md
    readme_content = f"""# Candidate C009: Boost Geometric Fusion

## Overview
- **Candidate ID**: C009
- **Name**: `boost_geometric_fusion`
- **Parent**: C008 Local Transformer Fusion / C004 Adaptive Lineage (0.948 LB anchor, ref 56391478)
- **Target Public LB**: **>= 0.955 (Silver/Gold Medal Frontier)**
- **Notebook SHA256**: `{nb_sha}`

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
"""
    (C009_DIR / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"README written to {C009_DIR / 'README.md'}")
    print("\n>>> Candidate C009 successfully built! <<<")


if __name__ == "__main__":
    build_c009()
