#!/usr/bin/env python3
"""Build script for Candidate C010: x138 Flow Fusion.

Fuses:
1. anvithpothula/biohub-x138 anchor notebook (public 0.953)
2. V1284_MODE='zero' (eliminates private biohub-v1284-head-s075 dataset dependency)
3. 3 core x138 algorithmic pillars:
   - Neighbourhood-flow motion prior (seed mode, K=12, radius=40um, tight=7um)
   - Discarded detection re-admission (radius=4um, min_score=0.965)
   - Low-detection gap filler (max_gap=3, min_score=0.5, synthetic=0)
   - Runtime hardening (validator=0, timeout=1200s, deadline=27000s)
4. C004 proven winning cytokinesis division parameters:
   - sister_max=16.0um, sister_symmetry_tau=0.85, diverge=0.5um, existing_child_max=12.0um, global_frac_cap=0.0050
5. C004 conservative short-track filter settings:
   - min_len=4, min_mean_edge_prob=0.88
6. Strict exclusion of local offline weights (zero 4070Ti weights)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NB_PATH = REPO_ROOT / "state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb"
DEST_DIR = REPO_ROOT / "experiments/candidates/c010_x138_flow_fusion"
DEST_NB_PATH = DEST_DIR / "biohub-c010-x138-flow-fusion.ipynb"


def build_c010() -> None:
    assert SOURCE_NB_PATH.exists(), f"Source notebook does not exist: {SOURCE_NB_PATH}"
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded source notebook: {len(nb['cells'])} cells")

    # 1. Update metadata
    nb["metadata"]["title"] = "biohub-c010-x138-flow-fusion"
    if "kaggle" in nb["metadata"]:
        nb["metadata"]["kaggle"]["dataSources"] = []

    # 2. Modify Cell 0
    cell0_src = "".join(nb["cells"][0]["source"])

    # Replace header
    header_target = "'''Biohub Harmonic Fusion\n\nProduction 3D lineage reconstruction with dual temporal models,\ndual edge-feature TTA, and geometry-validated divisions.\n\nRecord edition.'''"
    header_replacement = "'''Biohub C010: x138 Flow Fusion\n\nProduction 3D lineage reconstruction fusing x138 neighbourhood-flow motion prior,\ndiscarded detection re-admission, and low-detection gap filling with C004 proven\nwinning cytokinesis division geometry (sister=16.0um, child=12.0um, diverge=0.5um, tau=0.85).\n\nV1284_MODE='zero' eliminates private dataset dependency cleanly.\n'''"

    assert header_target in cell0_src, "Header target not found in Cell 0"
    cell0_src = cell0_src.replace(header_target, header_replacement)

    # Replace preset & axis
    cell0_src = cell0_src.replace(
        "BIOHUB_PRESET = 'harmonic_v3_division_wide'",
        "BIOHUB_PRESET = 'c010_x138_flow_fusion_c004_cytokinesis'",
    )
    cell0_src = cell0_src.replace(
        "BIOHUB_SCORE_AXIS = 'public 0.939 base + holdout-selected post-process configuration'",
        "BIOHUB_SCORE_AXIS = 'x138 flow prior + C004 calibrated cytokinesis'",
    )

    # Replace Cytokinesis division parameters with C004 winning values
    old_div_block = """os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"  


os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"  



os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"  
os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"  


os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375\""""

    new_div_block = """os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "16.0"
os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.85"
os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "0.5"
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "12.0"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.0050"
os.environ["V1284_MODE"] = "zero\""""

    assert old_div_block in cell0_src, "old_div_block not found in Cell 0"
    cell0_src = cell0_src.replace(old_div_block, new_div_block)

    # Verify AST of cell 0
    ast.parse(cell0_src)
    nb["cells"][0]["source"] = [line + "\n" for line in cell0_src.splitlines()]
    print("Cell 0 updated and verified.")

    # 3. Modify Cell 4: Remove private dataset check & set V1284_MODE='zero'
    cell4_src = "".join(nb["cells"][4]["source"])

    v1284_old_block = """_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))
if len(_myhead) != 1:
    raise RuntimeError(('my V1284 head mount mismatch', [str(p) for p in _myhead]))
# My own head, same architecture the module loads. Trained on x107's 20-movie TRAIN capture
# (4,136 detection<->GT pairs). Held-out BY MOVIE it moves centres CLOSER to truth:
#   ridge -10.8%   mlp -10.4%   POOLED, 16 of 20 movies improve.
# The 4-movie version of this head was +14.9% WORSE, so the null there was volume, not concept.
os.environ['V1284_MODE']='candidate'
os.environ['V1284_HEAD']=str(_myhead[0])"""

    v1284_new_block = """# C010: V1284_MODE='zero' disables private head shift and runs coordinate refinement in zero-shift mode,
# eliminating private dataset dependency while preserving clean execution.
os.environ['V1284_MODE'] = 'zero'"""

    assert v1284_old_block in cell4_src, "v1284_old_block not found in Cell 4"
    cell4_src = cell4_src.replace(v1284_old_block, v1284_new_block)

    # Verify AST of cell 4
    ast.parse(cell4_src)
    nb["cells"][4]["source"] = [line + "\n" for line in cell4_src.splitlines()]
    print("Cell 4 updated and verified.")

    # 4. Check all cells for AST syntax and no forbidden strings
    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        ast.parse(src)
        assert "biohub-local-4070ti-weights" not in src, f"Forbidden local weights found in cell {idx}"
        assert "best_local_unet_transformer.pth" not in src, f"Forbidden local weights file found in cell {idx}"
        assert "biohub-v1284-head-s075" not in src, f"Private dataset biohub-v1284-head-s075 found in cell {idx}"
        assert "v1284_head.pt" not in src, f"Private dataset weight v1284_head.pt found in cell {idx}"

    print("All 12 cells passed AST validation and exclusion checks.")

    # 5. Save notebook
    with open(DEST_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Saved Candidate C010 notebook to {DEST_NB_PATH}")


if __name__ == "__main__":
    build_c010()
