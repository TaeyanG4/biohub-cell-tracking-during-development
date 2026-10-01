#!/usr/bin/env python3
"""Verify Candidate C011 (x138 with V1284_MODE='zero') against its x138 parent.

Checks:
  1. every code cell parses;
  2. only the intended text changed: cells other than 0 and 4 are byte-identical
     to x138, every os.environ assignment in cell 0 is identical, and cell 4 differs
     only in the private-head block;
  3. no private-dataset or local-weight references remain;
  4. the V1284 module embedded in cell 4 is a pass-through in 'zero' mode and its
     trilinear feature lookup equals the support pack's native integer gather at
     integer coordinates (exact tensor equality, CPU and CUDA when available);
  5. kernel-metadata.json is private, GPU T4, internet off, public inputs only;
  6. with --replay: post-processing replayed on the 12 cached C004 ILP graphs
     (src/eval_pp_variants_local.py) is identical, movie by movie, to x138's.

Run with the global Python (torch + tracksdata): python src/verify_c011.py [--replay]
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
X138_NB = REPO_ROOT / "state" / "notebook_radar" / "pulled" / "biohub-x138" / "biohub-x138.ipynb"
C011_DIR = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero"
C011_NB = C011_DIR / "biohub-c011-x138-zero.ipynb"
REPLAY_REFERENCE = REPO_ROOT / "reports" / "pp_replay" / "x138_calib_div_all12.csv"
SUPPORT_TRAIN_SCRIPT = REPO_ROOT / "artifacts" / "pilkwang_support50" / "repo" / "scripts" / "train_unet_transformer.py"
FORBIDDEN = ("biohub-v1284-head-s075", "v1284_head.pt", "biohub-local-4070ti-weights", "best_local_unet_transformer.pth")
ENV_ASSIGN = re.compile(r"""^os\.environ\[["']([A-Z0-9_]+)["']\]\s*=\s*(.+?)\s*$""", re.M)


def cells_of(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def check_text(x138: list[str], c011: list[str]) -> None:
    assert len(c011) == len(x138) == 12, (len(x138), len(c011))
    for index, src in enumerate(c011):
        ast.parse(src)
        for token in FORBIDDEN:
            assert token not in src, f"cell {index} references {token}"
    for index in range(12):
        if index not in (0, 4):
            assert c011[index] == x138[index], f"cell {index} differs from x138"
    assert ENV_ASSIGN.findall(c011[0]) == ENV_ASSIGN.findall(x138[0]), "cell 0 environment drifted from x138"
    changed = [line for line in difflib.unified_diff(x138[4].splitlines(), c011[4].splitlines(), lineterm="", n=0)
               if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    removed = [line[1:] for line in changed if line.startswith("-")]
    added = [line[1:] for line in changed if line.startswith("+")]
    assert any("_myhead = sorted(" in line for line in removed), removed
    assert all(line.startswith("#") or line.startswith("os.environ['V1284_MODE'] = 'zero'") for line in added), added
    assert len(removed) == 9 and len(added) == 5, (len(removed), len(added))
    print("PASS text: 12 cells parse; only cell 0 labels and the cell 4 head block differ from x138; "
          "cell 0 environment identical; no private references")


def embedded_v1284_module(cell4: str) -> str:
    for node in ast.walk(ast.parse(cell4)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
                and "v1284_coordinate_refinement.py" in ast.unparse(node.func.value)):
            return node.args[0].value
    raise AssertionError("v1284_coordinate_refinement.py source not found in cell 4")


def native_index_features(feat_maps, coords, mask):
    """The support pack's UNetNodeTransformer._index_features (checked below)."""
    import torch

    B, C = feat_maps.shape[:2]
    spatial = feat_maps.shape[2:]
    out = torch.zeros(B, coords.shape[1], C, device=feat_maps.device, dtype=feat_maps.dtype)
    for b in range(B):
        nt = int(mask[b].sum().item())
        if nt == 0:
            continue
        z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
        y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
        x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
        out[b, :nt] = feat_maps[b, :, z, y, x].T
    return out


def check_v1284_zero(cell4: str) -> None:
    import torch

    native_src = SUPPORT_TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "out[b, :nt] = feat_maps[b, :, z, y, x].T" in native_src, "support-pack native gather changed"

    module: dict = {"__name__": "v1284_coordinate_refinement"}
    exec(compile(embedded_v1284_module(cell4), "v1284_coordinate_refinement.py", "exec"), module)
    os.environ["V1284_MODE"] = "zero"
    rng = np.random.default_rng(0)
    arr = np.column_stack([np.full(500, 7), rng.integers(0, 16, 500), rng.integers(0, 64, (500, 2))]).astype(np.int16)
    feature = torch.zeros(1, 32, 16, 64, 64)
    out = module["refine"](Path("x.zarr"), 7, arr, feature)
    assert out.dtype == np.float32 and np.array_equal(out, arr.astype(np.float32)), "zero mode moved coordinates"
    empty = np.empty((0, 4), dtype=np.int16)
    assert module["refine"](Path("x.zarr"), 7, empty, feature) is empty

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float32, torch.float16):
            maps = torch.randn(2, 32, 16, 64, 64, device=device).to(dtype)
            coords = torch.tensor(np.stack([
                np.column_stack([rng.integers(0, 16, 300), rng.integers(0, 64, 300), rng.integers(0, 64, 300)])
                for _ in range(2)]), dtype=torch.float32, device=device)
            coords[:, :4] = torch.tensor([[0, 0, 0], [15, 63, 63], [0, 63, 0], [15, 0, 63]], dtype=torch.float32, device=device)
            mask = torch.ones(2, 300, dtype=torch.bool, device=device)
            mask[1, 250:] = False
            got = module["index_features"](None, maps, coords, mask)
            want = native_index_features(maps, coords, mask)
            assert torch.equal(got, want), f"trilinear lookup != native gather on {device}/{dtype}"
    print(f"PASS v1284 zero mode: refine() is a pass-through; trilinear lookup == native gather "
          f"({', '.join(devices)}; fp32/fp16; boundary coordinates; masked slots)")


def check_metadata() -> None:
    meta = json.loads((C011_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "taeyangg4/biohub-c011-x138-zero" and meta["code_file"] == C011_NB.name
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert sorted(meta["dataset_sources"]) == [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ], meta["dataset_sources"]
    assert meta["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert not meta["kernel_sources"] and not meta["model_sources"]
    print("PASS kernel-metadata: private, T4 GPU, internet off, 3 public pilkwang datasets + competition data")


def check_replay() -> None:
    import pandas as pd

    variants = REPO_ROOT / "reports" / "pp_replay" / "variants_c011_as_is.json"
    variants.write_text(json.dumps({"x138": {}}) + "\n", encoding="utf-8")
    out = REPO_ROOT / "reports" / "pp_replay" / "c011_as_is_all12.csv"
    subprocess.run([sys.executable, str(REPO_ROOT / "src" / "eval_pp_variants_local.py"),
                    "--notebook", str(C011_NB), "--variants", str(variants), "--stems", "all12", "--out", str(out)],
                   check=True, cwd=REPO_ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    ref = pd.read_csv(REPLAY_REFERENCE)
    ref = ref[ref["config"] == "x138"].set_index("stem")
    got = pd.read_csv(out).set_index("stem")
    cols = ["edge_tp", "edge_fp", "edge_fn", "t_pred", "div_tp", "div_fp", "div_fn", "nodes", "edges", "adjusted_edge_jaccard"]
    assert list(got.index) == list(ref.index)
    diff = (got[cols] - ref[cols]).abs().to_numpy().max()
    assert diff == 0, f"replay differs from x138 (max abs diff {diff})"
    print("PASS replay: C011 post-processing is identical to x138 on all 12 cached movies")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true", help="also replay post-processing on the 12 cached movies (~4 min)")
    args = parser.parse_args()
    x138, c011 = cells_of(X138_NB), cells_of(C011_NB)
    check_text(x138, c011)
    check_v1284_zero(c011[4])
    check_metadata()
    if args.replay:
        check_replay()
    print("C011 verification: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
