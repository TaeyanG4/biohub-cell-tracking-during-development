#!/usr/bin/env python3
"""Verify Candidate C012 (C011 + our V1284-compatible head) before any push.

Checks:
  1. every code cell parses; only cell 0 labels and the cell 4 V1284-mode block differ
     from C011; the cell 0 environment is identical to C011 (= x138);
  2. the cell 4 block pins the SHA256 of the local head file and mounts it from the
     private dataset listed in kernel-metadata.json;
  3. the head file loads through x138's own embedded module with ``weights_only=True``
     in 'candidate' mode, and refine() returns finite coordinates whose displacement is
     below the module's 2 um guard for random feature maps (CPU and CUDA), with
     coordinates inside the volume and empty frames passed through;
  4. kernel-metadata.json: private, T4, internet off, pilkwang inputs + the head dataset.

    python src/verify_c012.py --head experiments/candidates/c012_v1284_head/dataset/v1284_head.pt
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from verify_c011 import embedded_v1284_module  # noqa: E402

from build_c012_candidate import CANDIDATES, DATASET_ID  # noqa: E402

C011_NB = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero" / "biohub-c011-x138-zero.ipynb"
ENV_ASSIGN = re.compile(r"""^os\.environ\[["']([A-Z0-9_]+)["']\]\s*=\s*(.+?)\s*$""", re.M)


def cells_of(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="c012")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--dataset-slug", default=DATASET_ID.split("/", 1)[1])
    parser.add_argument("--env", action="append", default=[], metavar="BIOHUB_KEY=VALUE",
                        help="intended cell 0 setting changes relative to C011 (everything else must match)")
    args = parser.parse_args()
    dataset_id = f"taeyangg4/{args.dataset_slug}"
    env_changes = dict(item.split("=", 1) for item in args.env)
    spec = CANDIDATES[args.candidate]
    cand_dir = REPO_ROOT / "experiments" / "candidates" / spec["dir"]
    cand_nb = cand_dir / f"{spec['slug']}.ipynb"
    c011, c012 = cells_of(C011_NB), cells_of(cand_nb)
    head_sha = hashlib.sha256(args.head.read_bytes()).hexdigest()

    assert len(c012) == len(c011) == 12
    for src in c012:
        ast.parse(src)
    for i in range(12):
        if i not in (0, 4):
            assert c012[i] == c011[i], f"cell {i} differs from C011"
    expected_env = [(k, f'"{env_changes[k]}"' if k in env_changes else v) for k, v in ENV_ASSIGN.findall(c011[0])]
    actual_env = [(k, v.split("#")[0].strip()) for k, v in ENV_ASSIGN.findall(c012[0])]
    expected_env = [(k, v.split("#")[0].strip()) for k, v in expected_env]
    assert actual_env == expected_env, "cell 0 environment differs from C011 beyond the declared --env changes"
    assert all(any(k == key for k, _ in actual_env) for key in env_changes), "declared --env key missing"
    added = [l[1:] for l in difflib.unified_diff(c011[4].splitlines(), c012[4].splitlines(), lineterm="", n=0)
             if l.startswith("+") and not l.startswith("+++")]
    assert f"if _head_sha256 != '{head_sha}':" in "\n".join(added), "cell 4 does not pin this head's SHA256"
    assert "os.environ['V1284_MODE'] = 'candidate'" in added
    print("PASS text: only labels, the V1284-mode block and the declared settings", env_changes or "{}",
          "differ from C011; head SHA256 pinned", head_sha[:16])

    import torch

    saved = torch.load(args.head, map_location="cpu", weights_only=True)
    assert set(saved) == {"state_dict", "mean", "scale"}, set(saved)
    assert saved["mean"].shape == (224,) and saved["scale"].shape == (224,)
    assert torch.isfinite(saved["mean"]).all() and (saved["scale"] > 0).all(), "scale must be positive and finite"
    module: dict = {"__name__": "v1284_coordinate_refinement"}
    exec(compile(embedded_v1284_module(c012[4]), "v1284_coordinate_refinement.py", "exec"), module)
    os.environ["V1284_MODE"] = "candidate"
    os.environ["V1284_HEAD"] = str(args.head)
    rng = np.random.default_rng(1)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        module["_CACHE"] = None
        feature = torch.randn(1, 32, 64, 64, 64, device=device) * 3.0
        arr = np.column_stack([np.full(2000, 5), rng.integers(0, 64, (2000, 3))]).astype(np.int16)
        arr[:3, 1:] = [[0, 0, 0], [63, 63, 63], [0, 63, 0]]
        with torch.no_grad():  # refine() only runs inside predict_video, which is @torch.no_grad()
            out = module["refine"](Path("x.zarr"), 5, arr, feature)
        shift_um = np.linalg.norm((out[:, 1:] - arr[:, 1:]) * 1.625, axis=1)
        assert np.isfinite(out).all() and out.dtype == np.float32 and out.shape == arr.shape
        assert shift_um.max() <= 2.00001 and (out[:, 1:] >= 0).all() and (out[:, 1:] <= 63).all()
        empty = np.empty((0, 4), np.int16)
        assert module["refine"](Path("x.zarr"), 5, empty, feature) is empty
        print(f"PASS head load+refine on {device}: weights_only load, mean shift {shift_um.mean():.3f} um, "
              f"max {shift_um.max():.3f} um (< 2 um guard), in-volume, finite")

    meta = json.loads((cand_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == f"taeyangg4/{spec['slug']}" and meta["code_file"] == cand_nb.name
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert dataset_id in meta["dataset_sources"]
    assert all(s.startswith("pilkwang/") or s == dataset_id for s in meta["dataset_sources"])
    print("PASS kernel-metadata:", meta["dataset_sources"])
    print(f"{args.candidate.upper()} verification: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
