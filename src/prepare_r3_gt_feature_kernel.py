from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "r3_gt_fulltrain_featuregen"
OUT.mkdir(parents=True, exist_ok=True)
NB_PATH = OUT / "biohub-r3-gt-fulltrain-features.ipynb"
EXTRACTOR = ROOT / "src" / "extract_hoct_r3_compact.py"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


setup = r'''
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIRS = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((p for p in COMP_DIRS if p.exists()), COMP_DIRS[0])
TRAIN_DIR = COMP_DIR / "train"
WORKING_DIR = Path("/kaggle/working")

if not TRAIN_DIR.exists():
    raise FileNotFoundError(TRAIN_DIR)

wheels = glob.glob("/kaggle/input/**/hoct-0.2.0-py3-none-any.whl", recursive=True)
if not wheels:
    raise FileNotFoundError("HOCT wheel dataset not attached")
wheel_dir = os.path.dirname(wheels[0])
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", wheel_dir,
    "hoct==0.2.0", "spatial-graph==0.1.1", "pooch==1.9.0",
], check=True)

ckpts = list(Path("/kaggle/input").rglob("general_v1.pt"))
if len(ckpts) != 1:
    raise RuntimeError(f"Expected exactly one general_v1.pt, found {ckpts}")
HOCT_CKPT = ckpts[0]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("train_dir", TRAIN_DIR)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("hoct_checkpoint", HOCT_CKPT)
print("hoct_sha256", sha256(HOCT_CKPT))
'''


extractor_src = EXTRACTOR.read_text(encoding="utf-8")
extractor_src = extractor_src.split("\ndef main() -> None:", 1)[0]


driver = r'''
import hashlib as _r3_hashlib
import json as _r3_json
import numpy as _r3_np
import pandas as _r3_pd
import torch as _r3_torch
import zarr as _r3_zarr

R3_HARD_K = 8
R3_JITTER_STD_UM = _r3_np.asarray([1.76, 0.96, 1.00], dtype=_r3_np.float32)
R3_JITTER_MAX_UM = 6.0
R3_VARIANTS = ("clean", "jitter")
R3_DEVICE = "cuda" if _r3_torch.cuda.is_available() else "cpu"

_r3_model = _r3_torch.jit.load(str(HOCT_CKPT), map_location=R3_DEVICE).eval()
print("R3_DEVICE", R3_DEVICE)
_r3_w0 = _r3_model.head.weight.detach().cpu().numpy().reshape(-1).astype(_r3_np.float32)
_r3_b0 = _r3_model.head.bias.detach().cpu().numpy().reshape(-1).astype(_r3_np.float32)


def _r3_gt_nodes(dataset: str) -> _r3_pd.DataFrame:
    graph = load_gt(TRAIN_DIR / f"{dataset}.geff")
    attrs = graph.node_attrs().to_pandas()
    need = ["node_id", "t", "z", "y", "x"]
    missing = [c for c in need if c not in attrs.columns]
    if missing:
        raise RuntimeError(f"{dataset}: GT missing columns {missing}")
    out = attrs[need].copy()
    out["dataset"] = dataset
    out["row_type"] = "node"
    return out


def _r3_jitter(nodes: _r3_pd.DataFrame, dataset: str) -> _r3_pd.DataFrame:
    digest = _r3_hashlib.sha256(dataset.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = _r3_np.random.default_rng(seed)
    out = nodes.copy()
    n = len(out)
    noise_um = rng.normal(0.0, R3_JITTER_STD_UM, size=(n, 3)).astype(_r3_np.float32)
    norm = _r3_np.linalg.norm(noise_um, axis=1)
    over = norm > R3_JITTER_MAX_UM
    if over.any():
        noise_um[over] *= (R3_JITTER_MAX_UM / norm[over])[:, None]
    noise_px = noise_um / _r3_np.asarray(SCALE_ZYX, dtype=_r3_np.float32)
    out.loc[:, ["z", "y", "x"]] = out[["z", "y", "x"]].to_numpy(dtype=_r3_np.float32) + noise_px

    image = _r3_zarr.open_group(TRAIN_DIR / f"{dataset}.zarr", mode="r")["0"]
    _, zmax, ymax, xmax = image.shape
    out["z"] = out["z"].clip(0.0, float(zmax - 1))
    out["y"] = out["y"].clip(0.0, float(ymax - 1))
    out["x"] = out["x"].clip(0.0, float(xmax - 1))
    return out


_r3_gt_paths = sorted(TRAIN_DIR.glob("*.geff"))
_r3_datasets = [p.stem for p in _r3_gt_paths if (TRAIN_DIR / f"{p.stem}.zarr").exists()]
if not _r3_datasets:
    raise RuntimeError("No train dataset pairs found")
_r3_prefixes = sorted({x.split("_", 1)[0] for x in _r3_datasets})
print("R3_GT_DATASETS", len(_r3_datasets), "prefixes", _r3_prefixes)

_r3_global_stats = {
    "datasets": len(_r3_datasets),
    "prefixes": _r3_prefixes,
    "hard_k": R3_HARD_K,
    "variants": list(R3_VARIANTS),
    "jitter_std_um_zyx": R3_JITTER_STD_UM.tolist(),
    "jitter_max_um": R3_JITTER_MAX_UM,
    "checkpoint_sha256": sha256(HOCT_CKPT),
    "outputs": {},
}

for _r3_prefix in _r3_prefixes:
    _r3_names = [x for x in _r3_datasets if x.startswith(_r3_prefix + "_")]
    _r3_buckets = {k: [] for k in (
        "features", "labels", "groups", "edge_ids", "source_ids", "target_ids",
        "base_scores", "views", "datasets", "variants",
    )}
    _r3_stats = []

    for _r3_name in _r3_names:
        _r3_clean = _r3_gt_nodes(_r3_name)
        for _r3_variant in R3_VARIANTS:
            _r3_nodes = _r3_clean if _r3_variant == "clean" else _r3_jitter(_r3_clean, _r3_name)
            _x, _y, _g, _e, _s, _t, _b, _v, _st = extract_one(
                _r3_name,
                _r3_nodes,
                TRAIN_DIR,
                TRAIN_DIR,
                _r3_model,
                R3_HARD_K,
            )
            _vals = (
                _x, _y, _g, _e, _s, _t, _b, _v,
                _r3_np.full(len(_y), _r3_name, dtype="U64"),
                _r3_np.full(len(_y), _r3_variant, dtype="U8"),
            )
            for _key, _value in zip(_r3_buckets, _vals):
                _r3_buckets[_key].append(_value)
            _st = dict(_st)
            _st["variant"] = _r3_variant
            _r3_stats.append(_st)

    _r3_arrays = {k: _r3_np.concatenate(v, axis=0) for k, v in _r3_buckets.items()}
    _r3_out = WORKING_DIR / f"r3_gt_fulltrain_{_r3_prefix}.npz"
    _r3_np.savez_compressed(_r3_out, **_r3_arrays)
    _r3_stats_out = WORKING_DIR / f"r3_gt_fulltrain_{_r3_prefix}.stats.json"
    _r3_stats_out.write_text(_r3_json.dumps({
        "prefix": _r3_prefix,
        "movies": _r3_stats,
        "rows": int(len(_r3_arrays["labels"])),
        "feature_shape": list(_r3_arrays["features"].shape),
    }, indent=2), encoding="utf-8")
    _r3_global_stats["outputs"][_r3_prefix] = {
        "movies": len(_r3_names),
        "rows": int(len(_r3_arrays["labels"])),
        "feature_shape": list(_r3_arrays["features"].shape),
        "npz": _r3_out.name,
        "stats": _r3_stats_out.name,
    }
    print("R3_GT_SAVED", _r3_prefix, _r3_out, _r3_arrays["features"].shape)

_r3_np.savez_compressed(
    WORKING_DIR / "r3_gt_head_contract.npz",
    head_weight=_r3_w0,
    head_bias=_r3_b0,
)
(WORKING_DIR / "r3_gt_fulltrain_manifest.json").write_text(
    _r3_json.dumps(_r3_global_stats, indent=2), encoding="utf-8"
)
print(_r3_json.dumps(_r3_global_stats, indent=2))
'''


nb = {
    "cells": [code_cell(setup), code_cell(extractor_src), code_cell(driver)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for cell in nb["cells"]:
    source = "".join(cell["source"])
    compile(source, str(NB_PATH), "exec")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = {
    "id": "taeyangg4/biohub-r3-gt-full-train-features",
    "title": "Biohub R3 GT Full Train Features",
    "code_file": NB_PATH.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "sjlee101/biohub-hoct-020-wheels",
        "taeyangg4/biohub-hoct-general-v1-private",
    ],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "kernel_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

CPU_OUT = ROOT / "experiments" / "r3_gt_fulltrain_featuregen_cpu"
CPU_OUT.mkdir(parents=True, exist_ok=True)
cpu_nb = CPU_OUT / NB_PATH.name
cpu_nb.write_text(NB_PATH.read_text(encoding="utf-8"), encoding="utf-8")
cpu_meta = dict(meta)
cpu_meta["id"] = "taeyangg4/biohub-r3-gt-full-train-features-cpu"
cpu_meta["title"] = "Biohub R3 GT Full Train Features CPU"
cpu_meta["enable_gpu"] = False
cpu_meta.pop("machine_shape", None)
(CPU_OUT / "kernel-metadata.json").write_text(json.dumps(cpu_meta, indent=2), encoding="utf-8")

print("prepared", OUT)
print("prepared", CPU_OUT)
