from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "exp_train_inventory"
OUT.mkdir(parents=True, exist_ok=True)

code = r'''
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


COMPETITION = "biohub-cell-tracking-during-development"
COMP_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((p for p in COMP_CANDIDATES if p.exists()), COMP_CANDIDATES[0])
TRAIN_DIR = COMP_DIR / "train"
WORK = Path("/kaggle/working")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


rows = []
zarrs = sorted(TRAIN_DIR.glob("*.zarr"))
for image_dir in zarrs:
    stem = image_dir.name[:-5]
    gt_dir = TRAIN_DIR / f"{stem}.geff"
    image_meta = read_json(image_dir / "0" / "zarr.json")
    shape = list(image_meta.get("shape", []))
    node_meta_path = gt_dir / "nodes" / "ids" / "zarr.json"
    edge_meta_path = gt_dir / "edges" / "ids" / "zarr.json"
    node_meta = read_json(node_meta_path) if node_meta_path.exists() else {}
    edge_meta = read_json(edge_meta_path) if edge_meta_path.exists() else {}
    node_shape = list(node_meta.get("shape", [0]))
    edge_shape = list(edge_meta.get("shape", [0, 2]))
    t = int(shape[0]) if len(shape) >= 1 else 0
    nodes = int(node_shape[0]) if node_shape else 0
    edges = int(edge_shape[0]) if edge_shape else 0
    rows.append({
        "dataset": stem,
        "embryo": stem.split("_", 1)[0],
        "t": t,
        "z": int(shape[1]) if len(shape) > 1 else None,
        "y": int(shape[2]) if len(shape) > 2 else None,
        "x": int(shape[3]) if len(shape) > 3 else None,
        "gt_nodes": nodes,
        "gt_edges": edges,
        "gt_nodes_per_frame": nodes / max(t, 1),
        "gt_edge_node_ratio": edges / max(nodes, 1),
        "gt_exists": gt_dir.exists(),
    })

df = pd.DataFrame(rows).sort_values(["embryo", "dataset"]).reset_index(drop=True)
df.to_csv(WORK / "train_inventory.csv", index=False)
summary = {
    "n_movies": int(len(df)),
    "embryos": {
        str(k): {
            "movies": int(len(g)),
            "gt_nodes": int(g.gt_nodes.sum()),
            "gt_edges": int(g.gt_edges.sum()),
            "median_nodes_per_frame": float(g.gt_nodes_per_frame.median()),
            "min_nodes_per_frame": float(g.gt_nodes_per_frame.min()),
            "max_nodes_per_frame": float(g.gt_nodes_per_frame.max()),
        }
        for k, g in df.groupby("embryo", sort=True)
    },
}
(WORK / "train_inventory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print(df.head(10).to_string(index=False))
print(df.tail(10).to_string(index=False))
print("TRAIN_INVENTORY_COMPLETE")
'''

nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(OUT / "biohub-train-inventory.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": "taeyangg4/biohub-train-inventory",
    "title": "Biohub Train Inventory",
    "code_file": "biohub-train-inventory.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "kernel_sources": [],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("prepared", OUT)
