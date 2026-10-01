from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "r0_pack_full_train_gt"
OUT.mkdir(parents=True, exist_ok=True)

code = r'''
from __future__ import annotations

import json
import tarfile
from pathlib import Path

COMP = "biohub-cell-tracking-during-development"
candidates = [
    Path(f"/kaggle/input/competitions/{COMP}"),
    Path(f"/kaggle/input/{COMP}"),
]
comp_dir = next((p for p in candidates if p.exists()), None)
if comp_dir is None:
    raise FileNotFoundError(candidates)

train_dir = comp_dir / "train"
geffs = sorted(p for p in train_dir.iterdir() if p.name.endswith(".geff"))
if not geffs:
    raise RuntimeError(f"No .geff directories under {train_dir}")

out = Path("/kaggle/working/full_train_gt_geff.tar.gz")
manifest = []
with tarfile.open(out, "w:gz", compresslevel=6) as tf:
    for p in geffs:
        tf.add(p, arcname=p.name)
        manifest.append({
            "name": p.name,
            "files": sum(1 for q in p.rglob("*") if q.is_file()),
            "bytes": sum(q.stat().st_size for q in p.rglob("*") if q.is_file()),
        })

Path("/kaggle/working/full_train_gt_manifest.json").write_text(
    json.dumps({
        "count": len(geffs),
        "total_bytes": sum(x["bytes"] for x in manifest),
        "movies": manifest,
    }, indent=2),
    encoding="utf-8",
)
print("PACKED", len(geffs), "GEFF directories", "archive_bytes", out.stat().st_size)
'''

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path = OUT / "biohub-r0-pack-full-train-gt.ipynb"
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = {
    "id": "taeyangg4/biohub-r0-pack-full-train-gt",
    "title": "biohub-r0-pack-full-train-gt",
    "code_file": nb_path.name,
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
