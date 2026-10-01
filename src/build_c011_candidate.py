#!/usr/bin/env python3
"""Build Candidate C011: x138 with the private V1284 head switched off.

Parent: anvithpothula/biohub-x138 (public 0.953, kernel v1, submitted 2026-09-21).

The only functional change is in code cell 4: the lookup of the author's private
``biohub-v1284-head-s075/v1284_head.pt`` (which raises on any fork) is replaced by
``V1284_MODE='zero'``, a mode the author's own refinement module implements
(``refine`` returns the detector coordinates unchanged; the trilinear feature
lookup reproduces the native integer gather exactly at integer coordinates).
Everything else - flow prior, readmit, low-detection gap filler, runtime guards and
x138's own division geometry (sister 14 / tau 0.6 / diverge 2.25 / child 10 /
cap 0.00375) - is byte-identical to x138.

Unlike C010 this does NOT import C004's wide division envelope: C004's own
validator rejected it (proxy 0.9359 vs 0.9511 strict, reproduced exactly by
src/eval_pp_variants_local.py), and on x138's post-processing it costs -0.0070
(0.9424 -> 0.9354 on 12 replayed movies; division FP 3 -> 16, TP unchanged).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NB = REPO_ROOT / "state" / "notebook_radar" / "pulled" / "biohub-x138" / "biohub-x138.ipynb"
SOURCE_SHA256 = "6b655e39bbfd2d3d"  # prefix of the pulled x138 v1 notebook
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero"
DEST_NB = DEST_DIR / "biohub-c011-x138-zero.ipynb"
KERNEL_ID = "taeyangg4/biohub-c011-x138-zero"

HEADER_OLD = (
    "'''Biohub Harmonic Fusion\n\n"
    "Production 3D lineage reconstruction with dual temporal models,\n"
    "dual edge-feature TTA, and geometry-validated divisions.\n\n"
    "Record edition.'''"
)
HEADER_NEW = (
    "'''Biohub C011: x138 without the private V1284 head\n\n"
    "anvithpothula/biohub-x138 (public 0.953) with V1284_MODE='zero': the private\n"
    "coordinate-refinement head is not mounted, detector coordinates pass through\n"
    "unchanged. Flow prior, readmit, gap filler and division geometry are x138's.'''"
)
PRESET_OLD = "BIOHUB_PRESET = 'harmonic_v3_division_wide'"
PRESET_NEW = "BIOHUB_PRESET = 'c011_x138_zero'"
AXIS_OLD = "BIOHUB_SCORE_AXIS = 'public 0.939 base + holdout-selected post-process configuration'"
AXIS_NEW = "BIOHUB_SCORE_AXIS = 'x138 (public 0.953) minus private V1284 head'"

HEAD_OLD = """_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))
if len(_myhead) != 1:
    raise RuntimeError(('my V1284 head mount mismatch', [str(p) for p in _myhead]))
# My own head, same architecture the module loads. Trained on x107's 20-movie TRAIN capture
# (4,136 detection<->GT pairs). Held-out BY MOVIE it moves centres CLOSER to truth:
#   ridge -10.8%   mlp -10.4%   POOLED, 16 of 20 movies improve.
# The 4-movie version of this head was +14.9% WORSE, so the null there was volume, not concept.
os.environ['V1284_MODE']='candidate'
os.environ['V1284_HEAD']=str(_myhead[0])"""
HEAD_NEW = """# C011: the author's private head dataset cannot be mounted by a fork.
# 'zero' is the module's own pass-through mode: refine() returns the detector
# coordinates unchanged, and the trilinear lookup equals the native gather at
# integer coordinates, so the rest of x138 runs exactly as without the head.
os.environ['V1284_MODE'] = 'zero'"""

FORBIDDEN = ("biohub-v1284-head-s075", "v1284_head.pt", "biohub-local-4070ti-weights", "best_local_unet_transformer.pth")

KERNEL_METADATA = {
    "id": KERNEL_ID,
    "title": "biohub-c011-x138-zero",
    "code_file": DEST_NB.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ],
    "kernel_sources": [],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    raw = SOURCE_NB.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not digest.startswith(SOURCE_SHA256):
        raise SystemExit(f"x138 source changed: sha256 {digest}")
    nb = json.loads(raw)
    cells = nb["cells"]

    cell0 = "".join(cells[0]["source"])
    cell0 = replace_once(cell0, HEADER_OLD, HEADER_NEW, "cell0 header")
    cell0 = replace_once(cell0, PRESET_OLD, PRESET_NEW, "cell0 preset")
    cell0 = replace_once(cell0, AXIS_OLD, AXIS_NEW, "cell0 score axis")
    cell4 = replace_once("".join(cells[4]["source"]), HEAD_OLD, HEAD_NEW, "cell4 V1284 head")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    nb["metadata"]["title"] = KERNEL_METADATA["title"]

    for index, cell in enumerate(cells):
        src = "".join(cell["source"])
        ast.parse(src)
        for token in FORBIDDEN:
            if token in src:
                raise SystemExit(f"cell {index} still references {token}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DEST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(KERNEL_METADATA, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST_NB.relative_to(REPO_ROOT)}")
    print(f"sha256 {hashlib.sha256(DEST_NB.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
