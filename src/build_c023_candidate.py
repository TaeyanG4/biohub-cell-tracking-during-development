#!/usr/bin/env python3
"""Build Candidate C023: C022 (every-pair stabilized relink + ILP-edge restore) with x138's own public V1284 head.

One change vs C022 (cell 4 + kernel metadata): the coordinate head is mounted from the public CC0 dataset
`anvithpothula/biohub-v1284-head-s075` (the head behind x138's public 0.953; SHA256-pinned) instead of our head v1
(`taeyangg4/biohub-c012-v1284-head`). Everything else is byte-identical to C022. Purpose: a model-component
variant of our best family for the final two picks (x138 head vs ours: same public score class, different
per-embryo behaviour - HANDOFF section 21).

    python src/build_c023_candidate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C022_DIR = REPO_ROOT / "experiments" / "candidates" / "c022_stabilize_all_restore"
C022_NB = C022_DIR / "biohub-c022-stabilize-all-restore.ipynb"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c023_x138_head_stabilize_restore"
KERNEL_SLUG = "biohub-c023-x138-head-stabilize-restore"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
OUR_DATASET, X138_DATASET = "taeyangg4/biohub-c012-v1284-head", "anvithpothula/biohub-v1284-head-s075"
OUR_SHA = "9d3484f794b48c379b657714878ff3d7bee6042dd932ef257fced992b34edda6"
X138_SHA = "625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c"
LOCAL_X138_HEAD = REPO_ROOT / "artifacts" / "anvithpothula_v1284_head_s075" / "v1284_head.pt"

CELL4_EDITS = [
    ("mount path", "Path('/kaggle/input/biohub-c012-v1284-head/v1284_head.pt'),", "Path('/kaggle/input/biohub-v1284-head-s075/v1284_head.pt'),"),
    ("mount path (datasets layout)", "Path('/kaggle/input/datasets/taeyangg4/biohub-c012-v1284-head/v1284_head.pt'),",
     "Path('/kaggle/input/datasets/anvithpothula/biohub-v1284-head-s075/v1284_head.pt'),"),
    ("rglob", "rglob('biohub-c012-v1284-head/v1284_head.pt')", "rglob('biohub-v1284-head-s075/v1284_head.pt')"),
    ("sha", f"if _head_sha256 != '{OUR_SHA}':", f"if _head_sha256 != '{X138_SHA}':"),
    ("print", "print('C012 V1284 head:', _head_found[0], _head_sha256)", "print('C023 V1284 head (x138 public s075):', _head_found[0], _head_sha256)"),
]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    if hashlib.sha256(LOCAL_X138_HEAD.read_bytes()).hexdigest() != X138_SHA:
        raise SystemExit("local copy of the x138 head does not match the pinned SHA256")
    nb = json.loads(C022_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    header = next(l for l in cell0.splitlines() if l.startswith("'''Biohub C022:"))
    cell0 = replace_once(cell0, header, header.replace("Biohub C022:", "Biohub C023:").replace("own V1284 head v1", "x138 public V1284 head (s075)"), "cell0 header")
    cell0 = replace_once(cell0, "BIOHUB_PRESET = 'c022_stabilize_all_restore'", "BIOHUB_PRESET = 'c023_x138_head_stabilize_restore'", "cell0 preset")
    axis = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_SCORE_AXIS = "))
    cell0 = replace_once(cell0, axis, "BIOHUB_SCORE_AXIS = 'C022 with the x138 public V1284 head'", "cell0 axis")
    cell4 = "".join(cells[4]["source"])
    for label, old, new in CELL4_EDITS:
        cell4 = replace_once(cell4, old, new, label)
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    nb["metadata"]["title"] = KERNEL_SLUG
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    meta = json.loads((C022_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    if OUR_DATASET not in meta["dataset_sources"]:
        raise SystemExit("C022 metadata does not mount our head dataset")
    meta["dataset_sources"] = sorted([d for d in meta["dataset_sources"] if d != OUR_DATASET] + [X138_DATASET])
    meta.update(id=f"taeyangg4/{KERNEL_SLUG}", title=KERNEL_SLUG, code_file=DEST_NB.name)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DEST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST_NB.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}")
    print(f"notebook sha256 {hashlib.sha256(DEST_NB.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
