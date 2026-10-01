#!/usr/bin/env python3
"""Build a head candidate: C011 (x138 minus private head) + our own V1284-compatible head.

x138's +0.007 over its code-identical head-less twin (thtennant frontier947-readmit-v1,
0.946) comes from a private coordinate head. These candidates reinstate that stage with a
head trained by src/v1284_head_train.py on features captured locally with the exact Kaggle
inference code (src/v1284_capture_local.py). The only change to C011 is the cell 4 block
that selects the V1284 mode: it mounts one head file from the private dataset
``taeyangg4/biohub-c012-v1284-head``, checks its SHA256, and runs ``V1284_MODE='candidate'``
(the path x138 itself ran).

    python src/build_c012_candidate.py --candidate c012 --head <head_v1.pt> --dataset-file v1284_head.pt
    python src/build_c012_candidate.py --candidate c013 --head <head_v1_a05.pt> --dataset-file v1284_head_a05.pt
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C011_NB = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero" / "biohub-c011-x138-zero.ipynb"
C011_META = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero" / "kernel-metadata.json"
DATASET_SLUG = "biohub-c012-v1284-head"
DATASET_ID = f"taeyangg4/{DATASET_SLUG}"
CANDIDATES = {
    "c012": {"dir": "c012_v1284_head", "slug": "biohub-c012-v1284-head-v1",
             "label": "x138 pipeline + own V1284-compatible head (v1)"},
    "c013": {"dir": "c013_v1284_head_a05", "slug": "biohub-c013-v1284-head-a05",
             "label": "x138 pipeline + own V1284-compatible head (v1, output layer x0.5)"},
    "c014": {"dir": "c014_v1284_head_noreadmit", "slug": "biohub-c014-v1284-head-noreadmit",
             "label": "x138 pipeline + own V1284-compatible head, readmit off"},
    "c015": {"dir": "c015_v1284_head_v1_noreadmit", "slug": "biohub-c015-v1284-head-v1-noreadmit",
             "label": "x138 pipeline + own V1284-compatible head v1 (C012), readmit off"},
}
ENV_LINE = '''os.environ["{key}"] = "{value}"'''

ZERO_BLOCK = """# C011: the author's private head dataset cannot be mounted by a fork.
# 'zero' is the module's own pass-through mode: refine() returns the detector
# coordinates unchanged, and the trilinear lookup equals the native gather at
# integer coordinates, so the rest of x138 runs exactly as without the head.
os.environ['V1284_MODE'] = 'zero'"""

HEAD_BLOCK = """# {cid}: our own V1284-compatible head (src/v1284_head_train.py), trained on features
# captured with this notebook's inference code and public weights. Same architecture,
# same bounded (< 2 um) shift, same 'candidate' path as x138.
_head_candidates = [
    Path('/kaggle/input/{dslug}/{fname}'),
    Path('/kaggle/input/datasets/taeyangg4/{dslug}/{fname}'),
]
_head_found = [p for p in _head_candidates if p.is_file()]
if not _head_found:
    _head_found = sorted(Path('/kaggle/input').rglob('{dslug}/{fname}'))
if len(_head_found) != 1:
    raise RuntimeError(('{cid} V1284 head mount mismatch', [str(p) for p in _head_found]))
import hashlib as _head_hashlib
_head_sha256 = _head_hashlib.sha256(_head_found[0].read_bytes()).hexdigest()
if _head_sha256 != '{sha256}':
    raise RuntimeError(('{cid} V1284 head checksum mismatch', _head_sha256))
os.environ['V1284_MODE'] = 'candidate'
os.environ['V1284_HEAD'] = str(_head_found[0])
print('{cid} V1284 head:', _head_found[0], _head_sha256)"""

LABEL_OLD = ("BIOHUB_PRESET = 'c011_x138_zero'", "BIOHUB_SCORE_AXIS = 'x138 (public 0.953) minus private V1284 head'")
HEADER_OLD = "'''Biohub C011: x138 without the private V1284 head"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--head", type=Path, required=True, help="local head file (its SHA256 is pinned)")
    parser.add_argument("--dataset-file", required=True, help="file name of this head inside the dataset")
    parser.add_argument("--dataset-slug", default=DATASET_SLUG, help="taeyangg4/<slug> holding the head file")
    parser.add_argument("--env", action="append", default=[], metavar="BIOHUB_KEY=VALUE",
                        help="change an existing literal os.environ setting in cell 0 (must exist exactly once)")
    args = parser.parse_args()
    spec = CANDIDATES[args.candidate]
    cid = args.candidate.upper()
    sha256 = hashlib.sha256(args.head.read_bytes()).hexdigest()
    dataset_id = f"taeyangg4/{args.dataset_slug}"
    dest_dir = REPO_ROOT / "experiments" / "candidates" / spec["dir"]
    dest_nb = dest_dir / f"{spec['slug']}.ipynb"

    nb = json.loads(C011_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    cell0 = replace_once(cell0, HEADER_OLD, f"'''Biohub {cid}: {spec['label']}", "cell0 header")
    cell0 = replace_once(cell0, LABEL_OLD[0], f"BIOHUB_PRESET = '{spec['dir']}'", "cell0 preset")
    cell0 = replace_once(cell0, LABEL_OLD[1], f"BIOHUB_SCORE_AXIS = '{spec['label']}'", "cell0 axis")
    for item in args.env:
        key, value = item.split("=", 1)
        matches = re.findall(rf'''^os\.environ\["{re.escape(key)}"\] = "([^"]*)"''', cell0, re.M)
        if len(matches) != 1:
            raise SystemExit(f"cell0: expected one literal setting of {key}, found {len(matches)}")
        cell0 = replace_once(cell0, ENV_LINE.format(key=key, value=matches[0]), ENV_LINE.format(key=key, value=value), f"cell0 {key}")
    block = (HEAD_BLOCK.replace("{cid}", cid).replace("{dslug}", args.dataset_slug)
             .replace("{fname}", args.dataset_file).replace("{sha256}", sha256))
    cell4 = replace_once("".join(cells[4]["source"]), ZERO_BLOCK, block, "cell4 V1284 mode")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    nb["metadata"]["title"] = spec["slug"]
    for cell in cells:
        ast.parse("".join(cell["source"]))

    meta = json.loads(C011_META.read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{spec['slug']}", title=spec["slug"], code_file=dest_nb.name)
    meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) | {dataset_id})

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest_nb.relative_to(REPO_ROOT)} (head {args.dataset_file} sha256 {sha256})")
    print(f"notebook sha256 {hashlib.sha256(dest_nb.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
