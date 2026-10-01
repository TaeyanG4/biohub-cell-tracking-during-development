#!/usr/bin/env python3
"""V1284 head ensemble: average the bounded coordinate shifts of one or more heads.

Heads are trained independently on the same frozen 224-d features; each shift is bounded to < 2 um, so the mean
is too. With one head the patched module is identical to the original.

  repo      build a local tracking_repo copy whose v1284_coordinate_refinement.py accepts ';'-joined V1284_HEAD
            (python src/build_head_ensemble.py repo [--source tmp/c022_output/tracking_repo] [--dest tmp/c024_head_ens/tracking_repo])
            then: python src/run_kaggle_predict_local.py --repo tmp/c024_head_ens/tracking_repo ... --v1284-mode candidate \\
                  --v1284-head "experiments/candidates/c012_v1284_head/heads/head_v1.pt;artifacts/anvithpothula_v1284_head_s075/v1284_head.pt"
  notebook  build a candidate = C022 with the given heads mounted (SHA256-pinned) and averaged
            (python src/build_head_ensemble.py notebook [--head DATASET=SHA256[:FILENAME] ...] [--label C024] [--slug ...] [--dir ...])
            default heads = C024: ours v1 + x138's public s075
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C022_DIR = REPO_ROOT / "experiments" / "candidates" / "c022_stabilize_all_restore"
C022_NB = C022_DIR / "biohub-c022-stabilize-all-restore.ipynb"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c024_head_ensemble"
KERNEL_SLUG = "biohub-c024-head-ensemble"
OUR_DATASET, X138_DATASET = "taeyangg4/biohub-c012-v1284-head", "anvithpothula/biohub-v1284-head-s075"
OUR_SHA = "9d3484f794b48c379b657714878ff3d7bee6042dd932ef257fced992b34edda6"
X138_SHA = "625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c"

MODULE_OLD = """    if _CACHE is None:
        saved = torch.load(os.environ['V1284_HEAD'], map_location='cpu', weights_only=True)
        head = make_head().to(feature.device)
        head.load_state_dict(saved['state_dict']); head.eval()
        _CACHE = (head, saved['mean'].to(feature.device), saved['scale'].to(feature.device))
    head, mean, scale = _CACHE
    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING
"""
MODULE_NEW = """    if _CACHE is None:
        heads = []
        for path in [p for p in os.environ['V1284_HEAD'].split(';') if p]:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            head = make_head().to(feature.device)
            head.load_state_dict(saved['state_dict']); head.eval()
            heads.append((head, saved['mean'].to(feature.device), saved['scale'].to(feature.device)))
        if not heads:
            raise RuntimeError('V1284_HEAD lists no head file')
        _CACHE = tuple(heads)
    # Head ensemble: mean of the bounded shifts (each < 2 um, so the mean is too); one head = the original module.
    shift = torch.stack([bounded(head, (x-mean)/scale) for head, mean, scale in _CACHE]).mean(dim=0).cpu().numpy() / SPACING
"""

MOUNT_OLD_START = "_head_candidates = ["
MOUNT_OLD_END = "print('C012 V1284 head:', _head_found[0], _head_sha256)\n"
MOUNT_NEW_TEMPLATE = """def _find_head(dataset_slug, expected_sha, filename):
    owner, name = dataset_slug.split('/')
    found = [p for p in (Path(f'/kaggle/input/{name}/{filename}'), Path(f'/kaggle/input/datasets/{owner}/{name}/{filename}')) if p.is_file()]
    if not found:
        found = sorted(Path('/kaggle/input').rglob(f'{name}/{filename}'))
    if len(found) != 1:
        raise RuntimeError(('V1284 head mount mismatch', dataset_slug, [str(p) for p in found]))
    import hashlib as _hh
    sha = _hh.sha256(found[0].read_bytes()).hexdigest()
    if sha != expected_sha:
        raise RuntimeError(('V1284 head checksum mismatch', dataset_slug, sha))
    print('__LABEL__ V1284 head:', dataset_slug, found[0], sha)
    return found[0]

_head_paths = [__HEADS__]
os.environ['V1284_MODE'] = 'candidate'
os.environ['V1284_HEAD'] = ';'.join(str(p) for p in _head_paths)
print('__LABEL__ V1284 head ensemble:', os.environ['V1284_HEAD'])
"""


def patch_module(text: str) -> str:
    if text.count(MODULE_OLD) != 1:
        raise SystemExit(f"module anchor: expected one, found {text.count(MODULE_OLD)}")
    return text.replace(MODULE_OLD, MODULE_NEW, 1)


def build_repo(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for part in ("scripts", "src"):
        shutil.copytree(source / part, dest / part, ignore=shutil.ignore_patterns("__pycache__"))
    module = dest / "scripts" / "v1284_coordinate_refinement.py"
    module.write_text(patch_module(module.read_text(encoding="utf-8")), encoding="utf-8")
    compile(module.read_text(encoding="utf-8"), str(module), "exec")
    print(f"wrote {module}")


def build_notebook(heads: list[tuple[str, str, str]], label: str, slug: str, dest_dir: Path) -> None:
    """heads = [(dataset slug, sha256, filename), ...]; the notebook mounts them all and averages the bounded shifts."""
    dest_nb = dest_dir / f"{slug}.ipynb"
    nb = json.loads(C022_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    names = " + ".join(h[0].split("/")[1] for h in heads)
    cell0 = "".join(cells[0]["source"])
    header = next(l for l in cell0.splitlines() if l.startswith("'''Biohub C022:"))
    cell0 = cell0.replace(header, header.replace("Biohub C022:", f"Biohub {label}:").replace("own V1284 head v1", f"V1284 head(s) {names}"), 1)
    assert cell0.count("BIOHUB_PRESET = 'c022_stabilize_all_restore'") == 1
    cell0 = cell0.replace("BIOHUB_PRESET = 'c022_stabilize_all_restore'", f"BIOHUB_PRESET = '{slug.replace('biohub-', '').replace('-', '_')}'", 1)
    axis = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_SCORE_AXIS = "))
    cell0 = cell0.replace(axis, f"BIOHUB_SCORE_AXIS = 'C022 with V1284 head(s) {names} (mean of bounded shifts)'", 1)
    cell4 = "".join(cells[4]["source"])
    # 1. the embedded module source: cell 4 writes it via write_text(<repr of the module text>), so patch the value
    #    and re-emit it with repr() (the notebook generator used repr; verified by the exact-match check)
    literal = next(node.args[0].value for node in ast.walk(ast.parse(cell4))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
                   and "v1284_coordinate_refinement.py" in ast.unparse(node.func.value))
    if cell4.count(repr(literal)) != 1:
        raise SystemExit("embedded module literal is not a plain repr() of its value; cannot patch safely")
    cell4 = cell4.replace(repr(literal), repr(patch_module(literal)), 1)
    # 2. the head mount block
    i, j = cell4.find(MOUNT_OLD_START), cell4.find(MOUNT_OLD_END)
    if i < 0 or j < 0 or cell4.count(MOUNT_OLD_START) != 1 or cell4.count(MOUNT_OLD_END) != 1:
        raise SystemExit("head mount block anchors not unique in cell 4")
    mount_new = MOUNT_NEW_TEMPLATE.replace("__LABEL__", label).replace(
        "__HEADS__", ", ".join(f"_find_head('{d}', '{sha}', '{fn}')" for d, sha, fn in heads))
    cell4 = cell4[:i] + mount_new + cell4[j + len(MOUNT_OLD_END):]
    cell4 = cell4.replace("os.environ['V1284_MODE'] = 'candidate'\nos.environ['V1284_HEAD'] = str(_head_found[0])\n", "", 1)
    if "_head_found" in cell4:
        raise SystemExit("stale _head_found reference left in cell 4")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    nb["metadata"]["title"] = slug
    for k, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{k}", "exec")
    meta = json.loads((C022_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    if OUR_DATASET not in meta["dataset_sources"]:
        raise SystemExit("C022 metadata does not mount our head dataset")
    meta["dataset_sources"] = sorted((set(meta["dataset_sources"]) - {OUR_DATASET}) | {d for d, _, _ in heads})
    meta.update(id=f"taeyangg4/{slug}", title=slug, code_file=dest_nb.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest_nb.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}")
    print(f"notebook sha256 {hashlib.sha256(dest_nb.read_bytes()).hexdigest()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("repo", "notebook"))
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "tmp" / "c022_output" / "tracking_repo")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "tmp" / "c024_head_ens" / "tracking_repo")
    ap.add_argument("--head", action="append", default=[], metavar="DATASET=SHA256[:FILENAME]",
                    help="notebook: heads to mount and average (default: C024 = ours v1 + x138 s075)")
    ap.add_argument("--label", default="C024")
    ap.add_argument("--slug", default=KERNEL_SLUG)
    ap.add_argument("--dir", type=Path, default=DEST_DIR)
    args = ap.parse_args()
    if args.what == "repo":
        build_repo(args.source, args.dest)
        return
    heads = []
    for item in args.head or [f"{OUR_DATASET}={OUR_SHA}", f"{X138_DATASET}={X138_SHA}"]:
        dataset, _, rest = item.partition("=")
        sha, _, filename = rest.partition(":")
        if not dataset or len(sha) != 64:
            raise SystemExit(f"--head expects DATASET=SHA256[:FILENAME], got {item!r}")
        heads.append((dataset, sha, filename or "v1284_head.pt"))
    build_notebook(heads, args.label, args.slug, args.dir.resolve())


if __name__ == "__main__":
    main()
