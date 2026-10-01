#!/usr/bin/env python3
"""Build a candidate that differs from a base notebook only in BIOHUB_* settings of cell 0 (the config cell).

Each --set KEY=VALUE rewrites the existing `os.environ["KEY"] = "..."` line of cell 0 (keeping its comment) or,
when the key is not set there, appends a new assignment after the last os.environ assignment of cell 0. The
harness variant {"NOTEBOOK_GLOBAL": value} and the env setting are equivalent because cell 2 derives every
post-processing global from its BIOHUB_* variable; verify the built notebook with the harness anyway.

    python src/build_env_variant_candidate.py --base experiments/candidates/c022_stabilize_all_restore/biohub-c022-stabilize-all-restore.ipynb \\
        --base-label C022 --label C025 --slug biohub-c025-relink-gates --dir experiments/candidates/c025_relink_gates \\
        --set BIOHUB_MOTION_RELINK_TIGHT_UM=5.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_env(cell0: str, key: str, value: str) -> str:
    pat = re.compile(r"""^(os\.environ\[["']%s["']\]\s*=\s*)(["'][^"']*["'])(.*)$""" % re.escape(key), re.M)
    hits = pat.findall(cell0)
    if len(hits) > 1:
        raise SystemExit(f"{key}: assigned {len(hits)} times in cell 0")
    if hits:
        return pat.sub(lambda m: f'{m.group(1)}"{value}"{m.group(3)}', cell0, count=1)
    lines = cell0.splitlines(keepends=True)
    last = max(i for i, l in enumerate(lines) if re.match(r"""^os\.environ\[["'][A-Z0-9_]+["']\]\s*=""", l))
    lines.insert(last + 1, f'os.environ["{key}"] = "{value}"  # variant setting\n')
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-label", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--axis", default="")
    args = ap.parse_args()
    base_nb, dest_dir = args.base.resolve(), args.dir.resolve()
    dest_nb = dest_dir / f"{args.slug}.ipynb"
    nb = json.loads(base_nb.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    header = next(l for l in cell0.splitlines() if l.startswith(f"'''Biohub {args.base_label}:"))
    cell0 = cell0.replace(header, header.replace(f"Biohub {args.base_label}:", f"Biohub {args.label}:") + " + settings " + ", ".join(args.set), 1)
    preset = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_PRESET = "))
    cell0 = cell0.replace(preset, f"BIOHUB_PRESET = '{args.slug.replace('biohub-', '').replace('-', '_')}'", 1)
    axis = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_SCORE_AXIS = "))
    cell0 = cell0.replace(axis, f"BIOHUB_SCORE_AXIS = '{args.axis or (args.base_label + ' + ' + ', '.join(args.set))}'", 1)
    for item in args.set:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        cell0 = set_env(cell0, key, value)
    cells[0]["source"] = cell0.splitlines(keepends=True)
    # cell 1 carries a configuration-drift guard with expected values; update the entries of the keys we changed
    cell1 = "".join(cells[1]["source"])
    for item in args.set:
        key, _, value = item.partition("=")
        pat = re.compile(r'("%s"\s*:\s*)(-?[0-9.]+|"[^"]*")' % re.escape(key))
        hits = pat.findall(cell1)
        if len(hits) > 1:
            raise SystemExit(f"{key}: {len(hits)} guard entries in cell 1")
        if hits:
            new_val = value if hits[0][1].startswith('"') else repr(float(value))
            if hits[0][1].startswith('"'):
                new_val = f'"{value}"'
            cell1 = pat.sub(lambda m: f"{m.group(1)}{new_val}", cell1, count=1)
            print(f"cell 1 guard: {key} {hits[0][1]} -> {new_val}")
    cells[1]["source"] = cell1.splitlines(keepends=True)
    nb["metadata"]["title"] = args.slug
    for k, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{k}", "exec")
    meta = json.loads((base_nb.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{args.slug}", title=args.slug, code_file=dest_nb.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    changed = [l.strip() for l in cell0.splitlines() if any(l.strip().startswith(f'os.environ["{i.partition("=")[0]}"]') for i in args.set)]
    print(f"wrote {dest_nb.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}")
    print("settings:", changed)
    print(f"notebook sha256 {hashlib.sha256(dest_nb.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
