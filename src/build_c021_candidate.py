#!/usr/bin/env python3
"""Build Candidate C021: C017 (C012 + jump-stabilized relink) + ILP-edge restore (josephadamski V1057).

One change vs C017 (cell 5, before `write_test_submission("base")`): `filter_output_graph` is wrapped; after all
post-processing every ILP edge with p >= BIOHUB_V1057_MIN_PROB (0.7) that is not in the final graph is put back,
conflicts resolved by descending probability (each source and target used once); nodes with two children and
their daughters are never touched; displaced final edges are dropped; no nodes are added or removed.
Same logic as the harness option `src/eval_pp_variants_local.py --ilp-edge-restore` (97 movies at 0.7: +0.0017 vs
C017, 36 better / 11 worse).

    python src/build_c021_candidate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C017_DIR = REPO_ROOT / "experiments" / "candidates" / "c017_jump_stabilized_relink"
C017_NB = C017_DIR / "biohub-c017-jump-stabilized-relink.ipynb"
C017_META = C017_DIR / "kernel-metadata.json"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c021_ilp_edge_restore"
KERNEL_SLUG = "biohub-c021-ilp-edge-restore"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
SUBMISSION_CALL = '\nwrite_test_submission("base")\n'

RESTORE_BLOCK = '''

# ----------------------------------------------------------------- C021 ILP-edge restore (josephadamski V1057)
V1057_MIN_PROB = float(os.environ.get("BIOHUB_V1057_MIN_PROB", "0.7"))
_unrestored_filter_output_graph = filter_output_graph


def filter_output_graph(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
    raw = [(int(e["source_id"]), int(e["target_id"]), e.get("edge_prob")) for e in raw_edges]
    nodes, edges, stats = _unrestored_filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
    if V1057_MIN_PROB <= 0 or not edges:
        return nodes, edges, stats
    node_time = {int(k): int(v["t"]) for k, v in nodes.items()}
    anchor = {(int(e["source_id"]), int(e["target_id"])): e for e in edges}
    outgoing = {k: [] for k in node_time}
    incoming = {k: [] for k in node_time}
    for s, t in anchor:
        outgoing.setdefault(s, []).append(t)
        incoming.setdefault(t, []).append(s)
    cand = {}
    for s, t, p in raw:
        if s not in node_time or t not in node_time or (s, t) in anchor or node_time[t] != node_time[s] + 1:
            continue
        if len(outgoing.get(s, [])) > 1:
            continue
        owner = incoming[t][0] if len(incoming.get(t, [])) == 1 else None
        if owner is not None and len(outgoing.get(owner, [])) > 1:
            continue
        if p is None or not np.isfinite(float(p)) or float(p) < V1057_MIN_PROB:
            continue
        cand[(s, t)] = max(float(p), cand.get((s, t), float("-inf")))
    used_s, used_t, selected = set(), set(), []
    for (s, t), p in sorted(cand.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        if s in used_s or t in used_t:
            continue
        selected.append((s, t, p))
        used_s.add(s)
        used_t.add(t)
    kept = [e for (s, t), e in anchor.items() if s not in used_s and t not in used_t]
    new = [{"source_id": s, "target_id": t, "edge_prob": p, "v1057_restored": 1} for s, t, p in selected]
    stats["v1057_restored"] = len(new)
    stats["v1057_displaced"] = len(anchor) - len(kept)
    print(f"  [{dataset}] C021 ILP-edge restore: restored {len(new)}, displaced {len(anchor) - len(kept)}", flush=True)
    return nodes, kept + new, stats

'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=C017_NB, help="notebook to add the restore stage to (default: C017)")
    parser.add_argument("--base-label", default="C017", help="label of the base notebook's cell-0 header")
    parser.add_argument("--slug", default=KERNEL_SLUG)
    parser.add_argument("--dir", type=Path, default=DEST_DIR)
    parser.add_argument("--label", default="C021")
    parser.add_argument("--min-prob", type=float, default=0.7)
    args = parser.parse_args()
    base_nb, dest_dir = args.base.resolve(), args.dir.resolve()
    dest_nb = dest_dir / f"{args.slug}.ipynb"
    nb = json.loads(base_nb.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    header = next(l for l in cell0.splitlines() if l.startswith(f"'''Biohub {args.base_label}:"))
    cell0 = replace_once(cell0, header, header.replace(f"Biohub {args.base_label}:", f"Biohub {args.label}:") + " + ILP-edge restore", "cell0 header")
    preset = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_PRESET = "))
    cell0 = replace_once(cell0, preset, f"BIOHUB_PRESET = '{args.slug.replace('biohub-', '').replace('-', '_')}'", "cell0 preset")
    axis = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_SCORE_AXIS = "))
    cell0 = replace_once(cell0, axis, f"BIOHUB_SCORE_AXIS = '{args.base_label} + ILP-edge restore (V1057, p >= {args.min_prob})'", "cell0 axis")
    cell5 = "".join(cells[5]["source"])
    cell5 = replace_once(cell5, SUBMISSION_CALL, RESTORE_BLOCK.replace('"BIOHUB_V1057_MIN_PROB", "0.7"', f'"BIOHUB_V1057_MIN_PROB", "{args.min_prob}"') + SUBMISSION_CALL, "cell5 submission call")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[5]["source"] = cell5.splitlines(keepends=True)
    nb["metadata"]["title"] = args.slug
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    meta = json.loads((base_nb.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{args.slug}", title=args.slug, code_file=dest_nb.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest_nb.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}")
    print(f"notebook sha256 {hashlib.sha256(dest_nb.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
