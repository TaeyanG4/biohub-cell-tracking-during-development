#!/usr/bin/env python3
"""Build Candidate C017: C012 (x138 pipeline + our V1284 head v1) + jump-stabilized motion relink.

One change vs C012, in cell 5 right before `write_test_submission("base")`: `motion_relink_edges` is wrapped
so that frame pairs whose ILP links move coherently by >= BIOHUB_STAB_MIN_UM (median link displacement) are
relinked on jump-free coordinates (every frame shifted by the cumulative jump sum; geometry only - image
lookups, readmit, gap filling and divisions keep the original coordinates). Same code as the harness option
`src/eval_pp_variants_local.py --stabilize-relink` (HANDOFF section 22). No new dataset: the kernel mounts
exactly C012's inputs.

    python src/build_c017_candidate.py [--min-um 3.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C012_DIR = REPO_ROOT / "experiments" / "candidates" / "c012_v1284_head"
C012_NB = C012_DIR / "biohub-c012-v1284-head-v1.ipynb"
C012_META = C012_DIR / "kernel-metadata.json"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c017_jump_stabilized_relink"
KERNEL_SLUG = "biohub-c017-jump-stabilized-relink"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
SUBMISSION_CALL = '\nwrite_test_submission("base")\n'

STAB_BLOCK = '''

# ----------------------------------------------------------------- C017 jump-stabilized motion relink
# Acquisition hiccups make every cell of a frame pair move together (frozen frames followed by double steps,
# whole-field jumps of 3-58 um in 137 of 199 train movies). The distance-gated Hungarian relink then breaks
# links the ILP had right. Per frame pair the global shift is the median displacement of the ILP's own links;
# pairs moving by >= STAB_MIN_UM are relinked on positions shifted by the cumulative jump sum.
STAB_MIN_UM = float(os.environ.get("BIOHUB_STAB_MIN_UM", "{min_um}"))
STAB_MIN_PROB = float(os.environ.get("BIOHUB_STAB_MIN_PROB", "0.5"))
STAB_MIN_EDGES = int(os.environ.get("BIOHUB_STAB_MIN_EDGES", "8"))
_unstabilized_motion_relink_edges = motion_relink_edges


def motion_relink_edges(nodes_by_id, stats, learned_edge_probs=None):
    if STAB_MIN_UM <= 0 or not learned_edge_probs:
        return _unstabilized_motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
    disp = {}
    for (source_id, target_id), prob in learned_edge_probs.items():
        a, b = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
        if a is None or b is None or int(b["t"]) != int(a["t"]) + 1 or float(prob) < STAB_MIN_PROB:
            continue
        disp.setdefault(int(a["t"]), []).append(
            (np.array([float(b["z"]), float(b["y"]), float(b["x"])]) - np.array([float(a["z"]), float(a["y"]), float(a["x"])])) * scale)
    offsets = {}
    for t, vectors in disp.items():
        if len(vectors) >= STAB_MIN_EDGES:
            med = np.median(np.array(vectors), axis=0)
            if float(np.linalg.norm(med)) >= STAB_MIN_UM:
                offsets[t] = med
    if not offsets:
        return _unstabilized_motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    cum, acc = {}, np.zeros(3)
    for t in sorted({int(n["t"]) for n in nodes_by_id.values()}):
        cum[t] = acc.copy()
        if t in offsets:
            acc = acc + offsets[t]
    shifted = {}
    for node_id, node in nodes_by_id.items():
        c = cum[int(node["t"])] / scale
        shifted[node_id] = {**node, "z": float(node["z"]) - c[0], "y": float(node["y"]) - c[1], "x": float(node["x"]) - c[2]}
    edges = _unstabilized_motion_relink_edges(shifted, stats, learned_edge_probs)
    for edge in edges:
        a, b = nodes_by_id.get(int(edge["source_id"])), nodes_by_id.get(int(edge["target_id"]))
        if a is not None and b is not None:
            edge["distance_um"] = edge_distance_um(a, b)
    stats["stabilized_jump_pairs"] = stats.get("stabilized_jump_pairs", 0) + len(offsets)
    return edges

'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-um", type=float, default=3.0)
    parser.add_argument("--slug", default=KERNEL_SLUG, help="kernel slug for a variant build (default: C017)")
    parser.add_argument("--dir", type=Path, default=DEST_DIR, help="output folder for a variant build")
    parser.add_argument("--label", default="C017", help="candidate label written into cell 0")
    args = parser.parse_args()
    slug, dest_dir = args.slug, args.dir.resolve()
    dest_nb = dest_dir / f"{slug}.ipynb"
    nb = json.loads(C012_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    cell0 = replace_once(cell0, "'''Biohub C012: x138 pipeline + own V1284-compatible head (v1)",
                         f"'''Biohub {args.label}: x138 pipeline + own V1284 head v1 + jump-stabilized motion relink", "cell0 header")
    cell0 = replace_once(cell0, "BIOHUB_PRESET = 'c012_v1284_head'", f"BIOHUB_PRESET = '{slug.replace('biohub-', '').replace('-', '_')}'", "cell0 preset")
    cell0 = replace_once(cell0, "BIOHUB_SCORE_AXIS = 'x138 pipeline + own V1284-compatible head (v1)'",
                         "BIOHUB_SCORE_AXIS = 'C012 + jump-stabilized motion relink'", "cell0 axis")
    cell5 = "".join(cells[5]["source"])
    cell5 = replace_once(cell5, SUBMISSION_CALL, STAB_BLOCK.replace("{min_um}", str(args.min_um)) + SUBMISSION_CALL, "cell5 submission call")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[5]["source"] = cell5.splitlines(keepends=True)
    nb["metadata"]["title"] = slug
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    meta = json.loads(C012_META.read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{slug}", title=slug, code_file=dest_nb.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest_nb.relative_to(REPO_ROOT)} (STAB_MIN_UM {args.min_um}); datasets {meta['dataset_sources']}")
    print(f"notebook sha256 {hashlib.sha256(dest_nb.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
