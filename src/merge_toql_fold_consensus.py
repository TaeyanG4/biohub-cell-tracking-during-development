from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = ROOT / "experiments" / "stabledet_visible_runtime" / "src"
sys.path.insert(0, str(RUNTIME_SRC))

from merge_toql_geometric_bridges import (  # noqa: E402
    EdgeKey,
    NodeKey,
    doubly_anchored_bridges,
    graph_sets,
    save_merged_graph,
)
from infer_stabledet_hoct import score_predictions  # noqa: E402


def endpoint_map(paths: list[list[EdgeKey]]) -> dict[tuple[NodeKey, NodeKey], list[EdgeKey]]:
    out: dict[tuple[NodeKey, NodeKey], list[EdgeKey]] = {}
    for path in paths:
        if not path:
            continue
        key = (path[0][0], path[-1][1])
        out[key] = path
    return out


def midpoint_path(path0: list[EdgeKey], path1: list[EdgeKey]) -> list[EdgeKey]:
    if len(path0) != len(path1):
        raise ValueError("consensus paths with same anchors must have the same frame span")
    nodes0 = [path0[0][0], *[edge[1] for edge in path0]]
    nodes1 = [path1[0][0], *[edge[1] for edge in path1]]
    if nodes0[0] != nodes1[0] or nodes0[-1] != nodes1[-1]:
        raise ValueError("anchor mismatch")
    merged = [nodes0[0]]
    for a, b in zip(nodes0[1:-1], nodes1[1:-1], strict=True):
        if a[0] != b[0]:
            raise ValueError("time mismatch")
        coords = np.rint((np.asarray(a[1:], dtype=float) + np.asarray(b[1:], dtype=float)) / 2.0).astype(int)
        merged.append((a[0], int(coords[0]), int(coords[1]), int(coords[2])))
    merged.append(nodes0[-1])
    return list(zip(merged[:-1], merged[1:]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-dir", type=Path, required=True)
    p.add_argument("--toql0-dir", type=Path, required=True)
    p.add_argument("--toql1-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-bridge-edges", type=int, default=4)
    p.add_argument("--gt-dir", type=Path)
    args = p.parse_args()

    expected = {x.stem for x in args.backbone_dir.glob("*.geff")}
    variants = {"fold0": {}, "fold1": {}, "midpoint": {}}
    summaries: dict[str, list[dict]] = {k: [] for k in variants}

    for bpath in sorted(args.backbone_dir.glob("*.geff")):
        sample = bpath.stem
        hn_nodes, hn_edges = graph_sets(bpath)
        _, e0 = graph_sets(args.toql0_dir / bpath.name)
        _, e1 = graph_sets(args.toql1_dir / bpath.name)
        b0 = doubly_anchored_bridges(hn_nodes, hn_edges, e0, args.max_bridge_edges)
        b1 = doubly_anchored_bridges(hn_nodes, hn_edges, e1, args.max_bridge_edges)
        m0, m1 = endpoint_map(b0), endpoint_map(b1)
        common = sorted(set(m0) & set(m1))

        chosen = {
            "fold0": [m0[k] for k in common],
            "fold1": [m1[k] for k in common],
            "midpoint": [midpoint_path(m0[k], m1[k]) for k in common],
        }
        print(
            sample,
            "fold0", len(b0),
            "fold1", len(b1),
            "consensus", len(common),
            flush=True,
        )
        for variant, bridges in chosen.items():
            outdir = args.output_root / variant
            outdir.mkdir(parents=True, exist_ok=True)
            summary = {
                "sample": sample,
                "fold0_candidates": len(b0),
                "fold1_candidates": len(b1),
                "consensus_bridges": len(common),
                **save_merged_graph(hn_nodes, hn_edges, bridges, outdir / bpath.name),
            }
            summaries[variant].append(summary)

    for variant, rows in summaries.items():
        outdir = args.output_root / variant
        (outdir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        if args.gt_dir is not None:
            metrics = score_predictions(outdir, args.gt_dir, expected)
            print(variant, "official metrics", metrics, flush=True)


if __name__ == "__main__":
    main()
