#!/usr/bin/env python3
"""Node-budget pre-analysis: which predicted nodes are (almost) never GT-matched, and what does removing them
do to the official score (edge Jaccard x node-count multiplier)?

The metric multiplies the edge Jaccard by 1 - 0.1 * (T_pred - T_true) / T_true (no ceiling), so every removed
node is worth ~0.1 J / T_true, while removing a GT-matched node loses its (up to two) TP edges, ~2 / N_gt_edges.
A node group is worth removing only if its match rate is below ~0.05 * J * (mean match rate) * T_pred / T_true
(~0.4 % on 6bba movies). This script measures, on the 22 evaluation movies with C012-configuration runs:

  stage attrs : replay the notebook's post-processing (final graph), attach per-node attributes (detection-peak
                score, DeepCenter score, raw intensity, synthetic flags, degree, track length, incident edge
                probability, frame position), match nodes to the sparse GT (7 um bipartite, as the metric does),
                cache everything per movie;
  stage rules : match-rate tables per attribute bin, then score candidate removal rules with the official
                per-movie metric (`score_sample`) and aggregate with the 6bba proxy.

    python src/node_budget_analysis.py --stage all --out reports/node_budget
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from division_scorer_stage import load_dump, make_intensity_fn  # noqa: E402
from eval_pp_variants_local import DEFAULT_GT_DIR, build_namespace, install_frame_caches, load_raw_graph  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "experiments/candidates/c011_x138_zero/biohub-c011-x138-zero.ipynb"
RUNS = [REPO / "experiments/candidates/c012_v1284_head/e2e/head_v1", REPO / "experiments/candidates/c012_v1284_head/e2e_confirm/head_v1"]
VOX = np.array([1.625, 0.40625, 0.40625])


def track_lengths(nodes, edges):
    """Weakly connected component size per node (tracks incl. divisions)."""
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for e in edges:
        a, b = find(int(e["source_id"])), find(int(e["target_id"]))
        if a != b:
            parent[a] = b
    root = {n: find(n) for n in nodes}
    size = {}
    for r in root.values():
        size[r] = size.get(r, 0) + 1
    return {n: size[root[n]] for n in nodes}


def stage_attrs(args, ns, frames, heatmaps):
    bundle = ns.get("DEEPCENTER_VETO_DETECTOR")
    cache_dir = args.out / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out / "attrs.log").open("a", encoding="utf-8")
    for run in RUNS:
        for pred_path in sorted((run / "predictions").glob("*.geff")):
            stem = pred_path.stem
            out_pkl = cache_dir / f"{stem}.pkl"
            if out_pkl.exists() and not args.force:
                continue
            t0 = time.time()
            os.environ["BIOHUB_CACHE_DIR"] = str(run / "edge_cache")
            raw_nodes, raw_edges = load_raw_graph(ns, pred_path)
            gt_path = args.gt_dir / f"{stem}.geff"
            gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](gt_path))
            t_true = ns["read_estimated_true_node_count"](gt_path)
            with contextlib.redirect_stdout(log):
                nodes, edges, stats = ns["filter_output_graph"](copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=bundle)
            admitted, low_by_t = load_dump(np.load(run / "edge_cache" / f"{stem}.npz"))
            frame_cache: dict = {}
            intensity = make_intensity_fn(stem, ns["read_test_frame"], frame_cache)
            in_e, out_e = {}, {}
            for e in edges:
                in_e.setdefault(int(e["target_id"]), []).append(e)
                out_e.setdefault(int(e["source_id"]), []).append(e)
            tlen = track_lengths(nodes, edges)
            plain = ns["nodes_by_id_to_plain"](nodes)
            plain_r = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in plain.items()}
            p2g, _ = ns["match_nodes_bipartite"](plain_r, gt_nodes, max_dist=7.0)
            t_max = max(int(n["t"]) for n in nodes.values())
            attrs = {}
            for nid, n in nodes.items():
                t = int(n["t"]); um = np.array([float(n["z"]), float(n["y"]), float(n["x"])]) * VOX
                entry = low_by_t.get(t)
                if entry is not None:
                    d, j = entry[0].query(um, k=1); det = float(entry[1][j]) if d <= 2.0 else 0.0
                else:
                    det = 0.0
                dc = ns["deepcenter_score_point"](stem, t, (float(n["z"]), float(n["y"]), float(n["x"])), bundle, frame_cache, heatmaps)
                probs = [float(e["edge_prob"]) for e in in_e.get(nid, []) + out_e.get(nid, []) if e.get("edge_prob") is not None]
                attrs[nid] = {
                    "t": t, "det": det, "dc": float(dc) if dc is not None else -1.0,
                    "int_peak": intensity(t, n["z"], n["y"], n["x"])[0],
                    "readmitted": int(n.get("readmitted", 0) or 0), "gap_synth": int((n.get("gap_synthetic", 0) or 0) == 1 or (n.get("gapfill_peak", 0) or 0) == 1),
                    "in_deg": len(in_e.get(nid, [])), "out_deg": len(out_e.get(nid, [])), "track_len": tlen[nid],
                    "max_prob": max(probs) if probs else -1.0, "n_pp_edges": sum(1 for e in in_e.get(nid, []) + out_e.get(nid, []) if e.get("edge_prob") is None),
                    "frame_frac": t / max(1, t_max), "matched": int(nid in p2g),
                }
            with out_pkl.open("wb") as f:
                pickle.dump({"stem": stem, "nodes": nodes, "edges": edges, "attrs": attrs, "gt_nodes": gt_nodes, "gt_edges": gt_edges,
                             "t_true": t_true, "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))}}, f)
            n_m = sum(a["matched"] for a in attrs.values())
            print(f"{stem}: nodes {len(nodes)} matched {n_m} ({100 * n_m / len(nodes):.1f} %) T_true {t_true} [{time.time() - t0:.0f}s]", flush=True)
            frames.clear(); heatmaps.clear()


def bin_table(items, key, edges_, label):
    """Match rate per bin of attribute `key`; items = list of (attr dict, stem prefix)."""
    vals = np.array([a[key] for a, _ in items]); m = np.array([a["matched"] for a, _ in items])
    print(f"  {label}:")
    for lo, hi in zip(edges_[:-1], edges_[1:]):
        sel = (vals >= lo) & (vals < hi)
        if sel.sum() == 0:
            continue
        print(f"    [{lo:>6.2f}, {hi:>6.2f}) share {100 * sel.mean():5.1f} %  match rate {100 * m[sel].mean():6.2f} %  (n={sel.sum()})")


def score_graph(ns, nodes, edges, gt_nodes, gt_edges, t_true, remove):
    keep_nodes = {nid: n for nid, n in nodes.items() if nid not in remove}
    keep_edges = [e for e in edges if int(e["source_id"]) in keep_nodes and int(e["target_id"]) in keep_nodes]
    plain = ns["nodes_by_id_to_plain"](keep_nodes)
    plain = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in plain.items()}
    row = ns["score_sample"](plain, [(int(e["source_id"]), int(e["target_id"])) for e in keep_edges], gt_nodes, gt_edges, t_true)
    row["nodes"] = len(keep_nodes); row["edges"] = len(keep_edges)
    return row


RULES = {
    "det<0.30": lambda a: a["det"] < 0.30,
    "det<0.45": lambda a: a["det"] < 0.45,
    "dc<0.05": lambda a: 0 <= a["dc"] < 0.05,
    "dc<0.10": lambda a: 0 <= a["dc"] < 0.10,
    "readmitted": lambda a: a["readmitted"] == 1,
    "gap_synth": lambda a: a["gap_synth"] == 1,
    "track_len<=2": lambda a: a["track_len"] <= 2,
    "track_len<=4": lambda a: a["track_len"] <= 4,
    "isolated(deg0)": lambda a: a["in_deg"] + a["out_deg"] == 0,
    "leaf&det<0.45": lambda a: (a["in_deg"] == 0 or a["out_deg"] == 0) and a["det"] < 0.45,
    "int_peak<0.15": lambda a: a["int_peak"] < 0.15,
    "det<0.45&dc<0.10": lambda a: a["det"] < 0.45 and 0 <= a["dc"] < 0.10,
    "max_prob<0.5(non-pp)": lambda a: 0 <= a["max_prob"] < 0.5,
}


def stage_rules(args, ns):
    cache_dir = args.out / "cache"
    movies = [pickle.load(p.open("rb")) for p in sorted(cache_dir.glob("*.pkl"))]
    items = [(a, m["stem"][:4]) for m in movies for a in m["attrs"].values()]
    for prefix in ("6bba", "44b6"):
        sub = [it for it in items if it[1] == prefix]
        mean_rate = np.mean([a["matched"] for a, _ in sub])
        print(f"\n===== {prefix}: {len(sub)} nodes over {sum(1 for m in movies if m['stem'].startswith(prefix))} movies, mean match rate {100 * mean_rate:.2f} % "
              f"-> removal pays only below ~{100 * 0.05 * 0.93 * mean_rate:.3f} %")
        bin_table(sub, "det", [0, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95, 1.01], "detection-peak score")
        bin_table(sub, "dc", [-1, 0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.01], "DeepCenter score (-1 = unavailable)")
        bin_table(sub, "int_peak", [-9, 0.15, 0.3, 0.5, 0.8, 1.2, 99], "raw intensity peak (frame-normalised)")
        bin_table(sub, "track_len", [1, 2, 3, 5, 10, 20, 50, 10 ** 9], "track (component) length")
        bin_table(sub, "max_prob", [-1, 0, 0.5, 0.8, 0.95, 1.01], "max incident edge prob (-1 = post-processing edges only)")
        for key in ("readmitted", "gap_synth"):
            sel = [a for a, _ in sub if a[key] == 1]
            print(f"  {key}: share {100 * len(sel) / len(sub):5.2f} %  match rate {100 * np.mean([a['matched'] for a in sel]) if sel else 0:6.2f} %  (n={len(sel)})")
        for name, rule in RULES.items():
            sel = [a for a, _ in sub if rule(a)]
            print(f"  rule {name:22s}: share {100 * len(sel) / len(sub):5.2f} %  match rate {100 * np.mean([a['matched'] for a in sel]) if sel else 0:6.2f} %  (n={len(sel)})")

    print("\n===== official per-movie scoring of removal rules (adjusted edge J + 0.1 divJ; 6bba proxy = 0.298 x 6bba total + 0.668)")
    rows = []
    for m in movies:
        base = score_graph(ns, m["nodes"], m["edges"], m["gt_nodes"], m["gt_edges"], m["t_true"], set())
        rows.append({**base, "stem": m["stem"], "config": "baseline"})
        for name, rule in RULES.items():
            if name not in args.rules:
                continue
            remove = {nid for nid, a in m["attrs"].items() if rule(a)}
            r = score_graph(ns, m["nodes"], m["edges"], m["gt_nodes"], m["gt_edges"], m["t_true"], remove)
            rows.append({**r, "stem": m["stem"], "config": name, "removed": len(remove)})
        print(f"{m['stem']}: baseline adj {base['adjusted_edge_jaccard']:.4f} (nodes {base['nodes']}, T_true {m['t_true']}) | " +
              " | ".join(f"{r['config']} {r['adjusted_edge_jaccard'] - base['adjusted_edge_jaccard']:+.4f} (-{r.get('removed', 0)})"
                         for r in rows if r["stem"] == m["stem"] and r["config"] != "baseline"), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(args.out / "rules_per_movie.csv", index=False)
    print(f"\n{'config':24s} {'group':6s} {'n':>3s} {'score':>7s} {'adj':>7s} {'divJ':>6s} {'d score':>8s}  better/worse movies")
    for group, stems in (("all", sorted({r['stem'] for r in rows})), ("6bba", sorted({r['stem'] for r in rows if r['stem'].startswith('6bba')})),
                         ("44b6", sorted({r['stem'] for r in rows if r['stem'].startswith('44b6')}))):
        base_agg = ns["aggregate_official"]([r for r in rows if r["config"] == "baseline" and r["stem"] in stems])
        for cfg in ["baseline"] + [n for n in RULES if n in args.rules]:
            sel = [r for r in rows if r["config"] == cfg and r["stem"] in stems]
            if not sel:
                continue
            agg = ns["aggregate_official"](sel)
            base_by = {r["stem"]: r["adjusted_edge_jaccard"] for r in rows if r["config"] == "baseline"}
            better = sum(1 for r in sel if r["adjusted_edge_jaccard"] > base_by[r["stem"]] + 1e-6)
            worse = sum(1 for r in sel if r["adjusted_edge_jaccard"] < base_by[r["stem"]] - 1e-6)
            print(f"{cfg:24s} {group:6s} {len(sel):3d} {agg['proxy_score']:7.4f} {agg['adjusted_edge_jaccard']:7.4f} {agg['division_jaccard']:6.4f} "
                  f"{agg['proxy_score'] - base_agg['proxy_score']:+8.4f}  {better}/{worse}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["attrs", "rules", "all"], default="all")
    parser.add_argument("--out", type=Path, default=REPO / "reports/node_budget")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--rules", default=",".join(RULES), help="comma list of RULES to score officially")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.rules = set(args.rules.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "namespace.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = build_namespace(NOTEBOOK.resolve(), {}, None)
    ns["TEST_DIR"] = args.gt_dir.resolve()
    frames, heatmaps = install_frame_caches(ns)
    if args.stage in ("attrs", "all"):
        stage_attrs(args, ns, frames, heatmaps)
    if args.stage in ("rules", "all"):
        stage_rules(args, ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
