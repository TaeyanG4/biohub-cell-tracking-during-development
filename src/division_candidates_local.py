#!/usr/bin/env python3
"""Generate division-candidate triples with features and GT labels from local C012-style runs.

Replays the notebook's post-processing through x138's safe-division stage (which runs
unchanged - C016 is additive), captures the graph it leaves behind and enumerates candidates and
features with src/division_scorer_stage.py - the same code the C016 notebook embeds - so
training features equal inference features.

Labels mirror the official metric (tracking_cellmot.division_metrics.score_divisions):
   1  the fork at P would be paired with a GT division: P or its predecessor matches the GT
      dividing node or its parent, and the two GT daughter lineages (daughter or grand-daughter)
      are matched in the two distinct predicted branches {D1 + children(D1)}, {D2 + children(D2)};
   0  a false-positive fork: considered for a GT division but not paired, or P matched to a GT
      node that has at least one child (evaluable fork), or the two branches carry matched
      evidence from two different GT weak components (cross-component fork);
  -1  not counted by the metric (P unmatched or matched to a GT track end, no cross-component
      evidence); excluded from training.

    python src/division_candidates_local.py --notebook <nb> --runs <e2e_dir>[,<e2e_dir>...] --out <csv>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from division_scorer_stage import FEATURES, enumerate_candidates, load_dump, make_intensity_fn  # noqa: E402
from eval_pp_variants_local import DEFAULT_GT_DIR, build_namespace, install_frame_caches, load_raw_graph  # noqa: E402


class MetricLabeller:
    """Label a (P, D1, D2) fork the way score_divisions() would count it (see module docstring)."""

    def __init__(self, gt_nodes, gt_edges, nodes, edges, p2g):
        self.p2g = p2g
        self.gt_out: dict[int, list] = {}
        self.gt_parent: dict[int, int] = {}
        for s, t in gt_edges:
            self.gt_out.setdefault(int(s), []).append(int(t))
            self.gt_parent[int(t)] = int(s)
        comp = {int(n): int(n) for n in gt_nodes}

        def find(a):
            while comp[a] != a:
                comp[a] = comp[comp[a]]
                a = comp[a]
            return a
        for s, t in gt_edges:
            ra, rb = find(int(s)), find(int(t))
            if ra != rb:
                comp[ra] = rb
        self.comp = {n: find(n) for n in comp}
        self.gt_divs = {g for g, kids in self.gt_out.items() if len(kids) >= 2}
        self.daughters = {g: set(self.gt_out[g]) for g in self.gt_divs}
        self.lineages = {g: [{a, *self.gt_out.get(a, [])} for a in self.gt_out[g]] for g in self.gt_divs}
        self.window = {g: {g, self.gt_parent[g]} if g in self.gt_parent else {g} for g in self.gt_divs}
        self.pred_in: dict[int, int] = {}
        self.pred_out: dict[int, list] = {}
        for e in edges:
            self.pred_in[int(e["target_id"])] = int(e["source_id"])
            self.pred_out.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))

    def branch_matches(self, d):
        return {self.p2g[n] for n in (d, *self.pred_out.get(d, [])) if n in self.p2g}

    def branch_component(self, d):
        if d in self.p2g:
            return self.comp[self.p2g[d]]
        comps = {self.comp[self.p2g[n]] for n in self.pred_out.get(d, []) if n in self.p2g}
        return next(iter(comps)) if len(comps) == 1 else None

    def label(self, P, D1, D2) -> tuple[int, str]:
        """(label, kind). Positive kinds: 'strict' = P matches the divider and D1/D2 the two daughters directly,
        'early' = the fork sits at the divider's parent (or P's predecessor carries the match), 'grand' = a daughter
        lineage is only matched through a grand-daughter. Negative kinds: 'considered', 'evaluable', 'cross'."""
        gp = self.p2g.get(P)
        side = {gp, self.p2g.get(self.pred_in.get(P))} - {None}
        considered = [g for g in self.gt_divs if side & self.window[g]]
        if considered:
            l1, l2 = self.branch_matches(D1), self.branch_matches(D2)
            m1, m2 = self.p2g.get(D1), self.p2g.get(D2)
            for g in considered:
                lins = self.lineages[g]
                for i in range(len(lins)):
                    for j in range(len(lins)):
                        if i != j and (l1 & lins[i]) and (l2 & lins[j]):
                            if gp != g:
                                return 1, "early"
                            direct = m1 in self.daughters[g] and m2 in self.daughters[g] and m1 != m2
                            return 1, ("strict" if direct else "grand")
            return 0, "considered"
        if gp is not None and len(self.gt_out.get(gp, [])) >= 1:
            return 0, "evaluable"
        c1, c2 = self.branch_component(D1), self.branch_component(D2)
        return (0, "cross") if (c1 is not None and c2 is not None and c1 != c2) else (-1, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run dirs, each with predictions/ and edge_cache/")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", default="", help="comma-separated subset of stems (default: every .geff in the runs)")
    args = parser.parse_args()
    only = {s for s in args.stems.split(",") if s}

    log_path = args.out.with_suffix(".log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = build_namespace(args.notebook.resolve(), {}, None)
    ns["TEST_DIR"] = args.gt_dir.resolve()
    frames, heatmaps = install_frame_caches(ns)
    captured: dict = {}
    rule_based_safe_div = ns["add_safe_divisions_postlink"]

    def capture_safe_div(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        # C016 is additive: x138's rule runs unchanged first, the scorer only sees the graph it leaves behind
        # (parents already forked by the rule are no longer single-child candidates). Capture that graph.
        edges = rule_based_safe_div(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                                    frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)
        captured["nodes"] = {k: dict(v) for k, v in nodes_by_id.items()}
        captured["edges"] = [dict(e) for e in edges]
        return edges
    ns["add_safe_divisions_postlink"] = capture_safe_div
    bundle = ns.get("DEEPCENTER_VETO_DETECTOR")

    rows_out: list[dict] = []
    with log_path.open("a", encoding="utf-8") as log:
        for run in [Path(r) for r in args.runs.split(",") if r]:
            for pred_path in sorted((run / "predictions").glob("*.geff")):
                stem = pred_path.stem
                cache_path = run / "edge_cache" / f"{stem}.npz"
                if not cache_path.exists() or (only and stem not in only):
                    continue
                t_start = time.time()
                os.environ["BIOHUB_CACHE_DIR"] = str(run / "edge_cache")
                raw_nodes, raw_edges = load_raw_graph(ns, pred_path)
                gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](args.gt_dir / f"{stem}.geff"))
                with contextlib.redirect_stdout(log):
                    ns["filter_output_graph"](copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=bundle)
                nodes, edges = captured["nodes"], captured["edges"]
                admitted, low_by_t = load_dump(np.load(cache_path))
                frame_cache: dict = {}

                def dc_point(node):
                    v = ns["deepcenter_score_point"](stem, int(node["t"]), (float(node["z"]), float(node["y"]), float(node["x"])),
                                                     bundle, frame_cache, heatmaps)
                    return float(v) if v is not None else 0.0

                t_pp = time.time()
                intensity = make_intensity_fn(stem, ns["read_test_frame"], frame_cache)
                cands = enumerate_candidates(nodes, edges, admitted, low_by_t, dc_point, intensity)
                t_enum = time.time()
                plain ={nid: (int(n["t"]), *(max(0, int(round(float(n[k])))) for k in ("z", "y", "x"))) for nid, n in nodes.items()}
                p2g, _ = ns["match_nodes_bipartite"](plain, gt_nodes, max_dist=7.0)
                labeller = MetricLabeller(gt_nodes, gt_edges, nodes, edges, p2g)
                n_pos = 0
                for c in cands:
                    label, kind = labeller.label(c["P"], c["D1"], c["D2"])
                    n_pos += label == 1
                    rows_out.append({"stem": stem, "P": c["P"], "D1": c["D1"], "D2": c["D2"], "t": c["t"], "label": label,
                                     "label_kind": kind, **{f: c[f] for f in FEATURES}})
                print(f"{stem}: candidates {len(cands)} labelled {sum(1 for c in cands if p2g.get(c['P']) is not None)} positives {n_pos} "
                      f"| post-processing {t_pp - t_start:.0f}s enumerate {t_enum - t_pp:.0f}s label {time.time() - t_enum:.0f}s", flush=True)
                frames.clear(); heatmaps.clear()
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print(f"\nTOTAL candidates {len(rows_out)} | labelled {sum(r['label'] >= 0 for r in rows_out)} | positives {sum(r['label'] == 1 for r in rows_out)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
