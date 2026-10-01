#!/usr/bin/env python3
"""Detail the 'no_fork' GT divisions: daughters detected, but no fork in the output graph.

For each GT division whose parent (or its predecessor) and both daughters have matched
predicted nodes but the output has out-degree <= 1 at the anchor, report:

  * which daughter is linked to the parent and where the other daughter's incoming edge
    comes from (another track / nothing),
  * parent->daughter and sister distances (um) on the predicted nodes,
  * whether x138's safe-division geometry would admit the triple
    (parent <= SAFE_DIV_MAX_UM, sister <= SAFE_DIV_SISTER_MAX_UM),
  * the learned edge probability of parent->daughter in the inference candidate graph
    (edge_cache 'admitted': candidate edges above the 0.48 threshold), if present.

    python src/diagnose_divisions_detail.py --notebook <nb> --pred-root <dir> --lowdet-dir <dir> --stems <list> --out <csv>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pp_variants_local import (  # noqa: E402
    DEFAULT_GT_DIR, STEM_SETS, build_namespace, install_frame_caches, load_raw_graph,
)

VOX = np.array([1.625, 0.40625, 0.40625])


def um(a, b):
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * VOX))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--lowdet-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--stems", default="all12")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stems = STEM_SETS.get(args.stems) or [s.strip() for s in args.stems.split(",") if s.strip()]

    log_path = args.out.with_suffix(".log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = build_namespace(args.notebook.resolve(), {}, args.lowdet_dir)
    ns["TEST_DIR"] = args.gt_dir.resolve()
    frames, heatmaps = install_frame_caches(ns)
    div_max, sister_max = float(ns["SAFE_DIV_MAX_UM"]), float(ns["SAFE_DIV_SISTER_MAX_UM"])

    out_rows = []
    with log_path.open("a", encoding="utf-8") as log:
        for stem in stems:
            raw_nodes, raw_edges = load_raw_graph(ns, next(args.pred_root.rglob(f"{stem}.geff")))
            gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](args.gt_dir / f"{stem}.geff"))
            with contextlib.redirect_stdout(log):
                nodes, edges, _ = ns["filter_output_graph"](copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                                                            deepcenter_bundle=ns.get("DEEPCENTER_VETO_DETECTOR"))
            pred_plain = ns["nodes_by_id_to_plain"](nodes)
            pred_round = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in pred_plain.items()}
            pred_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
            p2g, g2p = ns["match_nodes_bipartite"](pred_round, gt_nodes, max_dist=7.0)
            gt_out, gt_in = {}, {}
            for s, t in gt_edges:
                gt_out.setdefault(s, set()).add(t); gt_in[t] = s
            pred_out, pred_in = {}, {}
            for s, t in pred_edges:
                pred_out.setdefault(s, set()).add(t); pred_in.setdefault(t, []).append(s)
            # candidate edge probabilities from inference (node ids == coords row indices)
            cache = np.load(args.lowdet_dir / f"{stem}.npz")
            cand = {(int(a), int(b)): float(p) for a, b, p, _ in cache["admitted"]}
            coords = cache["coords"]
            ids_match = all(abs(coords[nid][1:] - np.asarray(raw_nodes[nid]["z"], float)).size for nid in list(raw_nodes)[:1]) and \
                all(np.allclose(coords[nid][1:], [raw_nodes[nid]["z"], raw_nodes[nid]["y"], raw_nodes[nid]["x"]], atol=1e-3) for nid in list(raw_nodes)[:50])
            for gsrc, children in gt_out.items():
                if len(children) < 2:
                    continue
                kids = sorted(children)[:2]
                anchors = [g2p[a] for a in ([gsrc] + ([gt_in[gsrc]] if gsrc in gt_in else [])) if a in g2p]
                if not anchors or not all(k in g2p for k in kids):
                    continue
                if any(len(pred_out.get(a, ())) >= 2 for a in anchors):
                    continue  # fork exists (tp or wrong daughters) - not a no_fork case
                P = g2p[gsrc] if gsrc in g2p else anchors[0]
                D = [g2p[k] for k in kids]
                pp = pred_plain[P][1:]
                dist_pd = [um(pp, pred_plain[d][1:]) for d in D]
                sister = um(pred_plain[D[0]][1:], pred_plain[D[1]][1:])
                linked = [P in pred_in.get(d, []) for d in D]
                other_src = [next((s for s in pred_in.get(d, []) if s != P), None) for d in D]
                probs = [cand.get((P, d)) if ids_match else None for d in D]
                out_rows.append({
                    "stem": stem, "gt_parent": gsrc, "t": gt_nodes[gsrc][0], "pred_parent": P,
                    "d1_linked_to_parent": linked[0], "d2_linked_to_parent": linked[1],
                    "d1_other_parent": other_src[0], "d2_other_parent": other_src[1],
                    "dist_p_d1_um": round(dist_pd[0], 2), "dist_p_d2_um": round(dist_pd[1], 2), "sister_um": round(sister, 2),
                    "geometry_admits": max(dist_pd) <= div_max and sister <= sister_max,
                    "cand_prob_p_d1": probs[0], "cand_prob_p_d2": probs[1],
                    "parent_outdeg": len(pred_out.get(P, ())), "parent_t_same_as_gt": pred_plain[P][0] == gt_nodes[gsrc][0],
                })
            frames.clear(); heatmaps.clear()
            print(f"{stem}: {sum(1 for r in out_rows if r['stem'] == stem)} no_fork divisions detailed (ids_match={ids_match})", flush=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"\n{'stem':15s} {'t':>3s} d1link d2link  otherP1 otherP2   dP-D1  dP-D2 sister geom  prob1  prob2")
    for r in out_rows:
        print(f"{r['stem']:15s} {r['t']:>3d} {str(r['d1_linked_to_parent']):6s} {str(r['d2_linked_to_parent']):6s}  "
              f"{str(r['d1_other_parent']):8s}{str(r['d2_other_parent']):8s} {r['dist_p_d1_um']:6.2f} {r['dist_p_d2_um']:6.2f} {r['sister_um']:6.2f} "
              f"{str(r['geometry_admits']):5s} {str(r['cand_prob_p_d1'])[:5]:6s} {str(r['cand_prob_p_d2'])[:5]:6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
