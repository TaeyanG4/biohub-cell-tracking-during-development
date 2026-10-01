#!/usr/bin/env python3
"""Why are GT divisions missed? Replay a notebook's post-processing and classify each GT division.

For every GT division (a GT node with >= 2 children) on the given movies, after the
notebook's own post-processing (same replay as src/eval_pp_variants_local.py):

  detection_miss   parent or a daughter has no predicted node within the 7 um match
  no_fork          parent (or its predecessor, as the metric allows) is matched but the
                   predicted graph has out-degree <= 1 there
  wrong_daughters  a fork exists at the anchor but the daughters' lineages are not the
                   ones the metric expects
  tp               counted as a division true positive

Also lists predicted forks (out-degree 2) whose parent matches a GT node that does not
divide (division false positives) and forks with an unmatched parent (ignored by the metric).

    python src/diagnose_divisions_local.py --notebook <nb> --pred-root <dir> --lowdet-dir <dir> --stems all12
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pp_variants_local import (  # noqa: E402
    DEFAULT_GT_DIR, STEM_SETS, build_namespace, install_frame_caches, load_raw_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--lowdet-dir", type=Path, default=None)
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

    records, fp_records = [], []
    with log_path.open("a", encoding="utf-8") as log:
        for stem in stems:
            pred_path = next(args.pred_root.rglob(f"{stem}.geff"), None)
            gt_path = args.gt_dir / f"{stem}.geff"
            raw_nodes, raw_edges = load_raw_graph(ns, pred_path)
            gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](gt_path))
            with contextlib.redirect_stdout(log):
                nodes, edges, _ = ns["filter_output_graph"](copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                                                            deepcenter_bundle=ns.get("DEEPCENTER_VETO_DETECTOR"))
            pred_nodes = ns["nodes_by_id_to_plain"](nodes)
            pred_nodes = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in pred_nodes.items()}
            pred_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
            p2g, g2p = ns["match_nodes_bipartite"](pred_nodes, gt_nodes, max_dist=7.0)
            gt_out: dict[int, set] = {}
            gt_in: dict[int, int] = {}
            for s, t in gt_edges:
                gt_out.setdefault(s, set()).add(t); gt_in[t] = s
            pred_out: dict[int, set] = {}
            for s, t in pred_edges:
                pred_out.setdefault(s, set()).add(t)
            tp, fp, fn = ns["compute_division_confusion"](pred_nodes, pred_edges, gt_nodes, gt_edges, p2g, g2p)
            # Which GT divisions count as TP? Re-run the metric's rule per division.
            components = ns["weakly_connected_components"](list(pred_nodes), pred_edges)
            fork_comps = {components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components}

            def lineage(child):
                seen, stack = {child}, [child]
                while stack:
                    cur = stack.pop()
                    for nxt in gt_out.get(cur, ()):
                        if nxt not in seen:
                            seen.add(nxt); stack.append(nxt)
                return seen

            for gsrc, children in gt_out.items():
                if len(children) < 2:
                    continue
                kids = sorted(children)[:2]
                parent_matched = gsrc in g2p
                prev_matched = gt_in.get(gsrc) in g2p if gsrc in gt_in else False
                daughters_matched = [k in g2p for k in kids]
                anchors = [g2p[a] for a in ([gsrc] + ([gt_in[gsrc]] if gsrc in gt_in else [])) if a in g2p]
                fork_at_anchor = any(len(pred_out.get(a, ())) >= 2 for a in anchors)
                hit = []
                for k in kids:
                    hit.append({components[p] for g in lineage(k) if (p := g2p.get(g)) is not None and p in components})
                counted_tp = bool(anchors) and all(hit) and any(c in hit[0] and c in hit[1] and c in fork_comps for c in {components[a] for a in anchors})
                if counted_tp:
                    kind = "tp"
                elif not (parent_matched or prev_matched) or not all(daughters_matched):
                    kind = "detection_miss"
                elif not fork_at_anchor:
                    kind = "no_fork"
                else:
                    kind = "wrong_daughters"
                records.append({"stem": stem, "gt_parent": gsrc, "t": gt_nodes[gsrc][0], "kind": kind,
                                "parent_matched": parent_matched, "prev_matched": prev_matched,
                                "daughters_matched": sum(daughters_matched),
                                "pred_outdeg_at_parent": len(pred_out.get(g2p.get(gsrc, -1), ())),
                                "pred_outdeg_at_prev": len(pred_out.get(g2p.get(gt_in.get(gsrc, -1), -1), ()))})
            for n, outs in pred_out.items():
                if len(outs) >= 2:
                    g = p2g.get(n)
                    status = "unmatched_parent" if g is None else ("gt_divides" if g in gt_out and len(gt_out[g]) >= 2 else ("gt_no_division" if g in gt_out else "gt_leaf"))
                    fp_records.append({"stem": stem, "pred_parent": n, "t": pred_nodes[n][0], "status": status})
            print(f"{stem}: metric div tp/fp/fn={tp}/{fp}/{fn} | GT divisions {sum(1 for c in gt_out.values() if len(c) >= 2)}: "
                  + ", ".join(f"{k}={v}" for k, v in Counter(r['kind'] for r in records if r['stem'] == stem).items())
                  + f" | predicted forks {sum(1 for r in fp_records if r['stem'] == stem)} "
                  + str(dict(Counter(r['status'] for r in fp_records if r['stem'] == stem))), flush=True)
            frames.clear(); heatmaps.clear()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    with args.out.with_name(args.out.stem + "_forks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_records[0].keys())); w.writeheader(); w.writerows(fp_records)
    print("\nGT divisions by outcome:", dict(Counter(r["kind"] for r in records)))
    print("of detection misses: parent/prev unmatched", sum(1 for r in records if r["kind"] == "detection_miss" and not (r["parent_matched"] or r["prev_matched"])),
          "| daughters unmatched", sum(1 for r in records if r["kind"] == "detection_miss" and r["daughters_matched"] < 2))
    print("predicted forks by status:", dict(Counter(r["status"] for r in fp_records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
