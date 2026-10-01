#!/usr/bin/env python3
"""C056 stage: V12 guarded endpoint readmission fed by a COMPLEMENTARY detector.

Same guard, constants and insertion point as C055 (`c055_guarded_readmit_stage`,
V12 verbatim), but the candidates are the retained Hengck StrongUNet peaks
(`src/cache_strongunet_gpu_peaks.py` model / preprocessing, fixed published
evaluation threshold p >= 0.5) instead of the primary detector's 0.94-0.965
range.  Motivation (state/c055_diagnostics_20260929): of the GT nodes the C023
control misses, 101/213 have no primary-detector peak at all, while the
priority review found StrongUNet peaks at 16 of 21 such nodes on the visible
movies.  Existing nodes' own peaks are excluded by the same one-to-one identity
rule as C055 (<= 2.5 um) before V12's separation test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c055_guarded_readmit_stage import exclude_node_peaks, make_readmit

STRONGUNET_MIN_P = 0.5   # published evaluation threshold, not selected on these labels


def load_strongunet_candidates(peaks_dir, dataset, min_p=STRONGUNET_MIN_P):
    """StrongUNet peaks (t, z, y, x already in the node voxel frame, p) at p >= min_p,
    one entry per rounded (t, z, y, x) keeping the maximum score (V12 loader semantics)."""
    path = Path(peaks_dir) / f"{dataset}_peaks.parquet"
    if not path.exists():
        raise RuntimeError(f"StrongUNet peak cache missing for {dataset}: {path}")
    data = pd.read_parquet(path)
    data = data[data.p >= min_p]
    result = {}
    for t, z, y, x, p in data[["t", "z", "y", "x", "p"]].itertuples(index=False):
        key = tuple(int(round(v)) for v in (t, z, y, x))
        result[key] = max(result.get(key, 0.0), float(p))
    return [{"t": t, "z": z, "y": y, "x": x, "readmit_score": s} for (t, z, y, x), s in result.items()]


def install_complementary_readmit(ns, peaks_dir, graph_folder=None):
    """Like c055_guarded_readmit_stage.install_guarded_readmit, with StrongUNet candidates.
    Inert while ns['COMPLEMENTARY_READMIT'] is 0."""
    ns.setdefault("COMPLEMENTARY_READMIT", 0)
    ns.setdefault("_C056_LABEL", "unlabelled")
    ns["_C056_RECORDS"] = {}
    readmit = make_readmit(ns)
    close_gaps = ns["close_single_frame_gaps"]
    outer = ns["filter_output_graph"]
    last = {}

    def close_after_guarded(nodes_by_id, edges, stats, *args, **kwargs):
        last.clear()
        if int(ns["COMPLEMENTARY_READMIT"]):
            dataset = kwargs.get("dataset")
            candidates = exclude_node_peaks(load_strongunet_candidates(peaks_dir, dataset), nodes_by_id, stats)
            before_nodes, before_edges = len(nodes_by_id), len(edges)
            nodes_by_id, edges = readmit(nodes_by_id, edges, stats, candidates)
            print(f"  [{dataset}] complementary guarded readmit: candidates={stats['readmit_candidates']} "
                  f"(all p>={STRONGUNET_MIN_P} {stats['readmit_candidates_all']}, node peaks excluded {stats['readmit_node_peaks_excluded']}) "
                  f"proposals={stats['readmit_proposals']} budget={stats['readmit_budget']} added={stats['readmit_nodes']} "
                  f"bridges={stats['readmit_bridges']} edges={stats['readmit_edges']} nodes {before_nodes}->{len(nodes_by_id)} edges {before_edges}->{len(edges)}")
            last.update(added_ids=list(stats["readmit_added_ids"]), added_edges=list(stats["readmit_added_edges"]),
                        candidates=stats["readmit_candidates"], candidates_all=stats["readmit_candidates_all"],
                        node_peaks_excluded=stats["readmit_node_peaks_excluded"], proposals=stats["readmit_proposals"],
                        budget=stats["readmit_budget"], bridges=stats["readmit_bridges"],
                        node_peak_unassigned=stats["readmit_node_peak_unassigned"], node_peak_conflicts=stats["readmit_node_peak_conflicts"])
        return close_gaps(nodes_by_id, edges, stats, *args, **kwargs)

    def filter_with_record(nodes_by_id, raw_edges, *args, **kwargs):
        dataset = kwargs.get("dataset")
        nodes, edges, stats = outer(nodes_by_id, raw_edges, *args, **kwargs)
        label = str(ns["_C056_LABEL"])
        added = set(last.get("added_ids", []))
        added_edges = set(tuple(e) for e in last.get("added_edges", []))
        final_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
        record = dict(stem=dataset, label=label,
                      readmit_candidates=int(last.get("candidates", 0)), readmit_candidates_all=int(last.get("candidates_all", 0)),
                      readmit_node_peaks_excluded=int(last.get("node_peaks_excluded", 0)),
                      readmit_node_peak_unassigned=int(last.get("node_peak_unassigned", 0)),
                      readmit_node_peak_conflicts=int(last.get("node_peak_conflicts", 0)),
                      readmit_proposals=int(last.get("proposals", 0)), readmit_budget=int(last.get("budget", 0)),
                      readmit_nodes=len(added), readmit_bridges=int(last.get("bridges", 0)), readmit_edges=len(added_edges),
                      readmit_nodes_final=sum(1 for i in added if i in nodes),
                      readmit_edges_final=sum(1 for p in added_edges if p in final_pairs),
                      readmit_incident_final=sum(1 for s, t in final_pairs if s in added or t in added),
                      final_nodes=len(nodes), final_edges=len(edges))
        for key in ("readmitted_nodes", "gapfill_added_nodes", "safe_divisions_added", "v1057_restored", "v1057_displaced"):
            record["stat_" + key] = int(stats.get(key, 0))
        ns["_C056_RECORDS"][(dataset, label)] = record
        if graph_folder is not None:
            folder = Path(graph_folder)
            folder.mkdir(parents=True, exist_ok=True)
            plain = ns["nodes_by_id_to_plain"](nodes)
            ids = sorted(plain)
            np.savez_compressed(folder / f"{dataset}_{label}.npz", ids=np.array(ids, np.int64),
                                txyz=np.array([[int(plain[i][0]), *[max(0, int(round(v))) for v in plain[i][1:]]] for i in ids], np.int64),
                                edges=np.array(sorted(final_pairs), np.int64).reshape(-1, 2),
                                readmitted_ids=np.array(sorted(added), np.int64))
        return nodes, edges, stats

    ns["close_single_frame_gaps"] = close_after_guarded
    ns["filter_output_graph"] = filter_with_record
