"""C058 passive C023 provenance capture and adapters to existing official tools.

No metric is implemented here. Matching uses tracksdata DistanceMatching and
scoring uses the vendored organizer metric through evaluate_local. CSV output
uses C023's actual write_test_submission function with frozen graph input.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import c055_guarded_readmit as baseline
import eval_pp_variants_local as harness
import evaluate_local as official


def namespace():
    """Build the unchanged C023 namespace with original image IO and scorer."""
    ns = harness.build_namespace(baseline.BASE, {}, None)
    ns["TEST_DIR"] = ROOT / "data/train"
    return ns


def _rounded(txyz):
    """The original writer's integer conversion, including nonnegative clamp."""
    return np.array([[int(r[0]), *[max(0, int(round(float(v)))) for v in r[1:]]]
                     for r in np.asarray(txyz)], dtype=np.int64).reshape(-1, 4)


def _plain(ids, txyz):
    return {int(i): tuple(r) for i, r in zip(ids, txyz, strict=True)}


def _organizer_builder():
    official._load_official()
    scripts = str(official.VENDOR_SRC.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from csv_to_geffs import build_graph_from_rows
    return build_graph_from_rows


def _graph_and_ids(ids, txyz, edges):
    """Use the official CSV graph builder and verify its original-ID mapping."""
    import polars as pl
    ids = np.asarray(ids, np.int64)
    points = _rounded(txyz)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    assert len(ids) == len(points) and len(set(ids.tolist())) == len(ids)
    assert np.isfinite(points).all()
    assert set(edges.reshape(-1).tolist()) <= set(ids.tolist()), "dangling edge"
    node_rows = pl.DataFrame({"node_id": ids, "t": points[:, 0], "z": points[:, 1],
                              "y": points[:, 2], "x": points[:, 3]})
    edge_rows = pl.DataFrame({"source_id": edges[:, 0], "target_id": edges[:, 1]})
    graph = _organizer_builder()(node_rows, edge_rows)
    # The organizer builder assigns fresh IDs in node-row insertion order.
    # Verify rather than assuming an initial ID value or retaining only order.
    rows = graph.node_attrs().sort("node_id")
    assigned = rows["node_id"].to_list()
    assert np.array_equal(rows.select(["t", "z", "y", "x"]).to_numpy(), points)
    return graph, dict(zip(assigned, ids.tolist(), strict=True))


def official_graph(ids, txyz, edges):
    """Official CSV-builder graph; node IDs are organizer-assigned internally."""
    return _graph_and_ids(ids, txyz, edges)[0]


def _matches(graph, internal_to_original, td):
    key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    if key not in graph.node_attr_keys():
        return {}
    return {int(internal_to_original[int(r["node_id"])]): int(r[key])
            for r in graph.node_attrs().iter_rows(named=True)
            if r.get(key) is not None and int(r[key]) != -1}


def official_match(ids, txyz, edges, gt_path):
    """Return emitted predicted ID -> original GT ID using actual official matching."""
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options
    td, *_ = official._load_official()
    graph, inverse = _graph_and_ids(ids, txyz, edges)
    gt = official._load_graph(td, Path(gt_path))
    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        graph.match(gt, matching=DistanceMatching(max_distance=official.MAX_DISTANCE_UM,
                                                   scale=official.SCALE_ZYX))
    finally:
        set_options(show_progress=previous)
    return _matches(graph, inverse, td)


def official_score(ids, txyz, edges, gt_path):
    """Return the same official per-movie row as evaluate_local.evaluate_pair."""
    td, GeffMetadata, evaluate, node_recall, per_sample_metrics, _ = official._load_official()
    pred = official_graph(ids, txyz, edges)
    gt_path = Path(gt_path)
    gt = official._load_graph(td, gt_path)
    result = evaluate(pred, gt, scale=official.SCALE_ZYX, max_distance=official.MAX_DISTANCE_UM)
    recall = node_recall(pred, gt) if pred.num_nodes() and pred.num_edges() else 0.0
    n_total = official._estimated_total_nodes(GeffMetadata, gt_path)
    row = per_sample_metrics(result, n_total, recall)
    denom = result.division_tp + result.division_fp + result.division_fn
    row.update(dataset=gt_path.stem, node_count_pred=result.num_pred_nodes,
               node_count_gt_annotated=gt.num_nodes(), estimated_node_count_total=n_total,
               division_jaccard=result.division_tp / denom if denom else float("nan"))
    return row


def official_summary(rows):
    """Use the organizer's existing cross-movie aggregation without a new formula."""
    return official._load_official()[-1](rows)


def capture(stem, split, out, ns=None):
    """Replay C023 unchanged; save exact final integers and actual node provenance."""
    out = Path(out)
    destination = out / "baseline" / f"{stem}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".log")
    reference_path = baseline.DEST / "graphs" / split / f"{stem}_control.npz"
    gt_path = baseline.GT_DIR / f"{stem}.geff"
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        # The notebook's own readmission/gapfill reads its configured dump path.
        if ns is None:
            ns = harness.build_namespace(baseline.BASE, {}, baseline.run_dir(split) / "edge_cache")
            ns["TEST_DIR"] = baseline.GT_DIR
        else:
            # Existing readmission/gapfill functions read this at invocation time.
            os.environ["BIOHUB_CACHE_DIR"] = str(baseline.run_dir(split) / "edge_cache")
        frames, heatmaps = harness.install_frame_caches(ns)
        raw_nodes, raw_edges = harness.load_raw_graph(ns, baseline.pred_path(split, stem))
        # Original recover_strict_gap2 creates interpolated nodes WITHOUT a flag.
        # Observe actual newly inserted objects without changing returned graphs.
        gap2_original=ns['recover_strict_gap2'];gap2_objects={}
        def record_gap2(current,*args,**kwargs):
            before=dict(current)
            result=gap2_original(current,*args,**kwargs)
            gap2_objects.update({i:n for i,n in result[0].items() if i not in before or n is not before[i]})
            return result
        ns['recover_strict_gap2']=record_gap2
        try:
            nodes, kept_edges, stats = ns["filter_output_graph"](
                copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                deepcenter_bundle=ns.get("DEEPCENTER_VETO_DETECTOR"))
        finally:
            ns['recover_strict_gap2']=gap2_original
        ids = np.array(sorted(nodes), np.int64)
        float_txyz = np.array([ns["nodes_by_id_to_plain"]({int(i): nodes[int(i)]})[int(i)]
                              for i in ids], np.float64)
        txyz = _rounded(float_txyz)
        edges = np.array(sorted((int(e["source_id"]), int(e["target_id"]))
                                for e in kept_edges), np.int64).reshape(-1, 2)
        with np.load(reference_path) as ref:
            for name, actual in [("ids", ids), ("txyz", txyz), ("edges", edges)]:
                assert np.array_equal(actual, ref[name]), (stem, "C055 graph drift", name)
        gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](gt_path))
        replica = ns["score_sample"](_plain(ids, txyz), list(map(tuple, edges.tolist())), gt_nodes, gt_edges,
                                     ns["read_estimated_true_node_count"](gt_path))
        replica.update(stem=stem, nodes=len(ids), edges=len(edges))
        reference_row = next(r for r in baseline.read_rows(baseline.DEST / "study" / f"{split}.csv")
                             if r["stem"] == stem and r["config"] == "control")
        baseline.compare_rows([replica], [reference_row], "C058_zero_vs_C055")
        official_mapping = official_match(ids, txyz, edges, gt_path)
        replica_mapping, _ = ns["match_nodes_bipartite"](_plain(ids, txyz), gt_nodes, max_dist=7.0)
        actual_row = official_score(ids, txyz, edges, gt_path)
        flags = {name: np.array([bool(nodes[int(i)].get(name, 0)) for i in ids], bool)
                 for name in ["gap_synthetic", "readmitted", "gapfill_peak"]}
        gap2_synthetic=np.array([int(i) in gap2_objects and nodes[int(i)] is gap2_objects[int(i)] for i in ids],bool)
        explicit_gap_synthetic=flags['gap_synthetic'].copy()
        flags['gap_synthetic'] |= gap2_synthetic
        # _next_node_id can recycle a raw ID removed by an earlier stage. Flags,
        # not numerical membership alone, determine the actual node's origin.
        explicit = flags["gap_synthetic"] | flags["readmitted"] | flags["gapfill_peak"]
        raw_flag = np.array([int(i) in raw_nodes for i in ids], bool) & ~explicit
        # There must be a positively identified inference-time origin for every node.
        provenance_known = raw_flag | flags["gap_synthetic"] | flags["readmitted"] | flags["gapfill_peak"]
        if not provenance_known.all():
            baseline.save_json(destination.with_suffix('.unknown.json'),dict(stem=stem,
                nodes=[nodes[int(i)] for i in ids[~provenance_known]],
                raw_ids_first=sorted(raw_nodes)[:10],raw_count=len(raw_nodes)))
        assert provenance_known.all(), (stem, "unidentified final-node provenance")
        assert not (flags["gap_synthetic"] & (raw_flag | flags["readmitted"] | flags["gapfill_peak"])).any()
        np.savez_compressed(destination, ids=ids, txyz=txyz, edges=edges,
                            float_txyz=float_txyz, original_ilp=raw_flag,
                            gap2_synthetic=gap2_synthetic,explicit_gap_synthetic=explicit_gap_synthetic,**flags)
        mismatch = [int(i) for i in ids if official_mapping.get(int(i)) != replica_mapping.get(int(i))]
        metric_discrepancy = {k: {"official": actual_row[k], "replica": replica[rk]}
                              for k, rk in [("edge_tp", "edge_tp"), ("edge_fp", "edge_fp"),
                                            ("edge_fn", "edge_fn"), ("division_tp", "div_tp"),
                                            ("division_fp", "div_fp"), ("division_fn", "div_fn")]
                              if actual_row[k] != replica[rk]}
        baseline.save_json(destination.with_suffix(".json"), dict(
            status="passed", stem=stem, split=split, exact_c055_ids_txyz_edges=True,
            exact_c055_replica_metric=True, original_notebook_sha256=baseline.sha(baseline.BASE),
            c055_graph_sha256=baseline.sha(reference_path), saved_sha256=baseline.sha(destination),
            provenance={k: int(v.sum()) for k, v in flags.items()}, original_ilp=int(raw_flag.sum()),
            synthetic_exclusion="actual C023 gap_synthetic flag OR passively observed recover_strict_gap2 insertion; no GT-dependent eligibility",
            gap2_synthetic=int(gap2_synthetic.sum()),
            official_metric_commit=official.METRIC_COMMIT, official=actual_row, replica=replica,
            official_vs_replica_metric_discrepancy=metric_discrepancy,
            official_matches=len(official_mapping), replica_matches=len(replica_mapping),
            official_vs_replica_identity_differences=mismatch,
            zero_rounding_exact=True, stats=stats))
        frames.clear()
        heatmaps.clear()
    print(f"C058 captured {stem}: {len(ids)} nodes; synthetic={int(flags['gap_synthetic'].sum())}; "
          f"official/replica identity differences={len(mismatch)}", flush=True)
    return destination


def write_frozen_csv(graphs, split_by_stem, out, ns=None):
    """Invoke C023's unchanged writer on supplied (ids,txyz,edges) frozen graphs.

    ``graphs`` maps movie stem to a tuple (ids, txyz, edges). Coordinates may be
    floating point; all output uses the original integer writer. Original ILP
    GEFFs are copied solely to satisfy that writer's discovery/load interface.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if ns is None:
        ns = namespace()
    writer_root = out / "writer_repo"
    for stem in graphs:
        source = baseline.pred_path(split_by_stem[stem], stem)
        target = writer_root / "predictions/frozen" / ns["METHOD"] / "split_0" / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
    before = ns["filter_output_graph"]

    def frozen_filter(raw_nodes, raw_edges, dataset=None, **_):
        ids, txyz, edges = graphs[dataset]
        nodes = {int(i): dict(node_id=int(i), t=int(r[0]), z=float(r[1]), y=float(r[2]), x=float(r[3]))
                 for i, r in zip(ids, txyz, strict=True)}
        edge_rows = [dict(source_id=int(a), target_id=int(b)) for a, b in edges]
        return nodes, edge_rows, {"raw_edges": len(raw_edges), "repair_fallback": 0}

    ns.update(REPO_DIR=writer_root, test_stems=sorted(graphs), SUBMISSION_PATH=out / "submission.csv",
              RUN_STATS_PATH=out / "run_stats.csv", predict_seconds=0)
    ns["filter_output_graph"] = frozen_filter
    try:
        ns["write_test_submission"]("c058_frozen")
    finally:
        ns["filter_output_graph"] = before
    stats = pd.read_csv(out / "run_stats.csv")
    assert (stats.repair_fallback == 0).all() and (stats.deadline_degraded == 0).all()
    frame = pd.read_csv(out / "submission.csv")
    for stem, (ids, points, edges) in graphs.items():
        movie = frame[frame.dataset == stem]
        observed_nodes = {int(r.node_id): (int(r.t), int(r.z), int(r.y), int(r.x))
                          for r in movie[movie.row_type == "node"].itertuples()}
        observed_edges = sorted((int(r.source_id), int(r.target_id))
                                for r in movie[movie.row_type == "edge"].itertuples())
        assert observed_nodes == _plain(ids, _rounded(points)), (stem, "writer coordinate drift")
        assert observed_edges == sorted(map(tuple, np.asarray(edges).tolist())), (stem, "writer edge drift")
    baseline.save_json(out / "writer_parity.json", dict(status="passed", movies=len(graphs),
                       original_writer=True, repair_fallback=0, deadline_degraded=0,
                       csv_sha256=baseline.sha(out / "submission.csv")))
    return out / "submission.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "capture22"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem")
    parser.add_argument("--split", choices=baseline.SPLIT_ORDER)
    args = parser.parse_args()
    if args.command == "capture":
        assert args.stem and args.split
        capture(args.stem, args.split, args.out)
    else:
        for split, stems in baseline.split_stems().items():
            if split == "extension75":
                continue
            for stem in stems:
                capture(stem, split, args.out)


if __name__ == "__main__":
    main()
