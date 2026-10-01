from __future__ import annotations

"""Pinned local wrapper around the patched official Biohub metric.

This module intentionally vendors the exact organizer commit instead of relying
on a possibly stale copy bundled in a public notebook support pack.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
VENDOR_SRC = (
    ROOT
    / "vendor"
    / "royerlab_metric"
    / f"source-{METRIC_COMMIT}"
    / f"kaggle-cell-tracking-competition-{METRIC_COMMIT}"
    / "src"
)
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))


SCALE_ZYX = (1.625, 0.40625, 0.40625)
MAX_DISTANCE_UM = 7.0


def _load_official():
    try:
        import tracksdata as td  # type: ignore
        from geff import GeffMetadata  # type: ignore
        from tracking_cellmot.metrics import (  # type: ignore
            evaluate,
            node_recall,
            per_sample_metrics,
            summarise,
        )
    except ImportError as exc:
        raise SystemExit(
            "Official metric dependencies are missing. Install/use the pinned offline "
            "wheels from artifacts/pilkwang_support50 or run inside the Kaggle image. "
            f"Original import error: {exc}"
        ) from exc
    return td, GeffMetadata, evaluate, node_recall, per_sample_metrics, summarise


def _load_graph(td, path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def _estimated_total_nodes(GeffMetadata, path: Path) -> float:
    try:
        meta = GeffMetadata.read(path)
        value = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def evaluate_pair(pred_path: Path, gt_path: Path, scale=SCALE_ZYX, max_distance=MAX_DISTANCE_UM) -> dict:
    td, GeffMetadata, evaluate, node_recall, per_sample_metrics, _ = _load_official()
    pred = _load_graph(td, pred_path)
    gt = _load_graph(td, gt_path)
    result = evaluate(pred, gt, scale=scale, max_distance=max_distance)
    recall = node_recall(pred, gt) if pred.num_nodes() and pred.num_edges() else 0.0
    n_total = _estimated_total_nodes(GeffMetadata, gt_path)
    row = per_sample_metrics(result, n_total, recall)
    div_denom = result.division_tp + result.division_fp + result.division_fn
    row.update(
        {
            "dataset": gt_path.stem,
            "node_count_pred": result.num_pred_nodes,
            "node_count_gt_annotated": gt.num_nodes(),
            "estimated_node_count_total": n_total,
            "division_jaccard": result.division_tp / div_denom if div_denom else float("nan"),
        }
    )
    return row


def evaluate_dirs(pred_dir: Path, gt_dir: Path, scale=SCALE_ZYX, max_distance=MAX_DISTANCE_UM) -> tuple[list[dict], dict]:
    *_, summarise = _load_official()
    pred_names = {p.stem for p in pred_dir.glob("*.geff")}
    gt_names = {p.stem for p in gt_dir.glob("*.geff")}
    names = sorted(pred_names & gt_names)
    rows = [
        evaluate_pair(pred_dir / f"{name}.geff", gt_dir / f"{name}.geff", scale, max_distance)
        for name in names
    ]
    summary = summarise(rows)
    totals = {
        key: sum(int(r[key]) for r in rows)
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes")
    }
    summary = dict(summary)
    summary.update(totals)
    summary["total_score"] = summary.pop("score")
    summary["adjusted_edge_jaccard"] = summary.pop("adj_edge_jaccard")
    summary["raw_edge_jaccard"] = summary.pop("edge_jaccard")
    summary["metric_commit"] = METRIC_COMMIT
    summary["scale_zyx_um"] = list(scale)
    summary["max_distance_um"] = max_distance
    summary["node_count_pred"] = totals["num_pred_nodes"]
    summary["node_count_gt_annotated"] = sum(int(r["node_count_gt_annotated"]) for r in rows)
    return rows, summary


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted GEFF graphs using the pinned patched official metric.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred-dir", type=Path)
    source.add_argument("--csv", type=Path, help="submission CSV; converted with the organizer's csv_to_geffs first")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "evaluation")
    parser.add_argument("--max-distance-um", type=float, default=MAX_DISTANCE_UM)
    args = parser.parse_args()

    pred_dir = args.pred_dir
    if args.csv:
        sys.path.insert(0, str(VENDOR_SRC.parent / "scripts"))
        from csv_to_geffs import csv_to_geffs  # type: ignore

        pred_dir = args.out_dir / "pred_geffs"
        csv_to_geffs(args.csv, pred_dir)
    rows, summary = evaluate_dirs(pred_dir, args.gt_dir, SCALE_ZYX, args.max_distance_um)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
