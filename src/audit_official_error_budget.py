"""Diagnostic only: decompose official edge FNs without changing predictions.

GT is used only after the prediction graphs are fixed. This is NOT OOF validation.
Do not use the saved GT-matched rows as test-time features or submission edits.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL = ROOT / "vendor" / "royerlab_metric" / f"source-{COMMIT}" / f"kaggle-cell-tracking-competition-{COMMIT}"
sys.path.insert(0, str(OFFICIAL / "src"))

import pandas as pd
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

GT_DIR = ROOT / "data" / "visible_gt" / "train"
SCALE = (1.625, 0.40625, 0.40625)
PREDICTIONS = {
    "public0946": ROOT / "experiments" / "metric_public0946",
    "dctta0965": ROOT / "experiments" / "metric_dctta0965",
    "hoct0965": ROOT / "experiments" / "metric_hoct0965",
    "public_plus_toql4_cap4": ROOT / "experiments" / "stabledet_visible_runtime" / "public_plus_toql4_cap4",
}


def load_graph(path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def main() -> None:
    expected = {p.stem for p in GT_DIR.glob("*.geff")}
    if len(expected) != 4:
        raise RuntimeError(f"Expected the four diagnostic videos, found {expected}")
    keys = td.DEFAULT_ATTR_KEYS
    all_rows, all_missing, summaries = [], [], {}
    for name, directory in PREDICTIONS.items():
        found = {p.stem for p in directory.glob("*.geff")}
        if found != expected:
            raise RuntimeError(f"Coverage mismatch for {name}: {found ^ expected}")
        model_rows = []
        for sample in sorted(expected):
            gt_path = GT_DIR / f"{sample}.geff"
            gt = load_graph(gt_path)
            pred = load_graph(directory / f"{sample}.geff")
            er = evaluate(pred, gt, scale=SCALE, max_distance=7.0)
            recall = node_recall(pred, gt)
            metadata = GeffMetadata.read(gt_path)
            n_total = float((metadata.extra or {})["estimated_number_of_nodes"])
            row = per_sample_metrics(er, n_total, recall)
            attrs = pred.node_attrs(attr_keys=[keys.NODE_ID, keys.MATCHED_NODE_ID])
            mapping = {
                int(p): int(g)
                for p, g in attrs.iter_rows()
                if g is not None and int(g) >= 0
            }
            matched_gt = set(mapping.values())
            mapped_edges = set()
            times = dict(pred.node_attrs(attr_keys=[keys.NODE_ID, keys.T]).iter_rows())
            for s, t in pred.edge_attrs().select([keys.EDGE_SOURCE, keys.EDGE_TARGET]).iter_rows():
                s, t = int(s), int(t)
                if s in mapping and t in mapping and times[t] == times[s] + 1:
                    mapped_edges.add((mapping[s], mapping[t]))
            gt_edges = {
                (int(s), int(t))
                for s, t in gt.edge_attrs().select([keys.EDGE_SOURCE, keys.EDGE_TARGET]).iter_rows()
            }
            recovered = gt_edges & mapped_edges
            if len(recovered) != er.edge_tp:
                raise RuntimeError(f"TP decomposition mismatch {name}/{sample}: {len(recovered)} != {er.edge_tp}")
            counts = Counter()
            for s, t in sorted(gt_edges - recovered):
                if s not in matched_gt and t not in matched_gt:
                    category = "both_endpoints_missing"
                elif s not in matched_gt:
                    category = "source_endpoint_missing"
                elif t not in matched_gt:
                    category = "target_endpoint_missing"
                else:
                    category = "both_endpoints_present_wrong_or_missing_link"
                counts[category] += 1
                all_missing.append({"model": name, "dataset": sample, "gt_source_id": s, "gt_target_id": t, "category": category})
            if sum(counts.values()) != er.edge_fn:
                raise RuntimeError("FN decomposition does not sum to official FN count")
            missing_endpoint_fn = sum(v for k, v in counts.items() if k != "both_endpoints_present_wrong_or_missing_link")
            full = {
                "model": name, "dataset": sample, "embryo": sample.split("_", 1)[0],
                **row, "gt_nodes": gt.num_nodes(), "gt_edges": gt.num_edges(),
                "estimated_total_nodes": n_total,
                "fn_missing_endpoint": missing_endpoint_fn,
                "fn_both_endpoints_present": counts["both_endpoints_present_wrong_or_missing_link"],
                "fn_both_endpoints_missing": counts["both_endpoints_missing"],
            }
            all_rows.append(full)
            model_rows.append(full)
        summaries[name] = {
            **summarise(model_rows),
            "edge_tp": sum(r["edge_tp"] for r in model_rows),
            "edge_fp": sum(r["edge_fp"] for r in model_rows),
            "edge_fn": sum(r["edge_fn"] for r in model_rows),
            "fn_missing_endpoint": sum(r["fn_missing_endpoint"] for r in model_rows),
            "fn_both_endpoints_present": sum(r["fn_both_endpoints_present"] for r in model_rows),
            "gt_nodes": sum(r["gt_nodes"] for r in model_rows),
            "gt_edges": sum(r["gt_edges"] for r in model_rows),
        }
        print(name, json.dumps(summaries[name], sort_keys=True), flush=True)
    out = ROOT / "reports" / "research_20260913"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "official_error_budget_per_video.csv", index=False)
    pd.DataFrame(all_missing).to_csv(out / "diagnostic_gt_missing_edges.csv", index=False)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_commit": COMMIT,
        "metric_sha256": hashlib.sha256((OFFICIAL / "src" / "tracking_cellmot" / "metrics.py").read_bytes()).hexdigest(),
        "validation_kind": "four_visible_training_copies_diagnostic_only_NOT_OOF",
        "coverage": sorted(expected),
        "embryos": sorted({x.split("_", 1)[0] for x in expected}),
        "summaries": summaries,
        "warnings": [
            "Public example videos are copies of training videos; checkpoint exposure is not excluded.",
            "Repeated tuning on these four videos does not establish generalization.",
            "Unmatched predictions under sparse annotations are not automatically false detections.",
            "Nearest baseline node to a missed GT cell can be another real cell, not a localization error.",
            "GT-based candidate selection in analyze_miss_feature_ranking.py is oracle-only, not deployable.",
        ],
    }
    (out / "official_error_budget.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
