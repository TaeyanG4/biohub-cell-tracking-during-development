from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL = (
    ROOT
    / "vendor"
    / "royerlab_metric"
    / f"source-{COMMIT}"
    / f"kaggle-cell-tracking-competition-{COMMIT}"
)
sys.path.insert(0, str(OFFICIAL / "src"))

from tracking_cellmot.metrics import evaluate


GT_DIR = ROOT / "data" / "visible_gt" / "train"
PRED_DIR = ROOT / "experiments" / "metric_public0947_clean"
OUT_DIR = ROOT / "reports" / "research_20260914"
SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def load_graph(path: Path):
    value = td.graph.IndexedRXGraph.from_geff(path)
    return value[0] if isinstance(value, tuple) else value


def pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def main() -> None:
    keys = td.DEFAULT_ATTR_KEYS
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for gt_path in sorted(GT_DIR.glob("*.geff")):
        dataset = gt_path.stem
        pred_path = PRED_DIR / f"{dataset}.geff"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)

        gt = load_graph(gt_path)
        pred = load_graph(pred_path)
        evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)

        gt_attrs = gt.node_attrs(
            attr_keys=[keys.NODE_ID, keys.Z, keys.Y, keys.X]
        )
        gt_xyz = {
            int(node_id): np.asarray([z, y, x], dtype=np.float64)
            for node_id, z, y, x in gt_attrs.iter_rows()
        }
        pred_attrs = pred.node_attrs(
            attr_keys=[
                keys.NODE_ID,
                keys.MATCHED_NODE_ID,
                keys.Z,
                keys.Y,
                keys.X,
            ]
        )

        local = []
        for node_id, matched_id, z, y, x in pred_attrs.iter_rows():
            if matched_id is None or int(matched_id) < 0:
                continue
            matched_id = int(matched_id)
            if matched_id not in gt_xyz:
                continue
            pred_zyx = np.asarray([z, y, x], dtype=np.float64)
            delta_px = pred_zyx - gt_xyz[matched_id]
            delta_um = delta_px * SCALE
            distance_um = float(np.linalg.norm(delta_um))
            row = {
                "dataset": dataset,
                "pred_node_id": int(node_id),
                "gt_node_id": matched_id,
                "dz_px": float(delta_px[0]),
                "dy_px": float(delta_px[1]),
                "dx_px": float(delta_px[2]),
                "dz_um": float(delta_um[0]),
                "dy_um": float(delta_um[1]),
                "dx_um": float(delta_um[2]),
                "distance_um": distance_um,
            }
            rows.append(row)
            local.append(row)

        d = np.asarray([r["distance_um"] for r in local], dtype=np.float64)
        summary_rows.append(
            {
                "dataset": dataset,
                "embryo": dataset.split("_", 1)[0],
                "matched": int(len(local)),
                "distance_mean_um": float(d.mean()) if len(d) else float("nan"),
                "distance_p50_um": pct(d, 0.50),
                "distance_p90_um": pct(d, 0.90),
                "distance_p95_um": pct(d, 0.95),
                "distance_p99_um": pct(d, 0.99),
            }
        )

    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "localization_residual_public0947.csv", index=False)
    summary.to_csv(OUT_DIR / "localization_residual_public0947_by_video.csv", index=False)

    delta = frame[["dz_um", "dy_um", "dx_um"]].to_numpy(dtype=np.float64)
    dist = frame["distance_um"].to_numpy(dtype=np.float64)
    aggregate = {
        "matched": int(len(frame)),
        "distance_mean_um": float(dist.mean()),
        "distance_p50_um": pct(dist, 0.50),
        "distance_p90_um": pct(dist, 0.90),
        "distance_p95_um": pct(dist, 0.95),
        "distance_p99_um": pct(dist, 0.99),
        "axis_signed_mean_um": delta.mean(axis=0).tolist(),
        "axis_std_um": delta.std(axis=0).tolist(),
        "axis_abs_p90_um": [pct(np.abs(delta[:, i]), 0.90) for i in range(3)],
        "axis_abs_p95_um": [pct(np.abs(delta[:, i]), 0.95) for i in range(3)],
        "scale_zyx_um_per_px": SCALE.tolist(),
        "by_video": summary_rows,
    }
    (OUT_DIR / "localization_residual_public0947.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
