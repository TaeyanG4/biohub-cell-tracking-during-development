from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "data" / "visible_gt" / "train"
PROBE = ROOT / "experiments" / "strongunet_probe"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
OUT = PROBE / "vs_gt.csv"

STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
RADIUS_UM = 7.0


def load_gt(stem: str) -> pd.DataFrame:
    group = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame(
        {
            "t": np.asarray(group["nodes/props/t/values"][:]),
            "z": np.asarray(group["nodes/props/z/values"][:]),
            "y": np.asarray(group["nodes/props/y/values"][:]),
            "x": np.asarray(group["nodes/props/x/values"][:]),
        }
    )


def score(pred: pd.DataFrame, gt: pd.DataFrame) -> dict[str, float | int]:
    hit = 0
    pred_used = 0
    dists = []
    for t in sorted(set(gt["t"].astype(int)) | set(pred["t"].astype(int))):
        g = gt[gt["t"].astype(int).eq(t)][["z", "y", "x"]].to_numpy(np.float32)
        p = pred[pred["t"].astype(int).eq(t)][["z", "y", "x"]].to_numpy(np.float32)
        if len(g) == 0 or len(p) == 0:
            continue
        g = g * SCALE[None, :]
        p = p * SCALE[None, :]
        d = np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        ok = d[rr, cc] <= RADIUS_UM
        hit += int(ok.sum())
        pred_used += int(ok.sum())
        dists.extend(d[rr[ok], cc[ok]].tolist())
    return {
        "gt_nodes": int(len(gt)),
        "pred_nodes": int(len(pred)),
        "hit": int(hit),
        "miss": int(len(gt) - hit),
        "unmatched_pred": int(len(pred) - pred_used),
        "recall": float(hit / max(len(gt), 1)),
        "precision_like": float(hit / max(len(pred), 1)),
        "mean_match_um": float(np.mean(dists)) if dists else float("nan"),
    }


base = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    gt = load_gt(stem)
    baseline = base[(base["dataset"] == stem) & (base["row_type"] == "node")][
        ["t", "z", "y", "x"]
    ].copy()
    base_stats = score(baseline, gt)
    rows.append({"dataset": stem, "model": "public0946", "threshold": np.nan, **base_stats})

    peaks = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    for thr in THRESHOLDS:
        pred = peaks[peaks["p"] >= thr][["t", "z", "y", "x"]].copy()
        rows.append(
            {
                "dataset": stem,
                "model": "strongunet",
                "threshold": thr,
                **score(pred, gt),
            }
        )

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
print(result.to_string(index=False))

print("AGG")
agg = result.groupby(["model", "threshold"], dropna=False, as_index=False).agg(
    gt_nodes=("gt_nodes", "sum"),
    pred_nodes=("pred_nodes", "sum"),
    hit=("hit", "sum"),
    miss=("miss", "sum"),
    unmatched_pred=("unmatched_pred", "sum"),
)
agg["recall"] = agg["hit"] / agg["gt_nodes"]
agg["precision_like"] = agg["hit"] / agg["pred_nodes"]
print(agg.to_string(index=False))
