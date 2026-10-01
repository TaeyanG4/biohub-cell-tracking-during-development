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
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
RADIUS = 7.0
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
TEMP_GATES = (4.0, 6.0, 8.0)
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def gt_df(stem: str) -> pd.DataFrame:
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame({
        "t": np.asarray(g["nodes/props/t/values"][:]),
        "z": np.asarray(g["nodes/props/z/values"][:]),
        "y": np.asarray(g["nodes/props/y/values"][:]),
        "x": np.asarray(g["nodes/props/x/values"][:]),
    })


def matched_gt_mask(gt: pd.DataFrame, pred: pd.DataFrame) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    mask = np.zeros(len(gt), dtype=bool)
    mapping: dict[int, tuple[int, float]] = {}
    gt_t = gt["t"].astype(int).to_numpy()
    pr_t = pred["t"].astype(int).to_numpy()
    for t in sorted(set(gt_t) | set(pr_t)):
        gi = np.flatnonzero(gt_t == t)
        pi = np.flatnonzero(pr_t == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        gxyz = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        pxyz = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        d = np.linalg.norm(gxyz[:, None, :] - pxyz[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        for r, c in zip(rr, cc, strict=True):
            dist = float(d[r, c])
            if dist <= RADIUS:
                gidx = int(gi[r])
                pidx = int(pi[c])
                mask[gidx] = True
                mapping[gidx] = (pidx, dist)
    return mask, mapping


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    gt = gt_df(stem)
    base = base_all[(base_all["dataset"] == stem) & (base_all["row_type"] == "node")][
        ["t", "z", "y", "x"]
    ].reset_index(drop=True)
    base_hit, _ = matched_gt_mask(gt, base)
    missed_by_base = ~base_hit
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    for thr in THRESHOLDS:
        strong = peaks_all[peaks_all["p"] >= thr].reset_index(drop=True)
        strong_hit, strong_map = matched_gt_mask(gt, strong)
        recovered = missed_by_base & strong_hit

        points_um = strong[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        times = strong["t"].astype(int).to_numpy()
        frame_points = {
            int(t): points_um[times == int(t)]
            for t in np.unique(times)
        }
        temporal_counts = {gate: 0 for gate in TEMP_GATES}
        recovered_scores = []
        recovered_dist = []
        for gidx in np.flatnonzero(recovered):
            pidx, dist = strong_map[int(gidx)]
            recovered_scores.append(float(strong.iloc[pidx]["p"]))
            recovered_dist.append(float(dist))
            t = int(strong.iloc[pidx]["t"])
            p = points_um[pidx]
            for gate in TEMP_GATES:
                prev = frame_points.get(t - 1)
                nxt = frame_points.get(t + 1)
                prev_ok = prev is not None and len(prev) and np.linalg.norm(prev - p, axis=1).min() <= gate
                next_ok = nxt is not None and len(nxt) and np.linalg.norm(nxt - p, axis=1).min() <= gate
                if prev_ok and next_ok:
                    temporal_counts[gate] += 1

        row = {
            "dataset": stem,
            "threshold": thr,
            "gt_nodes": len(gt),
            "baseline_miss": int(missed_by_base.sum()),
            "strong_miss": int((~strong_hit).sum()),
            "baseline_miss_recovered": int(recovered.sum()),
            "recovery_rate": float(recovered.sum() / max(missed_by_base.sum(), 1)),
            "recovered_mean_p": float(np.mean(recovered_scores)) if recovered_scores else np.nan,
            "recovered_min_p": float(np.min(recovered_scores)) if recovered_scores else np.nan,
            "recovered_mean_dist_um": float(np.mean(recovered_dist)) if recovered_dist else np.nan,
        }
        for gate in TEMP_GATES:
            row[f"recovered_bidir_{int(gate)}um"] = int(temporal_counts[gate])
        rows.append(row)

result = pd.DataFrame(rows)
print(result.to_string(index=False))
print("TOTAL_BY_THRESHOLD")
agg = result.groupby("threshold", as_index=False).agg(
    baseline_miss=("baseline_miss", "sum"),
    strong_miss=("strong_miss", "sum"),
    recovered=("baseline_miss_recovered", "sum"),
    recovered_bidir_4um=("recovered_bidir_4um", "sum"),
    recovered_bidir_6um=("recovered_bidir_6um", "sum"),
    recovered_bidir_8um=("recovered_bidir_8um", "sum"),
)
print(agg.to_string(index=False))
result.to_csv(PROBE / "gt_rescue_hits.csv", index=False)
