from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "data" / "visible_gt" / "train"
PROBE = ROOT / "experiments" / "strongunet_probe"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
GT_RADIUS = 7.0
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
SNAP_GATES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def load_gt(stem: str) -> pd.DataFrame:
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame({k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")})


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> tuple[int, float]:
    hits = 0
    dists = []
    gt_t = gt.t.astype(int).to_numpy()
    pr_t = pred.t.astype(int).to_numpy()
    for t in sorted(set(gt_t) | set(pr_t)):
        gi = np.flatnonzero(gt_t == t)
        pi = np.flatnonzero(pr_t == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        g = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        p = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        d = np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        ok = d[rr, cc] <= GT_RADIUS
        hits += int(ok.sum())
        dists.extend(d[rr[ok], cc[ok]].tolist())
    return hits, float(np.mean(dists)) if dists else np.nan


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    gt = load_gt(stem)
    base = base_all[(base_all.dataset == stem) & (base_all.row_type == "node")][["t", "z", "y", "x"]].reset_index(drop=True)
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    base_hit, base_mean = evaluate(base, gt)
    rows.append({
        "dataset": stem,
        "threshold": np.nan,
        "snap_gate_um": 0.0,
        "snapped_nodes": 0,
        "hit": base_hit,
        "miss": int(len(gt) - base_hit),
        "recall": base_hit / len(gt),
        "mean_match_um": base_mean,
    })

    for thr in THRESHOLDS:
        peaks = peaks_all[peaks_all.p >= thr].copy()
        trees = {}
        peak_xyz = {}
        for t, frame in peaks.groupby(peaks.t.astype(int)):
            xyz = frame[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
            trees[int(t)] = cKDTree(xyz)
            peak_xyz[int(t)] = xyz

        for gate in SNAP_GATES:
            refined = base.copy()
            snapped = 0
            for t, idx in refined.groupby(refined.t.astype(int)).groups.items():
                t = int(t)
                if t not in trees:
                    continue
                ids = np.asarray(list(idx), dtype=int)
                bxyz = refined.loc[ids, ["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
                d, nn = trees[t].query(bxyz, k=1)
                ok = d <= gate
                if not np.any(ok):
                    continue
                xyz = peak_xyz[t][nn[ok]] / SCALE[None, :]
                refined.loc[ids[ok], ["z", "y", "x"]] = xyz
                snapped += int(ok.sum())
            hit, mean = evaluate(refined, gt)
            rows.append({
                "dataset": stem,
                "threshold": thr,
                "snap_gate_um": gate,
                "snapped_nodes": snapped,
                "hit": hit,
                "miss": int(len(gt) - hit),
                "recall": hit / len(gt),
                "mean_match_um": mean,
            })

df = pd.DataFrame(rows)
df.to_csv(PROBE / "snap_sweep.csv", index=False)
agg = df.groupby(["threshold", "snap_gate_um"], dropna=False, as_index=False).agg(
    snapped_nodes=("snapped_nodes", "sum"),
    hit=("hit", "sum"),
    miss=("miss", "sum"),
)
agg["recall"] = agg.hit / (agg.hit + agg.miss)
print(agg.sort_values(["recall", "snapped_nodes"], ascending=[False, True]).to_string(index=False))
