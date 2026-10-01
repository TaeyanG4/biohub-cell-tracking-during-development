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
MIN_DISAGREE = (5.0, 6.0, 7.0, 8.0)
MAX_DISAGREE = (10.0, 12.0, 15.0, 18.0)
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def load_gt(stem: str) -> pd.DataFrame:
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame({k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")})


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> int:
    hits = 0
    gt_t = gt.t.astype(int).to_numpy()
    pr_t = pred.t.astype(int).to_numpy()
    for t in sorted(set(gt_t) | set(pr_t)):
        gi = np.flatnonzero(gt_t == t)
        pi = np.flatnonzero(pr_t == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        g = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        p = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        d = np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        hits += int((d[rr, cc] <= GT_RADIUS).sum())
    return hits


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    gt = load_gt(stem)
    base = base_all[(base_all.dataset == stem) & (base_all.row_type == "node")][["t", "z", "y", "x"]].reset_index(drop=True)
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    for thr in THRESHOLDS:
        peaks = peaks_all[peaks_all.p >= thr].copy()
        trees = {}
        coords = {}
        for t, frame in peaks.groupby(peaks.t.astype(int)):
            xyz = frame[["z", "y", "x"]].to_numpy(np.float32) * SCALE
            trees[int(t)] = cKDTree(xyz)
            coords[int(t)] = xyz

        for lo in MIN_DISAGREE:
            for hi in MAX_DISAGREE:
                if hi <= lo:
                    continue
                refined = base.copy()
                changed = 0
                for t, idx in refined.groupby(refined.t.astype(int)).groups.items():
                    t = int(t)
                    if t not in trees:
                        continue
                    ids = np.asarray(list(idx), dtype=int)
                    bxyz = refined.loc[ids, ["z", "y", "x"]].to_numpy(np.float32) * SCALE
                    d, nn = trees[t].query(bxyz, k=1)
                    ok = (d >= lo) & (d <= hi)
                    if not np.any(ok):
                        continue
                    xyz = coords[t][nn[ok]] / SCALE
                    refined.loc[ids[ok], ["z", "y", "x"]] = xyz
                    changed += int(ok.sum())
                hit = evaluate(refined, gt)
                rows.append({
                    "dataset": stem,
                    "threshold": thr,
                    "min_disagree_um": lo,
                    "max_disagree_um": hi,
                    "changed_nodes": changed,
                    "hit": hit,
                    "miss": int(len(gt) - hit),
                })

df = pd.DataFrame(rows)
df.to_csv(PROBE / "selective_snap_sweep.csv", index=False)
agg = df.groupby(["threshold", "min_disagree_um", "max_disagree_um"], as_index=False).agg(
    changed_nodes=("changed_nodes", "sum"),
    hit=("hit", "sum"),
    miss=("miss", "sum"),
)
agg["recall"] = agg.hit / (agg.hit + agg.miss)
print(agg.sort_values(["recall", "changed_nodes"], ascending=[False, True]).head(30).to_string(index=False))
