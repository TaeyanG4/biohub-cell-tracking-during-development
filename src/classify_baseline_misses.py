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
RADIUS = 7.0
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def load_gt(stem):
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame({k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")})


def assignment_misses(gt, pred):
    miss = []
    for t in sorted(gt["t"].astype(int).unique()):
        gi = np.flatnonzero(gt["t"].astype(int).to_numpy() == t)
        pi = np.flatnonzero(pred["t"].astype(int).to_numpy() == t)
        if len(pi) == 0:
            miss.extend(gi.tolist())
            continue
        g = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        p = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        d = np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        hit = np.zeros(len(gi), bool)
        hit[rr[d[rr, cc] <= RADIUS]] = True
        miss.extend(gi[~hit].tolist())
    return miss


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    gt = load_gt(stem)
    base = base_all[(base_all.dataset == stem) & (base_all.row_type == "node")][["t", "z", "y", "x"]].reset_index(drop=True)
    strong = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    strong = strong[strong.p >= 0.2].reset_index(drop=True)
    for gi in assignment_misses(gt, base):
        g = gt.iloc[gi]
        t = int(g.t)
        gu = np.asarray([g.z, g.y, g.x], np.float32) * SCALE
        b = base[base.t.astype(int) == t]
        s = strong[strong.t.astype(int) == t]
        bxyz = b[["z", "y", "x"]].to_numpy(np.float32) * SCALE
        sxyz = s[["z", "y", "x"]].to_numpy(np.float32) * SCALE
        bd, bi = cKDTree(bxyz).query(gu, k=1) if len(bxyz) else (np.inf, -1)
        sd, si = cKDTree(sxyz).query(gu, k=1) if len(sxyz) else (np.inf, -1)
        sp = float(s.iloc[int(si)].p) if si >= 0 else np.nan
        rows.append({"dataset": stem, "t": t, "baseline_nearest_um": float(bd), "strong_nearest_um": float(sd), "strong_p": sp})

df = pd.DataFrame(rows)
df.to_csv(PROBE / "baseline_miss_classification.csv", index=False)
print(df.to_string(index=False))
print("bins", pd.cut(df.baseline_nearest_um, [0,7,10,15,25,np.inf], right=False).value_counts().sort_index().to_dict())
