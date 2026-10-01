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
GT_RADIUS = 7.0
SUPPORT_RADIUS = 3.0
MIN_ALT_DISP = (5.0, 7.0, 9.0)
MAX_ALT_DISP = (12.0, 15.0, 18.0, 22.0)
ALT_P_MIN = (0.5, 0.6, 0.7, 0.8, 0.9)
P_MARGIN = (0.0, 0.1, 0.2, 0.3, 0.4)
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def load_gt(stem: str) -> pd.DataFrame:
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame(
        {k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")}
    )


def hits(pred: pd.DataFrame, gt: pd.DataFrame) -> int:
    out = 0
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
        out += int((d[rr, cc] <= GT_RADIUS).sum())
    return out


base_all = pd.read_csv(BASELINE)
rows = []

for stem in STEMS:
    gt = load_gt(stem)
    base = base_all[(base_all.dataset == stem) & (base_all.row_type == "node")][
        ["t", "z", "y", "x"]
    ].copy().reset_index(drop=True)
    peaks = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    peaks = peaks[peaks.p >= 0.2].copy().reset_index(drop=True)
    base_hit = hits(base, gt)

    candidate_cache = {}
    for t, idx in base.groupby(base.t.astype(int)).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        b = base.loc[ids, ["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        frame = peaks[peaks.t.astype(int) == int(t)]
        if frame.empty:
            continue
        pxyz = frame[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        pp = frame.p.to_numpy(np.float32)
        d = np.linalg.norm(b[:, None, :] - pxyz[None, :, :], axis=2)
        support_mask = d <= SUPPORT_RADIUS
        support_p = np.where(support_mask, pp[None, :], -np.inf).max(axis=1)
        support_p[~np.isfinite(support_p)] = 0.0
        candidate_cache[int(t)] = (ids, b, pxyz, pp, d, support_p)

    for min_disp in MIN_ALT_DISP:
        for max_disp in MAX_ALT_DISP:
            if max_disp <= min_disp:
                continue
            for alt_p_min in ALT_P_MIN:
                for margin in P_MARGIN:
                    refined = base.copy()
                    changed = 0
                    for t, (ids, b, pxyz, pp, d, support_p) in candidate_cache.items():
                        alt_mask = (d >= min_disp) & (d <= max_disp) & (pp[None, :] >= alt_p_min)
                        score = np.where(alt_mask, pp[None, :], -np.inf)
                        best_j = score.argmax(axis=1)
                        best_p = score[np.arange(len(ids)), best_j]
                        ok = np.isfinite(best_p) & ((best_p - support_p) >= margin)
                        if not np.any(ok):
                            continue
                        chosen = pxyz[best_j[ok]] / SCALE[None, :]
                        refined.loc[ids[ok], ["z", "y", "x"]] = chosen
                        changed += int(ok.sum())
                    hit = hits(refined, gt)
                    rows.append(
                        {
                            "dataset": stem,
                            "min_alt_disp_um": min_disp,
                            "max_alt_disp_um": max_disp,
                            "alt_p_min": alt_p_min,
                            "p_margin": margin,
                            "changed_nodes": changed,
                            "base_hit": base_hit,
                            "hit": hit,
                            "delta_hit": hit - base_hit,
                        }
                    )

df = pd.DataFrame(rows)
df.to_csv(PROBE / "alternative_peak_snap_sweep.csv", index=False)
agg = (
    df.groupby(["min_alt_disp_um", "max_alt_disp_um", "alt_p_min", "p_margin"], as_index=False)
    .agg(
        changed_nodes=("changed_nodes", "sum"),
        base_hit=("base_hit", "sum"),
        hit=("hit", "sum"),
        delta_hit=("delta_hit", "sum"),
    )
)
agg["recall"] = agg.hit / 2193.0
agg.to_csv(PROBE / "alternative_peak_snap_aggregate.csv", index=False)
print("DELTA RANGE", int(agg.delta_hit.min()), int(agg.delta_hit.max()))
print(
    agg.sort_values(["delta_hit", "changed_nodes"], ascending=[False, True])
    .head(40)
    .to_string(index=False)
)
