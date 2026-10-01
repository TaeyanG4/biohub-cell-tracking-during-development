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
P_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
DISAGREE_THRESHOLDS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0)
MAX_PAIR_DISTANCE = 25.0

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


def evaluate_hits(pred: pd.DataFrame, gt: pd.DataFrame) -> int:
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


def match_detectors(base: pd.DataFrame, strong: pd.DataFrame) -> pd.DataFrame:
    """One-to-one framewise detector correspondence without using GT."""
    rows = []
    for t in sorted(set(base.t.astype(int)) | set(strong.t.astype(int))):
        bi = np.flatnonzero(base.t.astype(int).to_numpy() == t)
        si = np.flatnonzero(strong.t.astype(int).to_numpy() == t)
        if not len(bi) or not len(si):
            continue
        bxyz = base.iloc[bi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        sxyz = strong.iloc[si][["z", "y", "x"]].to_numpy(np.float32) * SCALE

        # Baseline is normally smaller than StrongUNet.  Full Hungarian is still
        # tractable per frame, but dense frames can be large.  To avoid a huge
        # dense matrix, only retain strong candidates within a loose 25 um box
        # around any baseline point before assignment.
        keep = np.zeros(len(si), dtype=bool)
        for j0 in range(0, len(bi), 128):
            d = np.linalg.norm(bxyz[j0 : j0 + 128, None, :] - sxyz[None, :, :], axis=2)
            keep |= (d <= MAX_PAIR_DISTANCE).any(axis=0)
        sj = si[keep]
        sxyz2 = strong.iloc[sj][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        if not len(sj):
            continue

        d = np.linalg.norm(bxyz[:, None, :] - sxyz2[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        for r, c in zip(rr, cc, strict=True):
            dist = float(d[r, c])
            if dist > MAX_PAIR_DISTANCE:
                continue
            bidx = int(bi[r])
            sidx = int(sj[c])
            rows.append(
                {
                    "base_idx": bidx,
                    "strong_idx": sidx,
                    "t": int(t),
                    "distance_um": dist,
                    "strong_p": float(strong.iloc[sidx].p),
                }
            )
    return pd.DataFrame(rows)


base_all = pd.read_csv(BASELINE)
rows = []
pair_details = []

for stem in STEMS:
    gt = load_gt(stem)
    base = base_all[(base_all.dataset == stem) & (base_all.row_type == "node")][
        ["t", "z", "y", "x"]
    ].reset_index(drop=True)
    base_hit = evaluate_hits(base, gt)
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")

    for p_thr in P_THRESHOLDS:
        strong = peaks_all[peaks_all.p >= p_thr].reset_index(drop=True)
        pairs = match_detectors(base, strong)
        pair_details.append(pairs.assign(dataset=stem, p_threshold=p_thr))

        for disagree in DISAGREE_THRESHOLDS:
            refined = base.copy()
            chosen = pairs[pairs.distance_um >= disagree]
            if len(chosen):
                bidx = chosen.base_idx.to_numpy(int)
                sidx = chosen.strong_idx.to_numpy(int)
                refined.loc[bidx, ["z", "y", "x"]] = strong.loc[
                    sidx, ["z", "y", "x"]
                ].to_numpy()
            hit = evaluate_hits(refined, gt)
            rows.append(
                {
                    "dataset": stem,
                    "p_threshold": p_thr,
                    "disagree_um": disagree,
                    "paired_nodes": int(len(pairs)),
                    "changed_nodes": int(len(chosen)),
                    "base_hit": int(base_hit),
                    "hit": int(hit),
                    "delta_hit": int(hit - base_hit),
                }
            )

df = pd.DataFrame(rows)
df.to_csv(PROBE / "hungarian_detector_snap_sweep.csv", index=False)
pd.concat(pair_details, ignore_index=True).to_parquet(
    PROBE / "hungarian_detector_pairs.parquet", index=False
)

agg = (
    df.groupby(["p_threshold", "disagree_um"], as_index=False)
    .agg(
        paired_nodes=("paired_nodes", "sum"),
        changed_nodes=("changed_nodes", "sum"),
        base_hit=("base_hit", "sum"),
        hit=("hit", "sum"),
        delta_hit=("delta_hit", "sum"),
    )
)
agg["recall"] = agg.hit / 2193.0
agg.to_csv(PROBE / "hungarian_detector_snap_aggregate.csv", index=False)
print(agg.sort_values(["delta_hit", "changed_nodes"], ascending=[False, True]).to_string(index=False))
