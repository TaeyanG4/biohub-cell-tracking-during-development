"""ORACLE-ONLY diagnostic; NOT a deployable feature ranking.

Both candidate locations are chosen by querying a tree at the GT coordinate
(see the gt.iterrows loop). Consequently baseline_strong_dist_um and its AUC
cannot justify an inference-time correction rule. Never use this table to
select test predictions or to claim held-out performance. Build candidate
pairs without GT first, and use GT only to evaluate those fixed candidates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "data" / "visible_gt" / "train"
IMG_ROOT = ROOT / "data" / "visible_test" / "test"
PROBE = ROOT / "experiments" / "strongunet_probe"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
MOTION_DIAG = PROBE / "motion_conditioned_node_diagnostics.parquet"

SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
GT_RADIUS = 7.0
STRONG_THRESHOLD = 0.2
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


def matched_gt_mask(gt: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(gt), dtype=bool)
    gt_t = gt.t.astype(int).to_numpy()
    pr_t = pred.t.astype(int).to_numpy()
    for t in sorted(set(gt_t) | set(pr_t)):
        gi = np.flatnonzero(gt_t == t)
        pi = np.flatnonzero(pr_t == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        gxyz = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        pxyz = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        d = np.linalg.norm(gxyz[:, None, :] - pxyz[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        mask[gi[rr[d[rr, cc] <= GT_RADIUS]]] = True
    return mask


def image_stats(stem: str):
    sample = IMG_ROOT / f"{stem}.zarr"
    arr = zarr.open_group(str(sample), mode="r")["0"]
    meta = json.loads((sample / "zarr.json").read_text(encoding="utf-8"))
    q = meta["attributes"]["image_statistics"]["quantiles"]
    lo, hi = float(q["0.001"]), float(q["0.999"])
    return arr, lo, hi


def sample_intensity(arr, t: int, zyx_vox: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    z, y, x = np.rint(zyx_vox).astype(int)
    z = int(np.clip(z, 0, arr.shape[1] - 1))
    y = int(np.clip(y, 0, arr.shape[2] - 1))
    x = int(np.clip(x, 0, arr.shape[3] - 1))
    vol = arr[t]
    center = float(vol[z, y, x])
    z0, z1 = max(0, z - 1), min(arr.shape[1], z + 2)
    y0, y1 = max(0, y - 4), min(arr.shape[2], y + 5)
    x0, x1 = max(0, x - 4), min(arr.shape[3], x + 5)
    patch = np.asarray(vol[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    local = float(np.percentile(patch, 90)) if patch.size else center
    denom = max(hi - lo, 1e-6)
    return (center - lo) / denom, (local - lo) / denom


base_all = pd.read_csv(BASELINE)
motion_all = pd.read_parquet(MOTION_DIAG)
motion_all = motion_all[np.isclose(motion_all.p_threshold, STRONG_THRESHOLD)].copy()
rows = []

for stem in STEMS:
    gt = load_gt(stem)
    group = base_all[base_all.dataset.eq(stem)]
    base = group[group.row_type.eq("node")][["node_id", "t", "z", "y", "x"]].copy()
    base["node_id"] = base.node_id.astype(int)
    base["t"] = base.t.astype(int)
    base = base.reset_index(drop=True)
    strong = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    strong = strong[strong.p >= STRONG_THRESHOLD].reset_index(drop=True)
    gt_hit = matched_gt_mask(gt, base[["t", "z", "y", "x"]])
    arr, qlo, qhi = image_stats(stem)
    md = motion_all[motion_all.dataset.eq(stem)].set_index("node_id")

    base_by_t = {}
    strong_by_t = {}
    for t, f in base.groupby("t"):
        xyz = f[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        base_by_t[int(t)] = (f.index.to_numpy(int), cKDTree(xyz), xyz)
    for t, f in strong.groupby(strong.t.astype(int)):
        xyz = f[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        strong_by_t[int(t)] = (f.index.to_numpy(int), cKDTree(xyz), xyz)

    for gi, g in gt.iterrows():
        t = int(g.t)
        gu = np.asarray([g.z, g.y, g.x], dtype=np.float32) * SCALE
        bi_idx, btree, bxyz = base_by_t[t]
        bdist, bj = btree.query(gu, k=1)
        b_row_idx = int(bi_idx[int(bj)])
        b = base.iloc[b_row_idx]
        b_vox = np.asarray([b.z, b.y, b.x], dtype=np.float32)
        b_um = b_vox * SCALE

        if t not in strong_by_t:
            continue
        si_idx, stree, sxyz = strong_by_t[t]
        sdist, sj = stree.query(gu, k=1)
        s_row_idx = int(si_idx[int(sj)])
        s = strong.iloc[s_row_idx]
        s_vox = np.asarray([s.z, s.y, s.x], dtype=np.float32)
        s_um = s_vox * SCALE

        b_center, b_local = sample_intensity(arr, t, b_vox, qlo, qhi)
        s_center, s_local = sample_intensity(arr, t, s_vox, qlo, qhi)
        bs_dist = float(np.linalg.norm(b_um - s_um))

        node_id = int(b.node_id)
        if node_id in md.index:
            m = md.loc[node_id]
            if isinstance(m, pd.DataFrame):
                m = m.iloc[0]
            base_resid = float(m.base_resid_um)
            strong_motion_resid = float(m.best_peak_resid_um)
            motion_improve = float(m.improve_um)
        else:
            base_resid = strong_motion_resid = motion_improve = np.nan

        rows.append(
            {
                "dataset": stem,
                "gt_index": int(gi),
                "t": t,
                "baseline_hit": bool(gt_hit[gi]),
                "label_miss": int(not gt_hit[gi]),
                "baseline_gt_dist_um": float(bdist),
                "strong_gt_dist_um": float(sdist),
                "baseline_strong_dist_um": bs_dist,
                "strong_p": float(s.p),
                "baseline_center_intensity": b_center,
                "strong_center_intensity": s_center,
                "center_intensity_gain": s_center - b_center,
                "baseline_local_p90": b_local,
                "strong_local_p90": s_local,
                "local_p90_gain": s_local - b_local,
                "baseline_motion_resid_um": base_resid,
                "strong_motion_resid_um": strong_motion_resid,
                "motion_improve_um": motion_improve,
                "strong_would_hit": int(sdist <= GT_RADIUS),
            }
        )

df = pd.DataFrame(rows)
df.to_parquet(PROBE / "miss_feature_table.parquet", index=False)

features = [
    "baseline_strong_dist_um",
    "strong_p",
    "baseline_center_intensity",
    "strong_center_intensity",
    "center_intensity_gain",
    "baseline_local_p90",
    "strong_local_p90",
    "local_p90_gain",
    "baseline_motion_resid_um",
    "strong_motion_resid_um",
    "motion_improve_um",
]
summary = []
y = df.label_miss.to_numpy(int)
for feature in features:
    x = df[feature].to_numpy(float)
    finite = np.isfinite(x)
    if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
        continue
    # Report both orientations and keep the better one. This is diagnostic only.
    auc = roc_auc_score(y[finite], x[finite])
    ap = average_precision_score(y[finite], x[finite])
    auc_inv = roc_auc_score(y[finite], -x[finite])
    ap_inv = average_precision_score(y[finite], -x[finite])
    if auc_inv > auc:
        direction = "low_is_miss"
        auc, ap = auc_inv, ap_inv
        score = -x
    else:
        direction = "high_is_miss"
        score = x
    finite_score = score.copy()
    finite_score[~finite] = -np.inf
    order = np.argsort(-finite_score)
    top20 = int(y[order[:20]].sum())
    top50 = int(y[order[:50]].sum())
    top100 = int(y[order[:100]].sum())
    summary.append(
        {
            "feature": feature,
            "direction": direction,
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "misses_in_top20": top20,
            "misses_in_top50": top50,
            "misses_in_top100": top100,
        }
    )

summary_df = pd.DataFrame(summary).sort_values(
    ["misses_in_top20", "roc_auc"], ascending=[False, False]
)
summary_df.to_csv(PROBE / "miss_feature_ranking.csv", index=False)
print("MISS COUNT", int(y.sum()), "TOTAL", len(y))
print(summary_df.to_string(index=False))
print("\nMISSES")
print(df[df.label_miss.eq(1)].to_string(index=False))
