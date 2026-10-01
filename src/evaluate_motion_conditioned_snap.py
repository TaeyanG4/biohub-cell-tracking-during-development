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
MIN_BASE_RESID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
MIN_IMPROVE = (1.0, 2.0, 3.0, 4.0)
MAX_DISPLACE = (8.0, 10.0, 12.0, 15.0, 18.0)

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


def gt_hits(pred: pd.DataFrame, gt: pd.DataFrame) -> int:
    hits = 0
    gt_t = gt.t.astype(int).to_numpy()
    pr_t = pred.t.astype(int).to_numpy()
    for t in sorted(set(gt_t) | set(pr_t)):
        gi = np.flatnonzero(gt_t == t)
        pi = np.flatnonzero(pr_t == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        gxyz = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        pxyz = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE
        d = np.linalg.norm(gxyz[:, None, :] - pxyz[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        hits += int((d[rr, cc] <= GT_RADIUS).sum())
    return hits


def graph_arrays(group: pd.DataFrame):
    nodes = group[group.row_type.eq("node")][["node_id", "t", "z", "y", "x"]].copy()
    nodes["node_id"] = nodes.node_id.astype(int)
    nodes["t"] = nodes.t.astype(int)
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    ids = nodes.node_id.to_numpy(int)
    id_to_idx = {int(node_id): i for i, node_id in enumerate(ids)}

    coords_um = nodes[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
    times = nodes.t.to_numpy(int)
    pred = {i: [] for i in range(len(nodes))}
    succ = {i: [] for i in range(len(nodes))}
    edges = group[group.row_type.eq("edge")][["source_id", "target_id"]]
    for row in edges.itertuples(index=False):
        source_id, target_id = int(row.source_id), int(row.target_id)
        if source_id not in id_to_idx or target_id not in id_to_idx:
            continue
        s, t = id_to_idx[source_id], id_to_idx[target_id]
        succ[s].append(t)
        pred[t].append(s)
    return nodes, coords_um, times, pred, succ


def motion_prediction(i: int, coords: np.ndarray, pred: dict[int, list[int]], succ: dict[int, list[int]]):
    """Return linear expected position and context type for a non-branch node."""
    p = pred[i]
    s = succ[i]
    # Strongest case: one predecessor and one successor => midpoint.
    if len(p) == 1 and len(s) == 1:
        return 0.5 * (coords[p[0]] + coords[s[0]]), "interior"

    # Start of a track: extrapolate backward from two unique successors.
    if len(p) == 0 and len(s) == 1:
        s1 = s[0]
        if len(succ[s1]) == 1:
            s2 = succ[s1][0]
            return 2.0 * coords[s1] - coords[s2], "start_extrap"

    # End of a track: extrapolate forward from two unique predecessors.
    if len(s) == 0 and len(p) == 1:
        p1 = p[0]
        if len(pred[p1]) == 1:
            p2 = pred[p1][0]
            return 2.0 * coords[p1] - coords[p2], "end_extrap"

    return None, None


base_all = pd.read_csv(BASELINE)
result_rows = []
detail_rows = []

for stem in STEMS:
    group = base_all[base_all.dataset.eq(stem)]
    nodes, coords_um, times, pred, succ = graph_arrays(group)
    gt = load_gt(stem)
    base_hit = gt_hits(nodes[["t", "z", "y", "x"]], gt)
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")

    # Precompute graph-based expectations for every node once.
    expected = np.full_like(coords_um, np.nan)
    context = np.empty(len(nodes), dtype=object)
    base_resid = np.full(len(nodes), np.nan, dtype=np.float32)
    for i in range(len(nodes)):
        q, c = motion_prediction(i, coords_um, pred, succ)
        if q is None:
            continue
        expected[i] = q
        context[i] = c
        base_resid[i] = float(np.linalg.norm(coords_um[i] - q))

    for p_thr in P_THRESHOLDS:
        peaks = peaks_all[peaks_all.p >= p_thr].copy().reset_index(drop=True)
        peak_um = peaks[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        peak_t = peaks.t.astype(int).to_numpy()

        # For every node, select the StrongUNet candidate at the same frame that
        # best agrees with the graph-motion expectation, subject only to a loose
        # displacement cap applied later in the sweep.
        best_peak_idx = np.full(len(nodes), -1, dtype=int)
        best_peak_resid = np.full(len(nodes), np.inf, dtype=np.float32)
        best_displace = np.full(len(nodes), np.inf, dtype=np.float32)
        for t in np.unique(times):
            ni = np.flatnonzero(times == int(t))
            pi = np.flatnonzero(peak_t == int(t))
            ni = ni[np.isfinite(base_resid[ni])]
            if not len(ni) or not len(pi):
                continue
            e = expected[ni]
            pxyz = peak_um[pi]
            # Chunk to avoid large temporary matrices in dense frames.
            for j0 in range(0, len(ni), 256):
                sub = ni[j0 : j0 + 256]
                d_exp = np.linalg.norm(expected[sub, None, :] - pxyz[None, :, :], axis=2)
                jj = d_exp.argmin(axis=1)
                cand = pi[jj]
                best_peak_idx[sub] = cand
                best_peak_resid[sub] = d_exp[np.arange(len(sub)), jj]
                best_displace[sub] = np.linalg.norm(coords_um[sub] - peak_um[cand], axis=1)

        for min_resid in MIN_BASE_RESID:
            for min_improve in MIN_IMPROVE:
                improve = base_resid - best_peak_resid
                for max_disp in MAX_DISPLACE:
                    ok = (
                        np.isfinite(base_resid)
                        & (best_peak_idx >= 0)
                        & (base_resid >= min_resid)
                        & (improve >= min_improve)
                        & (best_displace <= max_disp)
                    )
                    refined = nodes[["t", "z", "y", "x"]].copy()
                    ids = np.flatnonzero(ok)
                    if len(ids):
                        cand = best_peak_idx[ids]
                        refined.loc[ids, ["z", "y", "x"]] = peak_um[cand] / SCALE[None, :]
                    hit = gt_hits(refined, gt)
                    result_rows.append(
                        {
                            "dataset": stem,
                            "p_threshold": p_thr,
                            "min_base_resid_um": min_resid,
                            "min_improve_um": min_improve,
                            "max_displace_um": max_disp,
                            "changed_nodes": int(ok.sum()),
                            "base_hit": int(base_hit),
                            "hit": int(hit),
                            "delta_hit": int(hit - base_hit),
                        }
                    )

        # Persist node-level diagnostics once per p threshold for later analysis.
        for i in np.flatnonzero(np.isfinite(base_resid)):
            cand = best_peak_idx[i]
            detail_rows.append(
                {
                    "dataset": stem,
                    "node_id": int(i),
                    "t": int(times[i]),
                    "context": context[i],
                    "p_threshold": p_thr,
                    "base_resid_um": float(base_resid[i]),
                    "best_peak_resid_um": float(best_peak_resid[i]),
                    "improve_um": float(base_resid[i] - best_peak_resid[i]),
                    "displace_um": float(best_displace[i]),
                    "peak_p": float(peaks.iloc[cand].p) if cand >= 0 else np.nan,
                }
            )


df = pd.DataFrame(result_rows)
df.to_csv(PROBE / "motion_conditioned_snap_sweep.csv", index=False)
pd.DataFrame(detail_rows).to_parquet(PROBE / "motion_conditioned_node_diagnostics.parquet", index=False)

agg = (
    df.groupby(
        ["p_threshold", "min_base_resid_um", "min_improve_um", "max_displace_um"],
        as_index=False,
    )
    .agg(
        changed_nodes=("changed_nodes", "sum"),
        base_hit=("base_hit", "sum"),
        hit=("hit", "sum"),
        delta_hit=("delta_hit", "sum"),
    )
)
agg["recall"] = agg.hit / 2193.0
agg.to_csv(PROBE / "motion_conditioned_snap_aggregate.csv", index=False)
print(
    agg.sort_values(["delta_hit", "changed_nodes"], ascending=[False, True])
    .head(40)
    .to_string(index=False)
)
