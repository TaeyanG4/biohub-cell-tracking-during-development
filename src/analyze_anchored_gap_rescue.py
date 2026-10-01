from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "data" / "visible_gt" / "train"
PROBE = ROOT / "experiments" / "strongunet_probe"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
ANCHOR_GATES = (6.0, 8.0, 10.0)
MID_GATES = (2.5, 3.5, 5.0)
BASE_EXCLUDE_UM = 7.0
GT_MATCH_UM = 7.0
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def load_gt(stem: str) -> pd.DataFrame:
    g = zarr.open_group(str(GT_ROOT / f"{stem}.geff"), mode="r")
    return pd.DataFrame({
        "t": np.asarray(g["nodes/props/t/values"][:]),
        "z": np.asarray(g["nodes/props/z/values"][:]),
        "y": np.asarray(g["nodes/props/y/values"][:]),
        "x": np.asarray(g["nodes/props/x/values"][:]),
    })


def baseline_graph(group: pd.DataFrame):
    nodes = group[group["row_type"].eq("node")].copy()
    edges = group[group["row_type"].eq("edge")].copy()
    node_by_id = {
        int(r.node_id): (int(r.t), float(r.z), float(r.y), float(r.x))
        for r in nodes.itertuples(index=False)
    }
    outdeg = {nid: 0 for nid in node_by_id}
    indeg = {nid: 0 for nid in node_by_id}
    for r in edges.itertuples(index=False):
        s, t = int(r.source_id), int(r.target_id)
        outdeg[s] = outdeg.get(s, 0) + 1
        indeg[t] = indeg.get(t, 0) + 1
    tails = {}
    heads = {}
    for nid, (t, z, y, x) in node_by_id.items():
        if outdeg.get(nid, 0) == 0:
            tails.setdefault(t, []).append((nid, z, y, x))
        if indeg.get(nid, 0) == 0:
            heads.setdefault(t, []).append((nid, z, y, x))
    return nodes, tails, heads


def gt_hit_mask(gt: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    hit = np.zeros(len(gt), dtype=bool)
    for t in sorted(gt["t"].astype(int).unique()):
        gi = np.flatnonzero(gt["t"].astype(int).to_numpy() == t)
        pi = np.flatnonzero(pred["t"].astype(int).to_numpy() == t)
        if len(gi) == 0 or len(pi) == 0:
            continue
        g = gt.iloc[gi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        p = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        tree = cKDTree(p)
        d, _ = tree.query(g, k=1)
        hit[gi] = d <= GT_MATCH_UM
    return hit


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    group = base_all[base_all["dataset"].eq(stem)]
    base_nodes, tails, heads = baseline_graph(group)
    gt = load_gt(stem)
    base_hit = gt_hit_mask(gt, base_nodes[["t", "z", "y", "x"]])
    baseline_miss = ~base_hit
    peaks_all = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")

    base_by_t = {
        int(t): df[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        for t, df in base_nodes.groupby(base_nodes["t"].astype(int))
    }

    for thr in THRESHOLDS:
        peaks = peaks_all[peaks_all["p"] >= thr].copy()
        candidate_rows = []
        for t, frame in peaks.groupby(peaks["t"].astype(int)):
            pts = frame[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
            base_pts = base_by_t.get(int(t))
            if base_pts is None or len(base_pts) == 0:
                far = np.ones(len(frame), dtype=bool)
            else:
                d, _ = cKDTree(base_pts).query(pts, k=1)
                far = d > BASE_EXCLUDE_UM
            if far.any():
                tmp = frame.iloc[np.flatnonzero(far)].copy()
                tmp["_t"] = int(t)
                candidate_rows.append(tmp)
        extras = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else peaks.iloc[:0].copy()

        for anchor_gate in ANCHOR_GATES:
            for mid_gate in MID_GATES:
                accepted = []
                for r in extras.itertuples(index=False):
                    t = int(r.t)
                    prev = tails.get(t - 1, [])
                    nxt = heads.get(t + 1, [])
                    if not prev or not nxt:
                        continue
                    p = np.asarray((r.z, r.y, r.x), dtype=np.float32) * SCALE
                    prev_xyz = np.asarray([x[1:] for x in prev], dtype=np.float32) * SCALE[None, :]
                    next_xyz = np.asarray([x[1:] for x in nxt], dtype=np.float32) * SCALE[None, :]
                    dp = np.linalg.norm(prev_xyz - p[None, :], axis=1)
                    dn = np.linalg.norm(next_xyz - p[None, :], axis=1)
                    ip = np.flatnonzero(dp <= anchor_gate)
                    inn = np.flatnonzero(dn <= anchor_gate)
                    if len(ip) == 0 or len(inn) == 0:
                        continue
                    ok = False
                    for a in ip:
                        for b in inn:
                            midpoint = 0.5 * (prev_xyz[a] + next_xyz[b])
                            if float(np.linalg.norm(p - midpoint)) <= mid_gate:
                                ok = True
                                break
                        if ok:
                            break
                    if ok:
                        accepted.append((t, float(r.z), float(r.y), float(r.x), float(r.p)))

                accepted_df = pd.DataFrame(accepted, columns=["t", "z", "y", "x", "p"])
                rescue_hit = gt_hit_mask(gt, accepted_df[["t", "z", "y", "x"]]) if len(accepted_df) else np.zeros(len(gt), bool)
                recovered = baseline_miss & rescue_hit
                rows.append({
                    "dataset": stem,
                    "threshold": thr,
                    "anchor_gate_um": anchor_gate,
                    "mid_gate_um": mid_gate,
                    "baseline_miss": int(baseline_miss.sum()),
                    "strong_only": int(len(extras)),
                    "accepted_candidates": int(len(accepted_df)),
                    "baseline_miss_recovered": int(recovered.sum()),
                    "recovery_rate": float(recovered.sum() / max(int(baseline_miss.sum()), 1)),
                    "mean_p": float(accepted_df["p"].mean()) if len(accepted_df) else np.nan,
                })

result = pd.DataFrame(rows)
result.to_csv(PROBE / "anchored_gap_rescue.csv", index=False)
agg = result.groupby(["threshold", "anchor_gate_um", "mid_gate_um"], as_index=False).agg(
    baseline_miss=("baseline_miss", "sum"),
    strong_only=("strong_only", "sum"),
    accepted_candidates=("accepted_candidates", "sum"),
    recovered=("baseline_miss_recovered", "sum"),
)
agg["recovery_per_candidate"] = agg["recovered"] / agg["accepted_candidates"].clip(lower=1)
print(agg.sort_values(["recovered", "accepted_candidates"], ascending=[False, True]).to_string(index=False))
