from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "experiments" / "strongunet_probe"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
OUT = PROBE / "rescue_analysis.csv"

STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
BASE_EXCLUDE_UM = 7.0
TEMP_GATES = (4.0, 6.0, 8.0)


def frame_points(df: pd.DataFrame, t: int) -> np.ndarray:
    x = df[df["t"].astype(int).eq(t)][["z", "y", "x"]].to_numpy(np.float32)
    return x * SCALE[None, :]


def strong_only(pred: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(pred), dtype=bool)
    for t in sorted(pred["t"].astype(int).unique()):
        pi = np.flatnonzero(pred["t"].astype(int).to_numpy() == t)
        p = pred.iloc[pi][["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
        b = frame_points(base, t)
        if len(b) == 0 or len(p) == 0:
            continue
        dist, _ = cKDTree(b).query(p, k=1)
        keep[pi] = dist > BASE_EXCLUDE_UM
    return pred.loc[keep].copy()


def temporal_mask(df: pd.DataFrame, gate_um: float) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=bool)
    tarr = df["t"].astype(int).to_numpy()
    pts = df[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
    trees = {}
    indices = {}
    for t in np.unique(tarr):
        idx = np.flatnonzero(tarr == t)
        indices[int(t)] = idx
        trees[int(t)] = cKDTree(pts[idx])
    out = np.zeros(len(df), dtype=bool)
    for i, (t, p) in enumerate(zip(tarr, pts, strict=True)):
        prev_ok = False
        next_ok = False
        if int(t) - 1 in trees:
            d, _ = trees[int(t) - 1].query(p, k=1)
            prev_ok = d <= gate_um
        if int(t) + 1 in trees:
            d, _ = trees[int(t) + 1].query(p, k=1)
            next_ok = d <= gate_um
        out[i] = prev_ok and next_ok
    return out


base_all = pd.read_csv(BASELINE)
rows = []
for stem in STEMS:
    peaks = pd.read_parquet(PROBE / f"{stem}_peaks.parquet")
    base = base_all[(base_all["dataset"] == stem) & (base_all["row_type"] == "node")][
        ["t", "z", "y", "x"]
    ].copy()
    for thr in THRESHOLDS:
        pred = peaks[peaks["p"] >= thr].copy()
        extra = strong_only(pred, base)
        for gate in TEMP_GATES:
            mask = temporal_mask(extra, gate)
            kept = extra.loc[mask]
            rows.append(
                {
                    "dataset": stem,
                    "threshold": thr,
                    "temporal_gate_um": gate,
                    "strong_nodes": int(len(pred)),
                    "baseline_nodes": int(len(base)),
                    "strong_only_gt7um": int(len(extra)),
                    "persistent_bidir": int(mask.sum()),
                    "persistent_fraction": float(mask.mean()) if len(mask) else 0.0,
                    "mean_p_persistent": float(kept["p"].mean()) if len(kept) else float("nan"),
                }
            )

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
print(result.to_string(index=False))
print("AGG")
agg = result.groupby(["threshold", "temporal_gate_um"], as_index=False).agg(
    strong_nodes=("strong_nodes", "sum"),
    baseline_nodes=("baseline_nodes", "sum"),
    strong_only_gt7um=("strong_only_gt7um", "sum"),
    persistent_bidir=("persistent_bidir", "sum"),
)
agg["persistent_fraction"] = agg["persistent_bidir"] / agg["strong_only_gt7um"].clip(lower=1)
print(agg.to_string(index=False))
