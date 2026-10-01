from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIMPLE_FEATURES = [
    "distance_margin",
    "velocity_gain",
    "cosine_gain",
    "both_prev",
]

EXTENDED_FEATURES = [
    "distance_top1",
    "distance_top2",
    "distance_margin",
    "distance_ratio",
    "velocity_residual_top1",
    "velocity_residual_top2",
    "velocity_gain",
    "velocity_cosine_top1",
    "velocity_cosine_top2",
    "cosine_gain",
    "has_prev_top1",
    "has_prev_top2",
    "both_prev",
    "prev_speed_top1",
    "prev_speed_top2",
    "prev_speed_diff",
    "age_top1",
    "age_top2",
    "age_diff",
    "source_nn_top1",
    "source_nn_top2",
    "source_nn_diff",
    "target_nn",
    "candidate_count",
    "count_r4",
    "count_r6",
    "count_r8",
    "count_r10",
    "count_r14",
    "abs_dz_top1",
    "abs_dz_top2",
    "abs_dy_top1",
    "abs_dy_top2",
    "abs_dx_top1",
    "abs_dx_top2",
    "residual_dz_top1",
    "residual_dz_top2",
    "residual_dy_top1",
    "residual_dy_top2",
    "residual_dx_top1",
    "residual_dx_top2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    rows: list[dict] = []
    for (dataset, target_id), q in df.groupby(["dataset", "target_id"], sort=False):
        q = q.sort_values("distance", kind="stable").reset_index(drop=True)
        if len(q) < 2:
            continue
        a = q.iloc[0]
        b = q.iloc[1]
        row = {
            "dataset": str(dataset),
            "group": str(a.group),
            "target_id": int(target_id),
            "top1_source_id": int(a.source_id),
            "top2_source_id": int(b.source_id),
            "base_correct": int(a.label > 0),
            "top2_correct": int(b.label > 0),
            "swap_label": int((a.label == 0) and (b.label > 0)),
            "distance_top1": float(a.distance),
            "distance_top2": float(b.distance),
            "distance_margin": float(b.distance - a.distance),
            "distance_ratio": float(b.distance / max(float(a.distance), 0.25)),
            "velocity_residual_top1": float(a.velocity_residual),
            "velocity_residual_top2": float(b.velocity_residual),
            "velocity_gain": float(a.velocity_residual - b.velocity_residual),
            "velocity_cosine_top1": float(a.velocity_cosine),
            "velocity_cosine_top2": float(b.velocity_cosine),
            "cosine_gain": float(b.velocity_cosine - a.velocity_cosine),
            "has_prev_top1": int(a.has_prev),
            "has_prev_top2": int(b.has_prev),
            "both_prev": int(bool(a.has_prev) and bool(b.has_prev)),
            "prev_speed_top1": float(a.prev_speed),
            "prev_speed_top2": float(b.prev_speed),
            "prev_speed_diff": float(b.prev_speed - a.prev_speed),
            "age_top1": float(a.source_track_age),
            "age_top2": float(b.source_track_age),
            "age_diff": float(b.source_track_age - a.source_track_age),
            "source_nn_top1": float(a.source_nn),
            "source_nn_top2": float(b.source_nn),
            "source_nn_diff": float(b.source_nn - a.source_nn),
            "target_nn": float(a.target_nn),
            "candidate_count": float(a.candidate_count),
            "count_r4": float(a.count_r4),
            "count_r6": float(a.count_r6),
            "count_r8": float(a.count_r8),
            "count_r10": float(a.count_r10),
            "count_r14": float(a.count_r14),
            "abs_dz_top1": float(a.abs_dz),
            "abs_dz_top2": float(b.abs_dz),
            "abs_dy_top1": float(a.abs_dy),
            "abs_dy_top2": float(b.abs_dy),
            "abs_dx_top1": float(a.abs_dx),
            "abs_dx_top2": float(b.abs_dx),
            "residual_dz_top1": float(a.residual_dz),
            "residual_dz_top2": float(b.residual_dz),
            "residual_dy_top1": float(a.residual_dy),
            "residual_dy_top2": float(b.residual_dy),
            "residual_dx_top1": float(a.residual_dx),
            "residual_dx_top2": float(b.residual_dx),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    x_simple = out[SIMPLE_FEATURES].to_numpy(np.float32)
    x_extended = out[EXTENDED_FEATURES].to_numpy(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x_simple=x_simple,
        x_extended=x_extended,
        simple_feature_names=np.asarray(SIMPLE_FEATURES, dtype="U64"),
        extended_feature_names=np.asarray(EXTENDED_FEATURES, dtype="U64"),
        y=out.swap_label.to_numpy(np.int8),
        base_correct=out.base_correct.to_numpy(np.int8),
        top2_correct=out.top2_correct.to_numpy(np.int8),
        groups=out.group.astype(str).to_numpy(dtype="U32"),
        datasets=out.dataset.astype(str).to_numpy(dtype="U64"),
        target_ids=out.target_id.to_numpy(np.int64),
        top1_source_ids=out.top1_source_id.to_numpy(np.int64),
        top2_source_ids=out.top2_source_id.to_numpy(np.int64),
    )

    stats = {
        "rows": int(len(out)),
        "simple_features": SIMPLE_FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "groups": {},
    }
    for group, q in out.groupby("group"):
        stats["groups"][str(group)] = {
            "rows": int(len(q)),
            "swap_positive": int(q.swap_label.sum()),
            "base_errors": int((q.base_correct == 0).sum()),
            "top2_recoverable": int(((q.base_correct == 0) & (q.top2_correct == 1)).sum()),
        }
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
