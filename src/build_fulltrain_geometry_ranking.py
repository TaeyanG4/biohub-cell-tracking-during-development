from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import tracksdata as td


SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def load_graph(path: Path):
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def frame_nn_dist(pos: np.ndarray) -> np.ndarray:
    if len(pos) <= 1:
        return np.full(len(pos), 99.0, dtype=np.float32)
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=2)
    return d[:, 1].astype(np.float32)


def build_movie(path: Path, radius: float) -> tuple[list[dict], dict]:
    dataset = path.stem
    embryo = dataset.split("_", 1)[0]
    g = load_graph(path)
    ndf = g.node_attrs().to_pandas()
    edf = g.edge_attrs().to_pandas()
    ndf = ndf[["node_id", "t", "z", "y", "x"]].copy()
    ndf["node_id"] = ndf.node_id.astype(np.int64)
    ndf["t"] = ndf.t.astype(np.int64)
    pos = ndf[["z", "y", "x"]].to_numpy(np.float32) * SCALE[None, :]
    ids = ndf.node_id.to_numpy(np.int64)
    times = ndf.t.to_numpy(np.int64)
    id_to_idx = {int(n): i for i, n in enumerate(ids)}

    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for r in edf.itertuples(index=False):
        s, t = int(r.source_id), int(r.target_id)
        parent[t] = s
        children[s].append(t)

    age_cache: dict[int, int] = {}

    def age(nid: int) -> int:
        if nid in age_cache:
            return age_cache[nid]
        p = parent.get(nid)
        v = 1 if p is None else min(1000, age(p) + 1)
        age_cache[nid] = v
        return v

    frames: dict[int, np.ndarray] = {}
    trees: dict[int, cKDTree] = {}
    frame_nn: dict[int, np.ndarray] = {}
    for t in np.unique(times):
        idx = np.flatnonzero(times == t)
        frames[int(t)] = idx
        trees[int(t)] = cKDTree(pos[idx])
        frame_nn[int(t)] = frame_nn_dist(pos[idx])

    rows: list[dict] = []
    true_steps = []
    n_parent_edges = 0
    n_covered = 0
    n_supervised = 0
    nearest_correct = 0
    velocity_correct = 0

    for target_id, true_source in parent.items():
        ti = id_to_idx.get(target_id)
        si = id_to_idx.get(true_source)
        if ti is None or si is None:
            continue
        t = int(times[ti])
        if int(times[si]) != t - 1:
            continue
        n_parent_edges += 1
        step_true = pos[ti] - pos[si]
        true_dist = float(np.linalg.norm(step_true))
        true_steps.append(true_dist)
        src_idx = frames.get(t - 1)
        if src_idx is None or len(src_idx) == 0:
            continue
        local = trees[t - 1].query_ball_point(pos[ti], r=radius)
        if not local:
            continue
        cand_idx = src_idx[np.asarray(local, dtype=np.int64)]
        cand_ids = ids[cand_idx]
        pos_hits = np.flatnonzero(cand_ids == true_source)
        if len(pos_hits) == 0:
            continue
        n_covered += 1
        if len(cand_idx) < 2:
            continue
        n_supervised += 1

        cand_pos = pos[cand_idx]
        steps = pos[ti][None, :] - cand_pos
        dists = np.linalg.norm(steps, axis=1)
        dist_rank = np.argsort(np.argsort(dists, kind="stable"), kind="stable") + 1
        nearest_correct += int(int(cand_ids[int(np.argmin(dists))]) == true_source)

        velocity_scores = []
        target_local_idx = frames[t]
        target_local_pos = pos[target_local_idx]
        target_tree = trees[t]
        target_local_row = int(np.flatnonzero(target_local_idx == ti)[0])
        target_nn = float(frame_nn[t][target_local_row])
        candidate_count = len(cand_idx)
        counts = {
            4: int(np.sum(dists <= 4.0)),
            6: int(np.sum(dists <= 6.0)),
            8: int(np.sum(dists <= 8.0)),
            10: int(np.sum(dists <= 10.0)),
            14: int(np.sum(dists <= 14.0)),
        }

        for j, (ci, source_id) in enumerate(zip(cand_idx, cand_ids)):
            source_id = int(source_id)
            step = steps[j]
            prev_id = parent.get(source_id)
            has_prev = int(prev_id is not None and prev_id in id_to_idx)
            if has_prev:
                pi = id_to_idx[int(prev_id)]
                prev_vec = pos[ci] - pos[pi]
                prev_speed = float(np.linalg.norm(prev_vec))
                residual_vec = step - prev_vec
                residual = float(np.linalg.norm(residual_vec))
                cosine = safe_cos(step, prev_vec)
                speed_ratio = float(np.linalg.norm(step) / max(prev_speed, 0.25))
            else:
                prev_vec = np.zeros(3, dtype=np.float32)
                prev_speed = 0.0
                residual_vec = step.copy()
                residual = float(np.linalg.norm(step))
                cosine = 0.0
                speed_ratio = 1.0

            src_frame_idx = frames[t - 1]
            src_local_row = int(np.flatnonzero(src_frame_idx == ci)[0])
            src_nn = float(frame_nn[t - 1][src_local_row])
            score_vel = residual if has_prev else float(dists[j])
            velocity_scores.append(score_vel)
            rows.append(
                {
                    "dataset": dataset,
                    "group": embryo,
                    "target_id": int(target_id),
                    "source_id": source_id,
                    "label": int(source_id == true_source),
                    "t": t,
                    "distance": float(dists[j]),
                    "abs_dz": float(abs(step[0])),
                    "abs_dy": float(abs(step[1])),
                    "abs_dx": float(abs(step[2])),
                    "distance_rank": int(dist_rank[j]),
                    "candidate_count": int(candidate_count),
                    "count_r4": counts[4],
                    "count_r6": counts[6],
                    "count_r8": counts[8],
                    "count_r10": counts[10],
                    "count_r14": counts[14],
                    "has_prev": has_prev,
                    "prev_speed": prev_speed,
                    "velocity_residual": residual,
                    "velocity_cosine": cosine,
                    "speed_ratio": speed_ratio,
                    "residual_dz": float(abs(residual_vec[0])),
                    "residual_dy": float(abs(residual_vec[1])),
                    "residual_dx": float(abs(residual_vec[2])),
                    "source_track_age": int(age(source_id)),
                    "source_nn": src_nn,
                    "target_nn": target_nn,
                    "source_outdegree_gt": int(len(children.get(source_id, []))),
                }
            )
        velocity_correct += int(int(cand_ids[int(np.argmin(np.asarray(velocity_scores)))]) == true_source)

    true_steps_arr = np.asarray(true_steps, dtype=np.float32)
    stats = {
        "dataset": dataset,
        "group": embryo,
        "nodes": int(len(ndf)),
        "edges": int(len(edf)),
        "parent_edges_consecutive": int(n_parent_edges),
        "candidate_covered": int(n_covered),
        "candidate_coverage": float(n_covered / n_parent_edges) if n_parent_edges else float("nan"),
        "supervised_targets": int(n_supervised),
        "nearest_top1": float(nearest_correct / n_supervised) if n_supervised else float("nan"),
        "velocity_top1": float(velocity_correct / n_supervised) if n_supervised else float("nan"),
        "step_p50": float(np.quantile(true_steps_arr, 0.50)) if len(true_steps_arr) else float("nan"),
        "step_p90": float(np.quantile(true_steps_arr, 0.90)) if len(true_steps_arr) else float("nan"),
        "step_p99": float(np.quantile(true_steps_arr, 0.99)) if len(true_steps_arr) else float("nan"),
        "divisions": int(sum(len(v) >= 2 for v in children.values())),
    }
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=14.0)
    args = ap.parse_args()

    all_rows: list[dict] = []
    all_stats: list[dict] = []
    paths = sorted(args.gt_root.glob("*.geff"))
    if not paths:
        raise RuntimeError(f"No GEFF directories under {args.gt_root}")
    for i, path in enumerate(paths, 1):
        rows, stats = build_movie(path, args.radius)
        all_rows.extend(rows)
        all_stats.append(stats)
        if i % 20 == 0 or i == len(paths):
            print("processed", i, "of", len(paths), "candidate_rows", len(all_rows), flush=True)

    frame = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    summary = {
        "radius": args.radius,
        "movies": len(paths),
        "candidate_rows": int(len(frame)),
        "supervised_targets": int(sum(x["supervised_targets"] for x in all_stats)),
        "groups": {},
        "per_movie": all_stats,
    }
    for group in sorted({x["group"] for x in all_stats}):
        ss = [x for x in all_stats if x["group"] == group]
        covered = sum(x["candidate_covered"] for x in ss)
        edges = sum(x["parent_edges_consecutive"] for x in ss)
        sup = sum(x["supervised_targets"] for x in ss)
        nearest = sum(x["nearest_top1"] * x["supervised_targets"] for x in ss if x["supervised_targets"])
        velocity = sum(x["velocity_top1"] * x["supervised_targets"] for x in ss if x["supervised_targets"])
        summary["groups"][group] = {
            "movies": len(ss),
            "edges": int(edges),
            "coverage": float(covered / edges) if edges else float("nan"),
            "supervised_targets": int(sup),
            "nearest_top1": float(nearest / sup) if sup else float("nan"),
            "velocity_top1": float(velocity / sup) if sup else float("nan"),
            "divisions": int(sum(x["divisions"] for x in ss)),
        }
    args.stats.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_movie"}, indent=2))


if __name__ == "__main__":
    main()
