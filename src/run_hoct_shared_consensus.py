from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.spatial import cKDTree
from tracksdata.functional import TilingScheme

from hoct import load_model, predict
from hoct.features import create_graph


SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
RADIUS_UM = 3.0
TILE = (5, 32, 128, 128)
OVERLAP = (1, 8, 16, 16)


def rasterize_spheres(det: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    T, Z, Y, X = shape
    labels = np.zeros(shape, dtype=np.int16)
    rz, ry, rx = (int(math.ceil(RADIUS_UM / s)) for s in SCALE_ZYX)
    r2 = RADIUS_UM**2
    t_all = det[:, 0].astype(int)
    for t in range(T):
        rows = det[t_all == t]
        best = np.full((Z, Y, X), np.inf, dtype=np.float32)
        lab = labels[t]
        for k, (_, cz, cy, cx) in enumerate(rows, start=1):
            z0, z1 = max(0, int(math.floor(cz)) - rz), min(Z, int(math.ceil(cz)) + rz + 1)
            y0, y1 = max(0, int(math.floor(cy)) - ry), min(Y, int(math.ceil(cy)) + ry + 1)
            x0, x1 = max(0, int(math.floor(cx)) - rx), min(X, int(math.ceil(cx)) + rx + 1)
            zz = (np.arange(z0, z1) - cz) * SCALE_ZYX[0]
            yy = (np.arange(y0, y1) - cy) * SCALE_ZYX[1]
            xx = (np.arange(x0, x1) - cx) * SCALE_ZYX[2]
            d2 = (zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2).astype(np.float32)
            sub = best[z0:z1, y0:y1, x0:x1]
            hit = (d2 <= r2) & (d2 < sub)
            sub[hit] = d2[hit]
            lab[z0:z1, y0:y1, x0:x1][hit] = k
    return labels


def snap_indices(t: np.ndarray, zyx: np.ndarray, det: np.ndarray) -> np.ndarray:
    out = np.full(len(t), -1, dtype=np.int64)
    for tt in np.unique(t):
        q = np.where(t == tt)[0]
        d = np.where(det[:, 0].astype(int) == int(tt))[0]
        if not len(d):
            raise RuntimeError(f"no pipeline nodes at frame {tt}")
        _, k = cKDTree(det[d, 1:] * SCALE_ZYX).query(zyx[q] * SCALE_ZYX, k=1)
        out[q] = d[k]
    if len(np.unique(out)) != len(out):
        raise RuntimeError("HOCT nodes do not map one-to-one onto pipeline nodes")
    return out


def solution_pairs(sol, det_sub: np.ndarray, ids_sub: np.ndarray, global_start: int, own_end: int) -> set[tuple[int, int]]:
    nodes = sol.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = sol.edge_attrs(attr_keys=[])
    tloc = nodes["t"].to_numpy().astype(int)
    zyx = np.stack([nodes[c].to_numpy() for c in ("z", "y", "x")], axis=1).astype(float)
    snap = snap_indices(tloc, zyx, det_sub)
    nids = nodes["node_id"].to_list()
    h2p = {int(n): int(ids_sub[k]) for n, k in zip(nids, snap, strict=True)}
    global_t = {int(n): int(tt) + global_start for n, tt in zip(nids, tloc, strict=True)}
    pairs: set[tuple[int, int]] = set()
    for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list(), strict=True):
        if global_start <= global_t[int(a)] < own_end:
            pairs.add((h2p[int(a)], h2p[int(b)]))
    return pairs


def predict_pair_sets(model0, model1, labels, images, det, ids, chunks: int):
    T = labels.shape[0]
    starts = [round(i * T / chunks) for i in range(chunks)] + [T]
    scheme = TilingScheme(tile_shape=TILE, overlap_shape=OVERLAP)
    p0: set[tuple[int, int]] = set()
    p1: set[tuple[int, int]] = set()
    timings = {"create_graph": 0.0, "v0": 0.0, "v1": 0.0}
    for i in range(chunks):
        s, own_end = starts[i], starts[i + 1]
        frame_end = min(T, own_end + 1)
        sub = np.where((det[:, 0] >= s) & (det[:, 0] < frame_end))[0]
        det_sub = det[sub].copy()
        det_sub[:, 0] -= s
        t0 = time.time()
        graph = create_graph(
            labels=labels[s:frame_end],
            images=images[s:frame_end],
            gt_graph=None,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=1,
            scale=(1.0, *SCALE_ZYX.tolist()),
        )
        timings["create_graph"] += time.time() - t0
        g0 = graph.copy()
        g1 = graph.copy()
        del graph

        t0 = time.time()
        with torch.inference_mode():
            sol0 = predict(model0, graph=g0, max_delta_t=1, tiling_scheme=scheme)
        timings["v0"] += time.time() - t0
        p0 |= solution_pairs(sol0, det_sub, ids[sub], s, own_end)
        del sol0, g0
        torch.cuda.empty_cache()

        t0 = time.time()
        with torch.inference_mode():
            sol1 = predict(model1, graph=g1, max_delta_t=1, tiling_scheme=scheme)
        timings["v1"] += time.time() - t0
        p1 |= solution_pairs(sol1, det_sub, ids[sub], s, own_end)
        del sol1, g1
        torch.cuda.empty_cache()
    return p0, p1, timings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--weights-v0", type=Path, required=True)
    p.add_argument("--weights-v1", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--stats-json", type=Path, required=True)
    p.add_argument("--chunks", type=int, default=4)
    args = p.parse_args()

    model0 = load_model(args.weights_v0, device="cuda")
    model1 = load_model(args.weights_v1, device="cuda")
    df = pd.read_csv(args.input_csv)
    outputs = []
    stats = []
    for stem in sorted(df.dataset.unique()):
        group = df[df.dataset.eq(stem)].copy()
        nodes = group[group.row_type.eq("node")].copy()
        edges = group[group.row_type.eq("edge")].copy()
        nodes["node_id"] = nodes.node_id.astype(int)
        ids = nodes.node_id.to_numpy(dtype=int)
        det = nodes[["t", "z", "y", "x"]].to_numpy(dtype=float)
        arr = zarr.open_group(str(args.image_root / f"{stem}.zarr"), mode="r")["0"]
        images = np.asarray(arr[:])
        labels = rasterize_spheres(det, tuple(images.shape))
        started = time.time()
        p0, p1, timings = predict_pair_sets(model0, model1, labels, images, det, ids, args.chunks)
        keep_pairs = p0 & p1
        source = edges.source_id.astype(int).to_numpy()
        target = edges.target_id.astype(int).to_numpy()
        keep = np.fromiter(((int(s), int(t)) in keep_pairs for s, t in zip(source, target, strict=True)), dtype=bool, count=len(edges))
        outputs.append(pd.concat([nodes, edges.loc[keep]], ignore_index=True))
        row = {
            "dataset": stem,
            "nodes": len(nodes),
            "edges_before": len(edges),
            "v0_pairs": len(p0),
            "v1_pairs": len(p1),
            "intersection_pairs": len(keep_pairs),
            "edges_after": int(keep.sum()),
            "removed": int((~keep).sum()),
            "seconds": time.time() - started,
            **{f"seconds_{k}": v for k, v in timings.items()},
        }
        stats.append(row)
        print(row, flush=True)

    out = pd.concat(outputs, ignore_index=True)
    out["id"] = np.arange(len(out), dtype=int)
    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    out[cols].to_csv(args.output_csv, index=False)
    args.stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
