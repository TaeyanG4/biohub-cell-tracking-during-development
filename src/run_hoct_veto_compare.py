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


def predict_pairs(model, labels: np.ndarray, images: np.ndarray, det: np.ndarray, ids: np.ndarray, chunks: int) -> set[tuple[int, int]]:
    T = labels.shape[0]
    starts = [round(i * T / chunks) for i in range(chunks)] + [T]
    pairs: set[tuple[int, int]] = set()
    scheme = TilingScheme(tile_shape=TILE, overlap_shape=OVERLAP)
    for i in range(chunks):
        s, own_end = starts[i], starts[i + 1]
        frame_end = min(T, own_end + 1)
        sub = np.where((det[:, 0] >= s) & (det[:, 0] < frame_end))[0]
        det_sub = det[sub].copy()
        det_sub[:, 0] -= s
        with torch.inference_mode():
            sol = predict(
                model,
                labels=labels[s:frame_end],
                images=images[s:frame_end],
                scale=(1.0, *SCALE_ZYX.tolist()),
                max_delta_t=1,
                tiling_scheme=scheme,
            )
        nodes = sol.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        edges = sol.edge_attrs(attr_keys=[])
        tloc = nodes["t"].to_numpy().astype(int)
        xyz = np.stack([nodes[c].to_numpy() for c in ("z", "y", "x")], axis=1).astype(float)
        snap = snap_indices(tloc, xyz, det_sub)
        nids = nodes["node_id"].to_list()
        h2p = {int(n): int(ids[sub[k]]) for n, k in zip(nids, snap, strict=True)}
        global_t = {int(n): int(tt) + s for n, tt in zip(nids, tloc, strict=True)}
        for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list(), strict=True):
            if s <= global_t[int(a)] < own_end:
                pairs.add((h2p[int(a)], h2p[int(b)]))
        del sol
        torch.cuda.empty_cache()
    return pairs


def run_one(model, stem: str, group: pd.DataFrame, image_root: Path, chunks: int) -> tuple[pd.DataFrame, dict]:
    node_rows = group[group.row_type.eq("node")].copy()
    edge_rows = group[group.row_type.eq("edge")].copy()
    node_rows["node_id"] = node_rows.node_id.astype(int)
    ids = node_rows.node_id.to_numpy(dtype=int)
    det = node_rows[["t", "z", "y", "x"]].to_numpy(dtype=float)
    arr = zarr.open_group(str(image_root / f"{stem}.zarr"), mode="r")["0"]
    images = np.asarray(arr[:])
    labels = rasterize_spheres(det, tuple(images.shape))
    started = time.time()
    pairs = predict_pairs(model, labels, images, det, ids, chunks)
    source = edge_rows.source_id.astype(int).to_numpy()
    target = edge_rows.target_id.astype(int).to_numpy()
    keep = np.fromiter(((int(s), int(t)) in pairs for s, t in zip(source, target, strict=True)), dtype=bool, count=len(edge_rows))
    out = pd.concat([node_rows, edge_rows.loc[keep]], ignore_index=True)
    stats = {
        "dataset": stem,
        "nodes": int(len(node_rows)),
        "edges_before": int(len(edge_rows)),
        "hoct_pairs": int(len(pairs)),
        "edges_after": int(keep.sum()),
        "removed": int((~keep).sum()),
        "seconds": float(time.time() - started),
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--stats-json", type=Path, required=True)
    p.add_argument("--chunks", type=int, default=4)
    args = p.parse_args()

    model = load_model(args.weights, device="cuda")
    df = pd.read_csv(args.input_csv)
    outputs: list[pd.DataFrame] = []
    stats: list[dict] = []
    for stem in sorted(df.dataset.unique()):
        out, row = run_one(model, stem, df[df.dataset.eq(stem)].copy(), args.image_root, args.chunks)
        outputs.append(out)
        stats.append(row)
        print(row, flush=True)
    result = pd.concat(outputs, ignore_index=True)
    result["id"] = np.arange(len(result), dtype=int)
    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    result[cols].to_csv(args.output_csv, index=False)
    args.stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
