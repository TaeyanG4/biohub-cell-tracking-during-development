from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import tracksdata as td
import zarr

from hoct._api import _create_dataset
from hoct.features import create_graph
from hoct.features.constants import EDGE_GT_KEY
from hoct.features.features import add_is_div
from hoct.inference import extract_edge_features
from tracksdata.functional import TilingScheme
from tracksdata.metrics import DistanceMatching


SCALE_ZYX = (1.625, 0.40625, 0.40625)
RADIUS_UM = 3.0
TILE_SHAPE = (5, 32, 128, 128)
OVERLAP_SHAPE = (1, 8, 16, 16)


def rasterize_spheres(nodes: pd.DataFrame, shape: tuple[int, int, int, int]) -> np.ndarray:
    tmax, zmax, ymax, xmax = shape
    labels = np.zeros(shape, dtype=np.uint16)
    sz, sy, sx = SCALE_ZYX
    rz, ry, rx = [int(np.ceil(RADIUS_UM / s)) for s in SCALE_ZYX]
    r2 = RADIUS_UM**2

    for t, frame in nodes.groupby("t", sort=True):
        t = int(t)
        if not 0 <= t < tmax:
            continue
        best = np.full((zmax, ymax, xmax), np.inf, dtype=np.float32)
        lab = labels[t]
        for k, row in enumerate(frame.itertuples(index=False), start=1):
            cz, cy, cx = float(row.z), float(row.y), float(row.x)
            z0, z1 = max(0, int(np.floor(cz)) - rz), min(zmax, int(np.ceil(cz)) + rz + 1)
            y0, y1 = max(0, int(np.floor(cy)) - ry), min(ymax, int(np.ceil(cy)) + ry + 1)
            x0, x1 = max(0, int(np.floor(cx)) - rx), min(xmax, int(np.ceil(cx)) + rx + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            zz = (np.arange(z0, z1) - cz) * sz
            yy = (np.arange(y0, y1) - cy) * sy
            xx = (np.arange(x0, x1) - cx) * sx
            d2 = (zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2).astype(np.float32)
            sub = best[z0:z1, y0:y1, x0:x1]
            hit = (d2 <= r2) & (d2 < sub)
            sub[hit] = d2[hit]
            lab[z0:z1, y0:y1, x0:x1][hit] = k
    return labels


def load_gt(path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def extract_one(
    dataset: str,
    submission: pd.DataFrame,
    image_root: Path,
    gt_root: Path,
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = submission[(submission.dataset == dataset) & (submission.row_type == "node")]
    nodes = nodes[["node_id", "t", "z", "y", "x"]].copy()
    if nodes.empty:
        raise ValueError(f"no predicted nodes for {dataset}")

    zg = zarr.open_group(image_root / f"{dataset}.zarr", mode="r")
    image = zg["0"]
    labels = rasterize_spheres(nodes, tuple(image.shape))
    gt = load_gt(gt_root / f"{dataset}.geff")

    graph = create_graph(
        labels,
        distance_threshold=300.0,
        n_neighbors=5,
        delta_t=1,
        scale=(1.0, *SCALE_ZYX),
        images=image,
        gt_graph=None,
    )

    # Competition GT is point-based and does not carry masks, while HOCT's
    # create_graph(training mode) defaults to MaskMatching. Match with the
    # competition scorer geometry instead: optimal one-to-one, <= 7 um.
    graph.match(
        gt,
        matching=DistanceMatching(
            max_distance=7.0,
            optimal=True,
            attr_keys=("z", "y", "x"),
            scale=SCALE_ZYX,
        ),
    )
    add_is_div(graph, gt)
    gt_edge_ids = td.functional.ancestral_connected_edges(graph, gt, match=False)
    graph.add_edge_attr_key(EDGE_GT_KEY, pl.Boolean, False)
    graph.update_edge_attrs(attrs={EDGE_GT_KEY: True}, edge_ids=gt_edge_ids)
    # Match production HOCT's spatial tiling. Full-frame temporal windows can
    # become pathological in dense movies because attention scales poorly with
    # the number of candidate edges in a window.
    ds = _create_dataset(
        graph,
        tiling_scheme=TilingScheme(tile_shape=TILE_SHAPE, overlap_shape=OVERLAP_SHAPE),
        window_size=5,
        test_time_augs=0,
    )
    feat_df = extract_edge_features(model, ds)
    attrs = graph.edge_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.EDGE_ID,
            td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
            EDGE_GT_KEY,
        ]
    )
    joined = feat_df.join(attrs, on=td.DEFAULT_ATTR_KEYS.EDGE_ID, how="inner")

    raw = joined["edge_features"].to_torch().float()
    with torch.inference_mode():
        norm = model.head_norm(raw.to(next(model.parameters()).device)).float().cpu().numpy()
    labels_y = joined[EDGE_GT_KEY].to_numpy().astype(np.int64)
    edge_ids = joined[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy().astype(np.int64)
    source_ids = joined[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy().astype(np.int64)
    target_ids = joined[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy().astype(np.int64)
    groups = np.full(len(labels_y), dataset.split("_", 1)[0], dtype="U32")

    print(
        f"{dataset}: nodes={len(nodes)} candidates={graph.num_edges()} feature_rows={len(labels_y)} "
        f"positives={int(labels_y.sum())} raw_shape={tuple(raw.shape)}"
    )
    return norm.astype(np.float32), labels_y, groups, edge_ids, source_ids, target_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    submission = pd.read_csv(args.submission)
    model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
    print("device", args.device, "head", tuple(model.head.weight.shape))

    feats, ys, groups, edge_ids, source_ids, target_ids, datasets = [], [], [], [], [], [], []
    for name in args.datasets:
        x, y, g, e, s, t = extract_one(name, submission, args.image_root, args.gt_root, model)
        feats.append(x); ys.append(y); groups.append(g); edge_ids.append(e)
        source_ids.append(s); target_ids.append(t)
        datasets.append(np.full(len(y), name, dtype="U64"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.concatenate(feats),
        labels=np.concatenate(ys),
        groups=np.concatenate(groups),
        edge_ids=np.concatenate(edge_ids),
        source_ids=np.concatenate(source_ids),
        target_ids=np.concatenate(target_ids),
        datasets=np.concatenate(datasets),
    )
    print("saved", args.out, "rows", sum(map(len, ys)))


if __name__ == "__main__":
    main()
