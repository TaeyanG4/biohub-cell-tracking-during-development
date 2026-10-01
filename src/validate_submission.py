from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def load_shapes(test_root: Path) -> dict[str, tuple[int, int, int, int]]:
    import json

    shapes: dict[str, tuple[int, int, int, int]] = {}
    for zarr_dir in test_root.glob("*.zarr"):
        meta = json.loads((zarr_dir / "0" / "zarr.json").read_text(encoding="utf-8"))
        shape = tuple(int(v) for v in meta["shape"])
        if len(shape) != 4:
            raise ValueError(f"Expected TZYX shape for {zarr_dir.name}, got {shape}")
        shapes[zarr_dir.stem] = shape
    return shapes


def validate(csv_path: Path, test_root: Path) -> None:
    df = pd.read_csv(csv_path)
    if list(df.columns) != REQUIRED:
        raise ValueError(f"Columns must be exactly {REQUIRED}; got {list(df.columns)}")
    if df["id"].tolist() != list(range(len(df))):
        raise ValueError("id must be consecutive integers starting at 0")
    if df["id"].duplicated().any():
        raise ValueError("duplicate row ids")
    if df.isna().any().any():
        raise ValueError("NaN present")
    numeric = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite numeric value present")

    shapes = load_shapes(test_root)
    if not shapes:
        raise ValueError(f"No *.zarr found under {test_root}")
    missing = set(shapes) - set(df["dataset"])
    extra = set(df["dataset"]) - set(shapes)
    if missing or extra:
        raise ValueError(f"dataset mismatch missing={sorted(missing)} extra={sorted(extra)}")
    if not set(df["row_type"]).issubset({"node", "edge"}):
        raise ValueError("row_type must be node or edge")

    for dataset, group in df.groupby("dataset", sort=False):
        T, Z, Y, X = shapes[dataset]
        nodes = group[group.row_type == "node"].copy()
        edges = group[group.row_type == "edge"].copy()
        if nodes["node_id"].duplicated().any():
            raise ValueError(f"{dataset}: duplicate node_id")
        if not ((nodes.t >= 0) & (nodes.t < T)).all():
            raise ValueError(f"{dataset}: out-of-bounds t")
        if not ((nodes.z >= 0) & (nodes.z < Z)).all():
            raise ValueError(f"{dataset}: out-of-bounds z")
        if not ((nodes.y >= 0) & (nodes.y < Y)).all():
            raise ValueError(f"{dataset}: out-of-bounds y")
        if not ((nodes.x >= 0) & (nodes.x < X)).all():
            raise ValueError(f"{dataset}: out-of-bounds x")
        if not (nodes[["source_id", "target_id"]] == -1).all().all():
            raise ValueError(f"{dataset}: node rows must use source_id=target_id=-1")
        node_ids = set(nodes.node_id.astype(int))
        if not (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all():
            raise ValueError(f"{dataset}: edge rows must use unused node fields=-1")
        if not set(edges.source_id.astype(int)).issubset(node_ids) or not set(edges.target_id.astype(int)).issubset(node_ids):
            raise ValueError(f"{dataset}: edge references missing node")
        if (edges.source_id == edges.target_id).any():
            raise ValueError(f"{dataset}: self edge")
        t_by_id = dict(zip(nodes.node_id.astype(int), nodes.t.astype(int)))
        for src, dst in zip(edges.source_id.astype(int), edges.target_id.astype(int)):
            # The patched official metric keeps only adjacent-frame edges. Enforce
            # that stronger invariant rather than merely source.t < target.t.
            if t_by_id[dst] != t_by_id[src] + 1:
                raise ValueError(f"{dataset}: non-adjacent/backward edge {src}->{dst}")

        # A DAG is guaranteed by strictly increasing t, but enforce biological
        # degree constraints explicitly as well.
        outdeg = edges.groupby("source_id").size()
        indeg = edges.groupby("target_id").size()
        if (outdeg > 2).any():
            raise ValueError(f"{dataset}: outdegree > 2")
        if (indeg > 1).any():
            raise ValueError(f"{dataset}: indegree > 1")

    print(f"VALID submission rows={len(df)} datasets={len(shapes)} nodes={(df.row_type == 'node').sum()} edges={(df.row_type == 'edge').sum()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--test-root", type=Path, required=True)
    args = parser.parse_args()
    validate(args.csv, args.test_root)


if __name__ == "__main__":
    main()
