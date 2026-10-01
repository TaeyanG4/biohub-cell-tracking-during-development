from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "source_id", "target_id"]


def edge_keys(df: pd.DataFrame) -> set[tuple[str, int, int]]:
    edges = df[df.row_type.eq("edge")]
    return {
        (str(r.dataset), int(r.source_id), int(r.target_id))
        for r in edges.itertuples(index=False)
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v0", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--output-intersection", type=Path, required=True)
    p.add_argument("--output-union", type=Path, required=True)
    args = p.parse_args()

    v0 = pd.read_csv(args.v0)
    v1 = pd.read_csv(args.v1)
    n0 = v0[v0.row_type.eq("node")].copy()
    n1 = v1[v1.row_type.eq("node")].copy()
    cols = ["dataset", "node_id", "t", "z", "y", "x"]
    if not n0[cols].reset_index(drop=True).equals(n1[cols].reset_index(drop=True)):
        raise RuntimeError("v0 and v1 node tables differ")

    e0 = edge_keys(v0)
    e1 = edge_keys(v1)
    inter = e0 & e1
    union = e0 | e1
    source_edges = pd.concat(
        [v0[v0.row_type.eq("edge")], v1[v1.row_type.eq("edge")]],
        ignore_index=True,
    ).drop_duplicates(KEYS)

    for name, keys, path in (
        ("intersection", inter, args.output_intersection),
        ("union", union, args.output_union),
    ):
        mask = [
            (str(r.dataset), int(r.source_id), int(r.target_id)) in keys
            for r in source_edges.itertuples(index=False)
        ]
        out = pd.concat([n0, source_edges.loc[mask]], ignore_index=True)
        out["id"] = np.arange(len(out), dtype=int)
        ordered = [
            "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
            "source_id", "target_id",
        ]
        out[ordered].to_csv(path, index=False)
        print(name, "edges", len(keys), "v0_only", len(e0 - e1), "v1_only", len(e1 - e0))


if __name__ == "__main__":
    main()
