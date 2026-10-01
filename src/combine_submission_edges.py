from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def edge_pairs(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    e = frame[frame.row_type.eq("edge")]
    return {
        (str(r.dataset), int(r.source_id), int(r.target_id))
        for r in e.itertuples()
    }


def main() -> None:
    a_path, b_path, mode, out_path = map(Path, sys.argv[1:5])
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    ea, eb = edge_pairs(a), edge_pairs(b)
    if str(mode) == "intersection":
        keep = ea & eb
    elif str(mode) == "union":
        keep = ea | eb
    else:
        raise ValueError(mode)

    # Keep node coordinates from A. Edges can originate from either file.
    nodes = a[a.row_type.eq("node")].copy()
    edge_rows = pd.concat(
        [a[a.row_type.eq("edge")], b[b.row_type.eq("edge")]],
        ignore_index=True,
    ).drop_duplicates(subset=["dataset", "source_id", "target_id"])
    mask = [
        (str(r.dataset), int(r.source_id), int(r.target_id)) in keep
        for r in edge_rows.itertuples()
    ]
    edges = edge_rows.loc[mask].copy()
    out = pd.concat([nodes, edges], ignore_index=True)
    out = out.sort_values(["dataset", "row_type", "node_id", "source_id", "target_id"]).reset_index(drop=True)
    out["id"] = range(len(out))
    out = out[a.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("edges_a", len(ea), "edges_b", len(eb), "keep", len(keep), "rows", len(out))


if __name__ == "__main__":
    main()
