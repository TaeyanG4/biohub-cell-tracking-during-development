from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td


SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


def load_graph(path: Path):
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g


def frame_nodes(graph):
    df = graph.node_attrs().select(["node_id", "t", "z", "y", "x"]).sort(["t", "node_id"])
    out = {}
    for t in df["t"].unique().sort().to_list():
        f = df.filter(df["t"] == t)
        out[int(t)] = (
            f["node_id"].to_numpy().astype(np.int64),
            f.select(["z", "y", "x"]).to_numpy().astype(np.float64),
        )
    return out


def snap_one(backbone_path: Path, toql_path: Path, out_path: Path, tol_um: float):
    backbone = load_graph(backbone_path)
    toql = load_graph(toql_path)
    bf = frame_nodes(backbone)
    qf = frame_nodes(toql)
    updates = {}
    matched = 0

    for t in sorted(set(bf) & set(qf)):
        b_ids, b_vox = bf[t]
        q_ids, q_vox = qf[t]
        b_um = b_vox * SCALE
        q_um = q_vox * SCALE
        bt = cKDTree(b_um)
        qt = cKDTree(q_um)
        qd, qi = bt.query(q_um, k=1)
        bd, bi = qt.query(b_um, k=1)
        for b_idx, (dist, q_idx) in enumerate(zip(bd, bi, strict=True)):
            q_idx = int(q_idx)
            if dist > tol_um or float(qd[q_idx]) > tol_um:
                continue
            if int(qi[q_idx]) != b_idx:
                continue
            updates[int(q_ids[q_idx])] = b_vox[b_idx]
            matched += 1

    if updates:
        ids = np.asarray(sorted(updates), dtype=np.int64)
        coords = np.asarray([updates[int(i)] for i in ids], dtype=np.float64)
        toql.update_node_attrs(
            node_ids=ids,
            attrs={"z": coords[:, 0], "y": coords[:, 1], "x": coords[:, 2]},
        )

    from biohub_tracking.io import save_graph

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(toql, out_path)
    return matched, toql.num_nodes(), toql.num_edges()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-dir", type=Path, required=True)
    p.add_argument("--toql-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tol-um", type=float, default=4.0)
    args = p.parse_args()

    for bpath in sorted(args.backbone_dir.glob("*.geff")):
        qpath = args.toql_dir / bpath.name
        if not qpath.exists():
            continue
        matched, nodes, edges = snap_one(
            bpath, qpath, args.output_dir / bpath.name, args.tol_um
        )
        print(bpath.stem, "anchors", matched, "nodes", nodes, "edges", edges, flush=True)


if __name__ == "__main__":
    main()
