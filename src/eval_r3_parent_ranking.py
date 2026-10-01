from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_hoct_r3_probe import fit_head


def aggregate_edges(d, scores):
    df = pd.DataFrame(
        {
            "dataset": d["datasets"].astype(str),
            "edge_id": d["edge_ids"].astype(np.int64),
            "source_id": d["source_ids"].astype(np.int64),
            "target_id": d["target_ids"].astype(np.int64),
            "label": d["labels"].astype(np.int64),
            "score": scores.astype(np.float64),
        }
    )
    return (
        df.groupby(["dataset", "edge_id", "source_id", "target_id", "label"], as_index=False)
        .agg(score=("score", "mean"), n_views=("score", "size"))
    )


def ranking_metrics(df):
    eligible = []
    reciprocal = []
    top1 = 0
    top2 = 0
    for (_, target), g in df.groupby(["dataset", "target_id"], sort=False):
        if g["label"].sum() <= 0:
            continue
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        pos = np.flatnonzero(g["label"].to_numpy() > 0)
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        eligible.append(1)
        reciprocal.append(1.0 / rank)
        top1 += int(rank == 1)
        top2 += int(rank <= 2)
    n = len(eligible)
    return {
        "targets_with_gt_candidate": n,
        "top1_parent_acc": top1 / n if n else float("nan"),
        "top2_parent_acc": top2 / n if n else float("nan"),
        "mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hoct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--anchor", type=float, default=5e-3)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x = np.asarray(d["features"], np.float32)
    y = np.asarray(d["labels"], np.int64)
    groups = d["groups"].astype(str)
    model = torch.jit.load(args.hoct, map_location="cpu").eval()
    w0 = model.head.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(model.head.bias.detach().cpu().numpy().reshape(-1)[0])

    base_scores = x @ w0 + b0
    rows = []
    for holdout in sorted(set(groups.tolist())):
        tr = groups != holdout
        va = groups == holdout
        head = fit_head(x[tr], y[tr], w0, b0, epochs=args.epochs, lr=args.lr, anchor=args.anchor)
        with torch.no_grad():
            probe_scores = head(torch.from_numpy(x[va])).squeeze(1).numpy()

        base_df = aggregate_edges({k: d[k][va] for k in d.files if len(d[k]) == len(groups)}, base_scores[va])
        probe_df = aggregate_edges({k: d[k][va] for k in d.files if len(d[k]) == len(groups)}, probe_scores)
        row = {"holdout": holdout, "n_rows": int(va.sum()), "n_unique_edges": int(len(base_df))}
        row.update({f"base_{k}": v for k, v in ranking_metrics(base_df).items()})
        row.update({f"probe_{k}": v for k, v in ranking_metrics(probe_df).items()})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"folds": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"folds": rows}, indent=2))


if __name__ == "__main__":
    main()
