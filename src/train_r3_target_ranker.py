from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def aggregate_unique(d, w0, b0):
    x = np.asarray(d["features"], np.float32)
    base = x @ w0 + b0
    df = pd.DataFrame(
        {
            "dataset": d["datasets"].astype(str),
            "group": d["groups"].astype(str),
            "edge_id": d["edge_ids"].astype(np.int64),
            "source_id": d["source_ids"].astype(np.int64),
            "target_id": d["target_ids"].astype(np.int64),
            "label": d["labels"].astype(np.int64),
            "base": base.astype(np.float32),
        }
    )
    # Keep first row indices per edge, then average features across repeated windows.
    keys = ["dataset", "group", "edge_id", "source_id", "target_id", "label"]
    groups = df.groupby(keys, sort=False).indices
    rows = []
    feats = []
    for key, idx in groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        rows.append((*key, float(base[idx].mean())))
        feats.append(x[idx].mean(axis=0))
    meta = pd.DataFrame(rows, columns=keys + ["base"])
    return meta, np.asarray(feats, np.float32)


def build_pairs(meta, feat, train_mask, hard_k=4):
    sub_idx = np.flatnonzero(train_mask)
    sub = meta.iloc[sub_idx].copy()
    sub["global_idx"] = sub_idx
    pos_idx = []
    neg_idx = []
    targets = 0
    for (_, target), g in sub.groupby(["dataset", "target_id"], sort=False):
        pos = g[g.label > 0]
        neg = g[g.label == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        targets += 1
        # Usually one positive parent per target. Keep every positive if present.
        hard = neg.nlargest(min(hard_k, len(neg)), "base")
        for p in pos.global_idx.to_numpy():
            for n in hard.global_idx.to_numpy():
                pos_idx.append(int(p)); neg_idx.append(int(n))
    return np.asarray(pos_idx, np.int64), np.asarray(neg_idx, np.int64), targets


def fit_ranker(feat, pos_idx, neg_idx, w0, b0, epochs, lr, anchor, device):
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    head = torch.nn.Linear(feat.shape[1], 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(w0, dtype=torch.float32, device=device).view(1, -1))
        head.bias.copy_(torch.as_tensor([b0], dtype=torch.float32, device=device))
    w_ref = head.weight.detach().clone(); b_ref = head.bias.detach().clone()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    x = torch.from_numpy(feat)
    p = torch.from_numpy(pos_idx)
    n = torch.from_numpy(neg_idx)
    batch = 16384
    for _ in range(epochs):
        order = torch.randperm(len(p))
        for st in range(0, len(p), batch):
            q = order[st:st + batch]
            xp = x[p[q]].to(device, non_blocking=True)
            xn = x[n[q]].to(device, non_blocking=True)
            diff = head(xp) - head(xn)
            loss = torch.nn.functional.softplus(-diff).mean()
            loss = loss + anchor * (
                (head.weight - w_ref).square().sum() + (head.bias - b_ref).square().sum()
            )
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head.cpu()


def rank_metrics(meta, scores, mask):
    m = meta.loc[mask].copy()
    m["score"] = scores[mask]
    n = top1 = top2 = 0
    rr = []
    for (_, target), g in m.groupby(["dataset", "target_id"], sort=False):
        if g.label.sum() <= 0:
            continue
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        pos = np.flatnonzero(g.label.to_numpy() > 0)
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        n += 1; top1 += rank == 1; top2 += rank <= 2; rr.append(1.0 / rank)
    return {
        "n_targets": n,
        "top1": top1 / n if n else float("nan"),
        "top2": top2 / n if n else float("nan"),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hoct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--hard-k", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    m = torch.jit.load(args.hoct, map_location="cpu").eval()
    w0 = m.head.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(m.head.bias.detach().cpu().numpy().reshape(-1)[0])
    meta, feat = aggregate_unique(d, w0, b0)
    base = feat @ w0 + b0

    rows = []
    for holdout in sorted(meta.group.unique()):
        tr = meta.group.to_numpy() != holdout
        va = ~tr
        p, n, n_targets = build_pairs(meta, feat, tr, hard_k=args.hard_k)
        h = fit_ranker(feat, p, n, w0, b0, args.epochs, args.lr, args.anchor, args.device)
        with torch.no_grad():
            pred = h(torch.from_numpy(feat)).squeeze(1).numpy()
        row = {"holdout": holdout, "train_targets": n_targets, "pairs": int(len(p))}
        row.update({f"base_{k}": v for k, v in rank_metrics(meta, base, va).items()})
        row.update({f"ranker_{k}": v for k, v in rank_metrics(meta, pred, va).items()})
        row["weight_l2_drift"] = float(np.linalg.norm(h.weight.detach().numpy().reshape(-1) - w0))
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"folds": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"folds": rows}, indent=2))


if __name__ == "__main__":
    main()
