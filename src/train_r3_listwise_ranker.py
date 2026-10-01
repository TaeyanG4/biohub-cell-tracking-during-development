from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TargetBatch:
    features: np.ndarray
    base_scores: np.ndarray
    valid: np.ndarray
    positive: np.ndarray
    datasets: np.ndarray
    groups: np.ndarray
    target_ids: np.ndarray
    edge_ids: np.ndarray
    labels: np.ndarray


def build_target_batch(d: np.lib.npyio.NpzFile) -> TargetBatch:
    feat = np.asarray(d["features"], np.float32)
    base = np.asarray(d["base_scores"], np.float32)
    labels = np.asarray(d["labels"], np.int64)
    datasets = d["datasets"].astype(str)
    groups = d["groups"].astype(str)
    target_ids = np.asarray(d["target_ids"], np.int64)
    edge_ids = np.asarray(d["edge_ids"], np.int64)

    keys: dict[tuple[str, int], list[int]] = {}
    for i, key in enumerate(zip(datasets, target_ids)):
        keys.setdefault((str(key[0]), int(key[1])), []).append(i)

    kept: list[list[int]] = []
    meta: list[tuple[str, str, int]] = []
    for (dataset, target), idx in keys.items():
        arr = np.asarray(idx, np.int64)
        y = labels[arr]
        if y.sum() <= 0 or np.all(y > 0):
            continue
        kept.append(idx)
        meta.append((dataset, str(groups[arr[0]]), int(target)))

    max_cands = max(map(len, kept))
    n = len(kept)
    x = np.zeros((n, max_cands, feat.shape[1]), dtype=np.float32)
    b = np.full((n, max_cands), -np.inf, dtype=np.float32)
    valid = np.zeros((n, max_cands), dtype=bool)
    positive = np.zeros((n, max_cands), dtype=bool)
    eids = np.full((n, max_cands), -1, dtype=np.int64)
    labs = np.zeros((n, max_cands), dtype=np.int8)
    for row, idx in enumerate(kept):
        arr = np.asarray(idx, np.int64)
        m = len(arr)
        x[row, :m] = feat[arr]
        b[row, :m] = base[arr]
        valid[row, :m] = True
        positive[row, :m] = labels[arr] > 0
        eids[row, :m] = edge_ids[arr]
        labs[row, :m] = labels[arr].astype(np.int8)

    return TargetBatch(
        features=x,
        base_scores=b,
        valid=valid,
        positive=positive,
        datasets=np.asarray([m[0] for m in meta], dtype="U64"),
        groups=np.asarray([m[1] for m in meta], dtype="U32"),
        target_ids=np.asarray([m[2] for m in meta], dtype=np.int64),
        edge_ids=eids,
        labels=labs,
    )


class ResidualRanker(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.delta = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.delta.weight)
        torch.nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.delta(x).squeeze(-1)


def fit(
    data: TargetBatch,
    train_mask: np.ndarray,
    *,
    epochs: int,
    lr: float,
    anchor: float,
    batch_size: int,
    device: str,
    seed: int,
) -> ResidualRanker:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ResidualRanker(data.features.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    train_idx = np.flatnonzero(train_mask)

    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for st in range(0, len(order), batch_size):
            idx = order[st:st + batch_size]
            x = torch.from_numpy(data.features[idx]).to(dev)
            base = torch.from_numpy(data.base_scores[idx]).to(dev)
            valid = torch.from_numpy(data.valid[idx]).to(dev)
            pos = torch.from_numpy(data.positive[idx]).to(dev)

            scores = model(x, base)
            neg_inf = torch.finfo(scores.dtype).min
            all_scores = scores.masked_fill(~valid, neg_inf)
            pos_scores = scores.masked_fill(~pos, neg_inf)
            loss = (torch.logsumexp(all_scores, dim=1) - torch.logsumexp(pos_scores, dim=1)).mean()
            loss = loss + anchor * (
                model.delta.weight.square().sum() + model.delta.bias.square().sum()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.cpu()


def predict(data: TargetBatch, model: ResidualRanker, device: str) -> np.ndarray:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    out = np.empty_like(data.base_scores, dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(data.features), 1024):
            sl = slice(st, st + 1024)
            x = torch.from_numpy(data.features[sl]).to(dev)
            b = torch.from_numpy(data.base_scores[sl]).to(dev)
            out[sl] = model(x, b).cpu().numpy()
    return out


def metrics(data: TargetBatch, scores: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.flatnonzero(mask)
    top1 = top2 = 0
    rr: list[float] = []
    winners: list[int] = []
    correct: list[bool] = []
    for row in idx:
        valid = data.valid[row]
        s = scores[row, valid]
        y = data.labels[row, valid]
        e = data.edge_ids[row, valid]
        order = np.argsort(-s, kind="stable")
        pos = np.flatnonzero(y[order] > 0)
        rank = int(pos[0]) + 1
        top1 += int(rank == 1)
        top2 += int(rank <= 2)
        rr.append(1.0 / rank)
        winners.append(int(e[order[0]]))
        correct.append(bool(y[order[0]] > 0))
    n = len(idx)
    return {
        "n_targets": int(n),
        "top1": top1 / n if n else float("nan"),
        "top2": top2 / n if n else float("nan"),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
        "winners": winners,
        "correct": correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    data = build_target_batch(d)
    base = data.base_scores.copy()
    rows = []
    for holdout in sorted(np.unique(data.groups)):
        train_mask = data.groups != holdout
        val_mask = ~train_mask
        model = fit(
            data,
            train_mask,
            epochs=args.epochs,
            lr=args.lr,
            anchor=args.anchor,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        pred = predict(data, model, args.device)
        bm = metrics(data, base, val_mask)
        pm = metrics(data, pred, val_mask)
        bw = np.asarray(bm.pop("winners"), np.int64)
        pw = np.asarray(pm.pop("winners"), np.int64)
        bc = np.asarray(bm.pop("correct"), bool)
        pc = np.asarray(pm.pop("correct"), bool)
        changed = bw != pw
        rows.append({
            "holdout": holdout,
            "train_targets": int(train_mask.sum()),
            "base": bm,
            "ranker": pm,
            "winner_changes": int(changed.sum()),
            "fixes": int((changed & (~bc) & pc).sum()),
            "breaks": int((changed & bc & (~pc)).sum()),
            "both_correct_changed": int((changed & bc & pc).sum()),
            "delta_weight_l2": float(torch.linalg.vector_norm(model.delta.weight.detach()).item()),
            "delta_bias": float(model.delta.bias.detach().item()),
        })

    result = {
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "anchor": args.anchor,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
        },
        "n_targets": int(len(data.groups)),
        "folds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
