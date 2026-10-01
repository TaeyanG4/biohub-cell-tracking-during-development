from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ModelConfig:
    feature_key: str
    hidden: int
    pos_weight: float
    lr: float
    weight_decay: float
    epochs: int


class SwapNet(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        if hidden <= 0:
            self.net = torch.nn.Linear(dim, 1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def movie_fold(name: str, folds: int) -> int:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % folds


def standardize(train_x: np.ndarray, other: list[np.ndarray]):
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    tx = np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32)
    outs = [np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32) for x in other]
    return tx, outs


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: ModelConfig,
    device: str,
    seed: int,
) -> SwapNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SwapNet(x.shape[1], cfg.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    pw = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(cfg.epochs):
        logits = model(xt)
        loss = loss_fn(logits, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.cpu().eval()


def predict(model: SwapNet, x: np.ndarray, device: str) -> np.ndarray:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    with torch.inference_mode():
        p = torch.sigmoid(model(torch.from_numpy(x).to(dev))).cpu().numpy()
    return p.astype(np.float32)


def apply_stats(
    prob: np.ndarray,
    threshold: float | None,
    base_correct: np.ndarray,
    top2_correct: np.ndarray,
) -> dict:
    if threshold is None:
        use = np.zeros(len(prob), dtype=bool)
    else:
        use = prob >= threshold
    base = base_correct.astype(bool)
    top2 = top2_correct.astype(bool)
    fixes = int(((~base) & top2 & use).sum())
    breaks = int((base & (~top2) & use).sum())
    neutral_correct = int((base & top2 & use).sum())
    neutral_wrong = int(((~base) & (~top2) & use).sum())
    net = fixes - breaks
    return {
        "rows": int(len(prob)),
        "base_correct": int(base.sum()),
        "base_accuracy": float(base.mean()) if len(base) else float("nan"),
        "threshold": None if threshold is None else float(threshold),
        "swaps": int(use.sum()),
        "fixes": fixes,
        "breaks": breaks,
        "net": net,
        "neutral_correct": neutral_correct,
        "neutral_wrong": neutral_wrong,
        "accuracy_after": float((base.sum() + net) / len(base)) if len(base) else float("nan"),
    }


def choose_threshold(
    prob: np.ndarray,
    base_correct: np.ndarray,
    top2_correct: np.ndarray,
) -> tuple[float | None, dict]:
    order = np.argsort(-prob, kind="stable")
    p = prob[order]
    base = base_correct[order].astype(bool)
    top2 = top2_correct[order].astype(bool)
    fix_c = np.cumsum((~base) & top2)
    break_c = np.cumsum(base & (~top2))
    neutral_wrong_c = np.cumsum((~base) & (~top2))
    best = None
    for k in range(1, len(p) + 1):
        if k < len(p) and p[k - 1] == p[k]:
            continue
        fixes = int(fix_c[k - 1])
        breaks = int(break_c[k - 1])
        neutral_wrong = int(neutral_wrong_c[k - 1])
        net = fixes - breaks
        if net <= 0:
            continue
        if fixes < 2 * breaks:
            continue
        score = (net, fixes, -breaks, -neutral_wrong, -k)
        threshold = float(p[k - 1] - 1e-7)
        if best is None or score > best[0]:
            best = (score, threshold)
    if best is None:
        stats = apply_stats(prob, None, base_correct, top2_correct)
        return None, stats
    threshold = best[1]
    return threshold, apply_stats(prob, threshold, base_correct, top2_correct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = np.asarray(d["y"], dtype=np.int8)
    groups = d["groups"].astype(str)
    datasets = d["datasets"].astype(str)
    base_correct = np.asarray(d["base_correct"], dtype=np.int8)
    top2_correct = np.asarray(d["top2_correct"], dtype=np.int8)
    features = {
        "simple": np.asarray(d["x_simple"], dtype=np.float32),
        "extended": np.asarray(d["x_extended"], dtype=np.float32),
    }

    configs = []
    for feature_key, hidden, pos_weight, lr in itertools.product(
        ["simple", "extended"],
        [0, 16],
        [5.0, 10.0, 20.0, 50.0, 100.0],
        [1e-3, 3e-3],
    ):
        configs.append(ModelConfig(feature_key, hidden, pos_weight, lr, 1e-4, 100))

    outer_results = []
    for outer_idx, holdout in enumerate(sorted(set(groups.tolist()))):
        train_mask = groups != holdout
        test_mask = ~train_mask
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        fold_id = np.asarray([movie_fold(x, args.folds) for x in datasets], dtype=np.int64)
        candidates = []

        for cfg_idx, cfg in enumerate(configs):
            xall = features[cfg.feature_key]
            oof = np.full(len(train_idx), np.nan, dtype=np.float32)
            outer_preds = []
            for fold in range(args.folds):
                fit_global = train_idx[fold_id[train_idx] != fold]
                val_global = train_idx[fold_id[train_idx] == fold]
                if len(val_global) == 0 or int(y[fit_global].sum()) == 0:
                    continue
                x_fit, [x_val, x_test] = standardize(
                    xall[fit_global], [xall[val_global], xall[test_idx]]
                )
                model = fit_model(
                    x_fit,
                    y[fit_global],
                    cfg,
                    args.device,
                    args.seed + outer_idx * 100000 + cfg_idx * 100 + fold,
                )
                val_prob = predict(model, x_val, args.device)
                pos = np.searchsorted(train_idx, val_global)
                oof[pos] = val_prob
                outer_preds.append(predict(model, x_test, args.device))

            valid = np.isfinite(oof)
            if not valid.any() or not outer_preds:
                continue
            threshold, inner_stats = choose_threshold(
                oof[valid],
                base_correct[train_idx][valid],
                top2_correct[train_idx][valid],
            )
            outer_prob = np.mean(np.stack(outer_preds), axis=0)
            outer_stats = apply_stats(
                outer_prob,
                threshold,
                base_correct[test_idx],
                top2_correct[test_idx],
            )
            row = {
                "config": cfg.__dict__,
                "inner": inner_stats,
                "outer": outer_stats,
            }
            score = (
                inner_stats["net"],
                inner_stats["fixes"],
                -inner_stats["breaks"],
                -inner_stats["swaps"],
            )
            candidates.append((score, row))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1] if candidates else None
        outer_results.append(
            {
                "holdout": holdout,
                "train_rows": int(train_mask.sum()),
                "train_swap_positive": int(y[train_mask].sum()),
                "test_rows": int(test_mask.sum()),
                "test_swap_positive": int(y[test_mask].sum()),
                "selected": selected,
                "top_configs": [x[1] for x in candidates[:10]],
            }
        )

    result = {
        "folds": args.folds,
        "device": args.device,
        "outer_results": outer_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
