#!/usr/bin/env python3
"""Train and cross-validate a V1284-compatible coordinate head on captured pairs.

Input: ``<pairs>/<stem>.npz`` from src/v1284_capture_local.py (224-d frozen features
at fused detections, GT-minus-detection offset in um). The model is exactly the
architecture x138's ``v1284_coordinate_refinement.make_head`` builds
(Linear(224, 32) -> SiLU -> Linear(32, 3), zero-initialised output layer) with its
``bounded`` output (norm < 2 um), trained on standardised features, so the saved
``{'state_dict', 'mean', 'scale'}`` file loads unchanged in V1284_MODE='candidate'.

Cross-validation is grouped by movie (every movie is held out once). Reported per
fold and overall: mean / median distance to GT before and after the shift, the
fraction of pairs moved closer, and the same split by embryo prefix.

    python src/v1284_head_train.py --pairs experiments/candidates/c012_v1284_head/pairs \
        --out experiments/candidates/c012_v1284_head/v1284_head.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

IN_DIM, HIDDEN = 224, 32


def make_head() -> torch.nn.Sequential:
    head = torch.nn.Sequential(torch.nn.Linear(IN_DIM, HIDDEN), torch.nn.SiLU(), torch.nn.Linear(HIDDEN, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def load_pairs(pairs_dir: Path, max_offset_um: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = {}
    for path in sorted(pairs_dir.glob("*.npz")):
        with np.load(path) as z:
            if "features" not in z.files or z["features"].ndim != 2 or not len(z["features"]):
                continue
            x = z["features"].astype(np.float32)
            y = z["offset_um"].astype(np.float32)
        keep = np.linalg.norm(y, axis=1) <= max_offset_um
        if keep.any():
            data[path.stem] = (x[keep], y[keep])
    return data


def fit(x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None, y_val: np.ndarray | None, epochs: int,
        lr: float, weight_decay: float, batch: int, seed: int, device: torch.device,
        sample_weight: np.ndarray | None = None):
    torch.manual_seed(seed)
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6  # never divide by zero: a NaN shift raises inside refine()
    xt = torch.from_numpy((x - mean) / scale).to(device)
    yt = torch.from_numpy(y).to(device)
    wt = torch.from_numpy((sample_weight if sample_weight is not None else np.ones(len(y))).astype(np.float32)).to(device)
    xv = torch.from_numpy((x_val - mean) / scale).to(device) if x_val is not None else None
    yv = torch.from_numpy(y_val).to(device) if y_val is not None else None
    head = make_head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(xt), generator=gen).to(device)
        for start in range(0, len(xt), batch):
            idx = perm[start:start + batch]
            w = wt[idx]
            loss = (((bounded(head, xt[idx]) - yt[idx]) ** 2).sum(dim=1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if xv is not None:
            head.eval()
            with torch.no_grad():
                d_after = torch.linalg.vector_norm(yv - bounded(head, xv), dim=1).mean().item()
            history.append(d_after)
    head.eval()
    return head, mean.astype(np.float32), scale.astype(np.float32), history


def shift_stats(head, mean, scale, x, y, device) -> dict[str, float]:
    with torch.no_grad():
        pred = bounded(head, torch.from_numpy((x - mean) / scale).to(device)).cpu().numpy()
    before = np.linalg.norm(y, axis=1)
    after = np.linalg.norm(y - pred, axis=1)
    return {
        "n": int(len(y)),
        "mean_before": float(before.mean()), "mean_after": float(after.mean()),
        "median_before": float(np.median(before)), "median_after": float(np.median(after)),
        "rel_change": float(after.mean() / before.mean() - 1.0),
        "frac_closer": float((after < before).mean()),
        "mean_shift": float(np.linalg.norm(pred, axis=1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="save the head trained on all movies here")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--train-max-offset-um", type=float, default=3.0,
                        help="drop training pairs farther than this (likely mismatches)")
    parser.add_argument("--eval-max-offset-um", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance-prefix", action="store_true",
                        help="weight pairs so each embryo prefix (44b6 / 6bba) contributes equally to the loss")
    args = parser.parse_args()

    device = torch.device("cpu")
    train_data = load_pairs(args.pairs, args.train_max_offset_um)
    eval_data = load_pairs(args.pairs, args.eval_max_offset_um)
    stems = sorted(set(train_data) & set(eval_data))
    if len(stems) < args.folds:
        raise SystemExit(f"need at least {args.folds} movies, have {len(stems)}")
    n_pairs = sum(len(train_data[s][1]) for s in stems)
    print(f"movies={len(stems)} train pairs={n_pairs} "
          f"(44b6={sum(s.startswith('44b6') for s in stems)}, 6bba={sum(s.startswith('6bba') for s in stems)})")

    # Movie-grouped folds, interleaved within each embryo prefix.
    fold_of = {}
    for prefix in sorted({s.split("_")[0] for s in stems}):
        for i, s in enumerate(sorted(s for s in stems if s.startswith(prefix + "_"))):
            fold_of[s] = i % args.folds

    def weights_for(members: list[str]) -> np.ndarray | None:
        if not args.balance_prefix:
            return None
        prefix = np.concatenate([[s.split("_")[0]] * len(train_data[s][1]) for s in members])
        names, counts = np.unique(prefix, return_counts=True)
        per = {n: len(prefix) / (len(names) * c) for n, c in zip(names, counts)}
        return np.array([per[p] for p in prefix], dtype=np.float32)

    rows, curves = [], []
    for fold in range(args.folds):
        tr = [s for s in stems if fold_of[s] != fold]
        te = [s for s in stems if fold_of[s] == fold]
        if not te:
            continue
        x = np.concatenate([train_data[s][0] for s in tr]); y = np.concatenate([train_data[s][1] for s in tr])
        xv = np.concatenate([eval_data[s][0] for s in te]); yv = np.concatenate([eval_data[s][1] for s in te])
        head, mean, scale, hist = fit(x, y, xv, yv, args.epochs, args.lr, args.weight_decay, args.batch, args.seed, device,
                                      sample_weight=weights_for(tr))
        curves.append(hist)
        for group, members in (("all", te), ("44b6", [s for s in te if s.startswith("44b6")]),
                               ("6bba", [s for s in te if s.startswith("6bba")])):
            if not members:
                continue
            gx = np.concatenate([eval_data[s][0] for s in members]); gy = np.concatenate([eval_data[s][1] for s in members])
            stats = shift_stats(head, mean, scale, gx, gy, device)
            stats.update(fold=fold, group=group, movies=len(members))
            rows.append(stats)
        s = [r for r in rows if r["fold"] == fold and r["group"] == "all"][0]
        print(f"fold {fold}: movies={len(te)} pairs={s['n']} mean {s['mean_before']:.3f} -> {s['mean_after']:.3f} um "
              f"({100 * s['rel_change']:+.1f}%) closer={100 * s['frac_closer']:.1f}% shift={s['mean_shift']:.3f}um", flush=True)

    for group in ("all", "44b6", "6bba"):
        sel = [r for r in rows if r["group"] == group]
        if not sel:
            continue
        n = sum(r["n"] for r in sel)
        before = sum(r["mean_before"] * r["n"] for r in sel) / n
        after = sum(r["mean_after"] * r["n"] for r in sel) / n
        closer = sum(r["frac_closer"] * r["n"] for r in sel) / n
        worse_folds = sum(r["mean_after"] > r["mean_before"] for r in sel)
        print(f"CV {group:4s}: pairs={n} mean {before:.3f} -> {after:.3f} um ({100 * (after / before - 1):+.1f}%) "
              f"closer={100 * closer:.1f}% folds_worse={worse_folds}/{len(sel)}")
    mean_curve = np.mean(np.array(curves), axis=0)
    best_epoch = int(np.argmin(mean_curve)) + 1
    print(f"held-out mean distance by epoch (fold average): best epoch {best_epoch} "
          f"({mean_curve[best_epoch - 1]:.4f} um), last {mean_curve[-1]:.4f} um")

    if args.out:
        x = np.concatenate([train_data[s][0] for s in stems]); y = np.concatenate([train_data[s][1] for s in stems])
        head, mean, scale, _ = fit(x, y, None, None, args.epochs, args.lr, args.weight_decay, args.batch, args.seed, device,
                                   sample_weight=weights_for(stems))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)}, args.out)
        report = {"movies": stems, "train_pairs": int(len(y)), "epochs": args.epochs, "lr": args.lr,
                  "weight_decay": args.weight_decay, "train_max_offset_um": args.train_max_offset_um,
                  "balance_prefix": args.balance_prefix, "cv": rows}
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved {args.out} (+ .json report)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
