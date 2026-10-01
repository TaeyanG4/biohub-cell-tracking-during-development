#!/usr/bin/env python3
"""Appearance-based division classifier on image crops (HANDOFF section 24): movie-grouped CV + final model.

Input: <crops>/<stem>.npz from src/division_crops_extract.py (crops (N, 3, 8, 32, 32), label, kind).
A small 3D CNN (3 frames as channels) is trained with class-weighted BCE and flip augmentation; 5-fold CV
grouped by movie reports out-of-fold precision / recall per threshold for the division class (the decisive
number: precision at recall ~0.5). --hard <csv> adds the pipeline's false forks (C016 candidate rows with
label 0) as extra negatives via their P coordinates when crops for them exist in <crops-hard>.

    python src/division_cnn_train.py --crops experiments/candidates/c031_division_cnn/crops --out experiments/candidates/c031_division_cnn/division_cnn.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]


class DivisionCNN(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 16):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.SiLU(), nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.SiLU())
        self.b1 = block(in_ch, width)          # (8, 32, 32)
        self.b2 = block(width, width * 2)      # (4, 16, 16)
        self.b3 = block(width * 2, width * 4)  # (2, 8, 8)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width * 4, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.b1(x)
        x = F.max_pool3d(x, (2, 2, 2))
        x = self.b2(x)
        x = F.max_pool3d(x, (2, 2, 2))
        x = self.b3(x)
        x = x.mean(dim=(2, 3, 4))
        return self.head(x).squeeze(-1)


def load_crops(crops_dir: Path):
    data = {}
    for p in sorted(crops_dir.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        data[p.stem] = (d["crops"].astype(np.float32), d["label"].astype(np.int64), d["kind"].astype(str))
    return data


def augment(x: torch.Tensor) -> torch.Tensor:
    # random flips in y / x and transpose (the embryo has no preferred lateral orientation); z kept
    if torch.rand(1).item() < 0.5: x = x.flip(-1)
    if torch.rand(1).item() < 0.5: x = x.flip(-2)
    if torch.rand(1).item() < 0.5: x = x.transpose(-1, -2)
    return x


def fit(x, y, epochs, lr, device, seed, pos_weight):
    torch.manual_seed(seed)
    model = DivisionCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).float()
    n = len(yt); bs = 64
    pw = torch.tensor([pos_weight], device=device)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = augment(xt[idx]).to(device); yb = yt[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, x, device):
    model.eval(); out = []
    for i in range(0, len(x), 256):
        out.append(torch.sigmoid(model(torch.from_numpy(x[i:i + 256]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def report(y, p, kinds, label):
    order = np.argsort(-p)
    print(f"{label}: n={len(y)} positives={int(y.sum())}")
    print(f"  {'thr':>5s} {'prec':>6s} {'recall':>7s} {'TP':>4s} {'FP':>4s}  FP by kind")
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        sel = p >= thr; tp = int((sel & (y == 1)).sum()); fp = int((sel & (y == 0)).sum())
        kinds_fp = {k: int((sel & (y == 0) & (kinds == k)).sum()) for k in np.unique(kinds[y == 0])}
        print(f"  {thr:5.2f} {tp / max(tp + fp, 1):6.3f} {tp / max(int(y.sum()), 1):7.3f} {tp:4d} {fp:4d}  {kinds_fp}")
    # precision at recall ~0.5
    k = max(int(round(0.5 * y.sum())), 1); top = order[:k]
    hits = int(y[top].sum()); thr_at = float(p[top[-1]])
    print(f"  precision at recall 0.5 (top {k} by score): {hits / k:.3f} (threshold {thr_at:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_crops(args.crops)
    stems = sorted(data)
    print(f"movies {len(stems)}, samples {sum(len(v[1]) for v in data.values())}, positives {sum(int(v[1].sum()) for v in data.values())}, device {device}")
    fold_of = {}
    for prefix in sorted({s.split('_')[0] for s in stems}):
        for i, s in enumerate(sorted(s for s in stems if s.startswith(prefix + '_'))):
            fold_of[s] = i % args.folds
    oof_y, oof_p, oof_k = [], [], []
    for fold in range(args.folds):
        tr = [s for s in stems if fold_of[s] != fold]; te = [s for s in stems if fold_of[s] == fold]
        x = np.concatenate([data[s][0] for s in tr]); y = np.concatenate([data[s][1] for s in tr])
        model = fit(x, y, args.epochs, args.lr, device, args.seed + fold, args.pos_weight)
        xv = np.concatenate([data[s][0] for s in te]); yv = np.concatenate([data[s][1] for s in te]); kv = np.concatenate([data[s][2] for s in te])
        pv = predict(model, xv, device)
        oof_y.append(yv); oof_p.append(pv); oof_k.append(kv)
        print(f"fold {fold}: train {len(y)} (pos {int(y.sum())}) test {len(yv)} (pos {int(yv.sum())}) AUC-ish top-k hit rate {float(yv[np.argsort(-pv)[:max(int(yv.sum()), 1)]].mean()):.3f}", flush=True)
    y, p, k = np.concatenate(oof_y), np.concatenate(oof_p), np.concatenate(oof_k)
    report(y, p, k, "OOF (movie-grouped)")
    for prefix in ("44b6", "6bba"):
        pass
    if args.out:
        x = np.concatenate([data[s][0] for s in stems]); yy = np.concatenate([data[s][1] for s in stems])
        model = fit(x, yy, args.epochs, args.lr, device, args.seed, args.pos_weight)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "crop": [8, 32, 32], "frames": [-1, 0, 1], "clip": 3.0}, args.out)
        args.out.with_suffix(".json").write_text(json.dumps({"movies": stems, "samples": int(len(yy)), "positives": int(yy.sum()), "epochs": args.epochs, "lr": args.lr, "pos_weight": args.pos_weight}, indent=1))
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
