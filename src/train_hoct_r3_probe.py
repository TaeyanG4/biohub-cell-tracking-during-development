from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def fit_head(x, y, w0, b0, epochs=20, lr=3e-3, anchor=5e-3, batch_size=16384, device="cuda"):
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(w0, dtype=torch.float32, device=device).view(1, -1))
        head.bias.copy_(torch.as_tensor([b0], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    pos = max(1.0, float((y == 0).sum()) / max(1.0, float((y == 1).sum())))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos], device=device))
    w_ref = head.weight.detach().clone(); b_ref = head.bias.detach().clone()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx].to(device, non_blocking=True)
            yb = y[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            # Penalize the total L2 drift from the pretrained HOCT head. Using
            # mean() here made the effective constraint 288x weaker and allowed
            # the probe to move far outside the pretrained head's scale.
            loss = loss + anchor * ((head.weight - w_ref).square().sum() + (head.bias - b_ref).square().sum())
            loss.backward(); opt.step()
    return head.cpu()


def metrics(y, s):
    out = {"ap": float(average_precision_score(y, s))}
    out["auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="features, labels, groups arrays")
    ap.add_argument("--hoct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--anchor", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x = np.asarray(d["features"], np.float32)
    y = np.asarray(d["labels"], np.int64)
    groups = np.asarray(d["groups"]).astype(str)
    if x.ndim != 2 or x.shape[1] != 288:
        raise ValueError(f"expected N x 288 normalized HOCT features, got {x.shape}")

    m = torch.jit.load(args.hoct, map_location="cpu").eval()
    w0 = m.head.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(m.head.bias.detach().cpu().numpy().reshape(-1)[0])
    base = x @ w0 + b0

    rows = []
    for g in sorted(set(groups.tolist())):
        tr = groups != g; va = groups == g
        if tr.sum() == 0 or va.sum() == 0:
            continue
        h = fit_head(x[tr], y[tr], w0, b0, args.epochs, args.lr, args.anchor, args.batch_size, args.device)
        with torch.no_grad():
            pred = h(torch.from_numpy(x[va])).squeeze(1).numpy()
        row = {"holdout": g, "n_train": int(tr.sum()), "n_val": int(va.sum())}
        row.update({f"base_{k}": v for k, v in metrics(y[va], base[va]).items()})
        row.update({f"probe_{k}": v for k, v in metrics(y[va], pred).items()})
        row["weight_l2_drift"] = float(np.linalg.norm(h.weight.detach().numpy().reshape(-1) - w0))
        rows.append(row)

    final = fit_head(x, y, w0, b0, args.epochs, args.lr, args.anchor, args.batch_size, args.device)
    out = Path(args.out)
    if out.suffix.lower() != ".pt":
        out = out.with_suffix(".pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": final.weight.detach().cpu(), "bias": final.bias.detach().cpu()}, out)
    report = out.with_suffix(".json")
    report.write_text(json.dumps({"folds": rows, "n": int(len(y)), "groups": sorted(set(groups.tolist()))}, indent=2), encoding="utf-8")
    print(json.dumps({"folds": rows, "saved": str(out)}, indent=2))


if __name__ == "__main__":
    main()
