from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_r3_target_ranker import aggregate_unique, build_pairs, fit_ranker


def target_table(meta: pd.DataFrame, base: np.ndarray, probe: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    m = meta.loc[mask].copy()
    idx = np.flatnonzero(mask)
    m["base"] = base[idx]
    m["probe"] = probe[idx]
    rows = []
    for (dataset, target), g in m.groupby(["dataset", "target_id"], sort=False):
        if g.label.sum() <= 0 or len(g) < 2:
            continue
        gb = g.sort_values("base", ascending=False).reset_index(drop=True)
        gp = g.sort_values("probe", ascending=False).reset_index(drop=True)
        base_top = gb.iloc[0]
        probe_top = gp.iloc[0]
        base_margin = float(gb.iloc[0].base - gb.iloc[1].base)
        probe_margin = float(gp.iloc[0].probe - gp.iloc[1].probe)
        rows.append(
            {
                "dataset": dataset,
                "target_id": int(target),
                "base_correct": bool(base_top.label > 0),
                "probe_correct": bool(probe_top.label > 0),
                "winner_changed": int(base_top.edge_id) != int(probe_top.edge_id),
                "base_margin": base_margin,
                "probe_margin": probe_margin,
                "margin_gain": probe_margin - base_margin,
            }
        )
    return pd.DataFrame(rows)


def gate_stats(t: pd.DataFrame, threshold: float) -> dict:
    use_probe = t.base_margin <= threshold
    correct = np.where(use_probe, t.probe_correct.to_numpy(), t.base_correct.to_numpy())
    return {
        "threshold": float(threshold),
        "n_targets": int(len(t)),
        "n_probe": int(use_probe.sum()),
        "acc": float(correct.mean()) if len(correct) else float("nan"),
        "fixes": int(((~t.base_correct) & t.probe_correct & use_probe).sum()),
        "breaks": int((t.base_correct & (~t.probe_correct) & use_probe).sum()),
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
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    m = torch.jit.load(args.hoct, map_location="cpu").eval()
    w0 = m.head.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(m.head.bias.detach().cpu().numpy().reshape(-1)[0])
    meta, feat = aggregate_unique(d, w0, b0)
    base = feat @ w0 + b0

    result = []
    for holdout in sorted(meta.group.unique()):
        tr = meta.group.to_numpy() != holdout
        va = ~tr
        p, n, _ = build_pairs(meta, feat, tr, hard_k=args.hard_k)
        h = fit_ranker(feat, p, n, w0, b0, args.epochs, args.lr, args.anchor, "cuda")
        with torch.no_grad():
            probe = h(torch.from_numpy(feat)).squeeze(1).numpy()
        t = target_table(meta, base, probe, va)
        changed = t[t.winner_changed]
        quantiles = np.unique(np.quantile(t.base_margin.to_numpy(), [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]))
        gates = [gate_stats(t, float(q)) for q in quantiles]
        blends = []
        for alpha in np.linspace(0.0, 1.0, 11):
            blended = (1.0 - alpha) * base + alpha * probe
            bt = target_table(meta, base, blended, va)
            blends.append(
                {
                    "alpha": float(alpha),
                    "acc": float(bt.probe_correct.mean()),
                    "winner_changes": int(bt.winner_changed.sum()),
                    "fixes": int(((~bt.base_correct) & bt.probe_correct).sum()),
                    "breaks": int((bt.base_correct & (~bt.probe_correct)).sum()),
                }
            )
        result.append(
            {
                "holdout": holdout,
                "n_targets": int(len(t)),
                "base_acc": float(t.base_correct.mean()),
                "probe_acc": float(t.probe_correct.mean()),
                "winner_changes": int(t.winner_changed.sum()),
                "changed_fixes": int(((~changed.base_correct) & changed.probe_correct).sum()),
                "changed_breaks": int((changed.base_correct & (~changed.probe_correct)).sum()),
                "changed_both_correct": int((changed.base_correct & changed.probe_correct).sum()),
                "changed_examples": changed[
                    [
                        "dataset",
                        "target_id",
                        "base_correct",
                        "probe_correct",
                        "base_margin",
                        "probe_margin",
                        "margin_gain",
                    ]
                ].to_dict(orient="records"),
                "gates": gates,
                "blends": blends,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"folds": result}, indent=2), encoding="utf-8")
    print(json.dumps({"folds": result}, indent=2))


if __name__ == "__main__":
    main()
