#!/usr/bin/env python3
"""How well can the organiser's `estimated_number_of_nodes` (T_true) be predicted from what a test movie gives us?

Features per movie (97 local C012-configuration runs): ILP node count, low-detection peak counts above several
score thresholds (from the edge_cache dump), mean peaks per frame. Log-space least squares per embryo prefix with
leave-one-out prediction; reports residual spread for T_pred-only vs the best 1-2 extra features.

    python src/node_budget_ttrue_fit.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = [REPO / "experiments/candidates/c012_v1284_head/e2e/head_v1", REPO / "experiments/candidates/c012_v1284_head/e2e_confirm/head_v1"] + \
       [REPO / f"experiments/candidates/c016_division_scorer/e2e/head_v1_b{b:02d}" for b in range(3)]
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def loo_fit(X, y):
    """Leave-one-out OLS predictions (X with intercept column)."""
    n = len(y); pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred[i] = X[i] @ beta
    return pred


def main() -> int:
    rows = []
    for run in RUNS:
        for pred in sorted((run / "predictions").glob("*.geff")):
            stem = pred.stem
            gt = REPO / "data/train" / f"{stem}.geff"
            meta = json.loads((gt / "zarr.json").read_text(encoding="utf-8")); geff = meta["attributes"].get("geff", meta["attributes"])
            t_true = geff.get("extra", {}).get("estimated_number_of_nodes")
            n_pred = int(json.loads((pred / "nodes" / "ids" / "zarr.json").read_text())["shape"][0])
            dump = np.load(run / "edge_cache" / f"{stem}.npz")
            ls = np.asarray(dump["low_score"], dtype=np.float64).reshape(-1)
            counts = {f"peaks>={thr}": int((ls >= thr).sum()) for thr in THRESHOLDS}
            rows.append({"stem": stem, "prefix": stem[:4], "t_true": t_true, "t_pred": n_pred, **counts})
    print(f"{len(rows)} movies")
    for prefix in ("6bba", "44b6"):
        sub = [r for r in rows if r["prefix"] == prefix]
        y = np.log([r["t_true"] for r in sub])
        feats = ["t_pred"] + [f"peaks>={thr}" for thr in THRESHOLDS]
        F = {f: np.log(np.maximum([r[f] for r in sub], 1)) for f in feats}
        print(f"\n== {prefix}: n={len(sub)}")
        results = []
        for k in (1, 2):
            for combo in itertools.combinations(feats, k):
                X = np.column_stack([np.ones(len(sub))] + [F[f] for f in combo])
                pred = loo_fit(X, y); resid = y - pred
                results.append((float(np.sqrt(np.mean(resid ** 2))), combo, resid))
        results.sort(key=lambda r: r[0])
        base = next(r for r in results if r[1] == ("t_pred",))
        print(f"  T_pred only: LOO residual rms {base[0]:.3f} (~{100 * (np.exp(base[0]) - 1):.1f} %), max |resid| {np.abs(base[2]).max():.3f}")
        for rms, combo, resid in results[:6]:
            print(f"  {'+'.join(combo):28s} LOO rms {rms:.3f} (~{100 * (np.exp(rms) - 1):.1f} %), max |resid| {np.abs(resid).max():.3f}, "
                  f"share |resid|>0.1: {100 * np.mean(np.abs(resid) > 0.1):.0f} %")
        # what the best model says about the over-predicting movies
        rms, combo, resid = results[0]
        X = np.column_stack([np.ones(len(sub))] + [F[f] for f in combo]); pred = loo_fit(X, y)
        print("  movies where T_pred exceeds the LOO-predicted T_true by > 10 %:")
        for r, p in sorted(zip(sub, pred), key=lambda rp: -rp[0]["t_pred"] / np.exp(rp[1])):
            ratio = r["t_pred"] / np.exp(p)
            if ratio > 1.10:
                print(f"    {r['stem']}: T_pred {r['t_pred']:6d}, predicted T_true {np.exp(p):8.0f} (ratio {ratio:.2f}), actual T_true {r['t_true']:6d} (actual ratio {r['t_pred'] / r['t_true']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
