#!/usr/bin/env python3
"""Train / cross-validate the C016 division scorer on candidate triples.

Input: CSVs from src/division_candidates_local.py. Only labelled rows (label 0/1, i.e.
the parent matches an annotated GT node) are used. The model is a small torch MLP on
standardised features with class-weighted BCE, saved as {'state_dict', 'mean', 'scale',
'features'} so the notebook can rebuild it with plain torch (no sklearn).

Movie-grouped K-fold CV (folds interleaved within each embryo prefix). Reported per fold
and pooled: average precision, precision/recall at the chosen threshold, and - the number
that matters for the metric - how many labelled positives are recovered at how many
labelled negatives accepted (annotated false divisions), compared with x138's rule
(candidates that would pass parent <= 9 um, sister <= 14 um, symmetry <= 0.6).

    python src/division_scorer_train.py --candidates a.csv b.csv --out heads/division_scorer_v1.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from division_scorer_stage import FEATURES, make_model  # noqa: E402


def load(paths: list[Path], positive_kinds: set[str] | None = None):
    """Labelled rows. With positive_kinds, positives of other kinds are dropped (neither clean positives nor
    negatives); files without a label_kind column keep every positive."""
    rows = []
    for p in paths:
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if int(r["label"]) < 0:
                continue
            if positive_kinds and r["label"] == "1" and r.get("label_kind", "") and r["label_kind"] not in positive_kinds:
                continue
            rows.append(r)
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([int(r["label"]) for r in rows], dtype=np.float32)
    stems = np.array([r["stem"] for r in rows])
    rule = np.array([float(r["dist_p_d1"]) <= 9.0 and float(r["dist_p_d2"]) <= 9.0 and float(r["sister"]) <= 14.0
                     and float(r["asym"]) <= 0.6 for r in rows])
    parents = np.array([f"{r['stem']}:{r['P']}" for r in rows])
    kinds_col = np.array([r.get("label_kind", "") for r in rows])
    return x, y, stems, rule, parents, kinds_col


def per_parent(y, s, parents, thresholds, kinds=None):
    """What the stage would do: one candidate per parent (its best score). TP = that candidate is a positive,
    FP = it is a negative, FN = parents that have a positive candidate but were not recovered.
    Returns (thr, tp, fp, fn, fp_by_kind) rows."""
    best: dict[str, int] = {}
    has_pos: set[str] = set()
    for i, p in enumerate(parents):
        if p not in best or s[i] > s[best[p]]:
            best[p] = i
        if y[i] == 1:
            has_pos.add(p)
    idx = np.array(list(best.values()))
    rows = []
    for thr in thresholds:
        acc = s[idx] >= thr
        tp = int((acc & (y[idx] == 1)).sum()); fp_mask = acc & (y[idx] == 0); fp = int(fp_mask.sum())
        by_kind = {}
        if kinds is not None:
            for k in kinds[idx][fp_mask]:
                by_kind[k or "?"] = by_kind.get(k or "?", 0) + 1
        rows.append((thr, tp, fp, len(has_pos) - tp, by_kind))
    return rows


def fit(x, y, epochs, lr, wd, seed, pos_weight, hidden):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / scale); yt = torch.from_numpy(y)
    model = make_model(x.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            idx = perm[i:i + 512]
            loss = loss_fn(model(xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, mean, scale


def predict(model, mean, scale, x):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy((x - mean) / scale)).squeeze(1)).numpy()


def average_precision(y, s):
    order = np.argsort(-s); y = y[order]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def report(y, s, rule, parents, thresholds, title, kinds=None):
    n_pos = int(y.sum())
    print(f"[{title}] labelled {len(y)}, positives {n_pos}, average precision {average_precision(y, s):.3f}")
    print(f"{'threshold':>9s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'precision':>9s} {'recall':>7s}  divJ if these were all the divisions")
    for thr in thresholds:
        acc = s >= thr
        tp = int((acc & (y == 1)).sum()); fp = int((acc & (y == 0)).sum()); fn = n_pos - tp
        print(f"{thr:9.2f} {tp:4d} {fp:4d} {fn:4d} {tp / max(tp + fp, 1):9.3f} {tp / max(n_pos, 1):7.3f}  {tp / max(tp + fp + fn, 1):.3f}")
    tp_r = int((rule & (y == 1)).sum()); fp_r = int((rule & (y == 0)).sum())
    print(f"{'geom rule':>9s} {tp_r:4d} {fp_r:4d} {n_pos - tp_r:4d} {tp_r / max(tp_r + fp_r, 1):9.3f} {tp_r / max(n_pos, 1):7.3f}  "
          f"{tp_r / max(tp_r + fp_r + n_pos - tp_r, 1):.3f}   (x138 geometry only: parent <= 9, sister <= 14, asym <= 0.6; x138 also requires an orphan D2)")
    print("  per parent (best candidate per P, as the stage accepts them):")
    print(f"{'threshold':>9s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'precision':>9s} {'recall':>7s}  FP by label kind")
    for thr, tp, fp, fn, by_kind in per_parent(y, s, parents, thresholds, kinds):
        print(f"{thr:9.2f} {tp:4d} {fp:4d} {fn:4d} {tp / max(tp + fp, 1):9.3f} {tp / max(tp + fn, 1):7.3f}  {by_kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--eval", nargs="*", type=Path, default=[], help="candidate CSVs never trained on; scored by the final model")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dump-scores", type=Path, default=None, help="CSV of OOF / held-out candidate scores for error analysis")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=None, help="default: n_neg / n_pos")
    parser.add_argument("--max-pos-weight", type=float, default=None, help="cap on the class weight (huge weights destabilise training)")
    parser.add_argument("--hidden", type=int, default=32, help="MLP width; 0 = logistic regression")
    parser.add_argument("--positive-kinds", default="", help="comma list of label kinds kept as training positives (strict,early,grand); default all")
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (the candidate generators share the machine)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    kinds = {k for k in args.positive_kinds.split(",") if k} or None
    x, y, stems, rule, parents, kind_col = load(args.candidates, kinds)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    pos_weight = args.pos_weight or n_neg / max(n_pos, 1)
    if args.max_pos_weight:
        pos_weight = min(pos_weight, args.max_pos_weight)
    print(f"labelled candidates {len(y)}: positives {n_pos}, negatives {n_neg}, movies {len(set(stems))}; "
          f"x138 rule would accept {int(rule[y == 1].sum())} positives and {int(rule[y == 0].sum())} negatives")
    uniq = sorted(set(stems)); fold_of = {}
    for prefix in sorted({s.split('_')[0] for s in uniq}):
        for i, s in enumerate([s for s in uniq if s.startswith(prefix + "_")]):
            fold_of[s] = i % args.folds
    oof = np.zeros(len(y), dtype=np.float32)
    for fold in range(args.folds):
        te = np.array([fold_of[s] == fold for s in stems])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        model, mean, scale = fit(x[~te], y[~te], args.epochs, args.lr, args.weight_decay, args.seed, pos_weight, args.hidden)
        oof[te] = predict(model, mean, scale, x[te])
    thresholds = [float(t) for t in args.thresholds.split(",")]
    report(y, oof, rule, parents, thresholds, f"movie-grouped {args.folds}-fold OOF, hidden={args.hidden}, pos_weight={pos_weight:.1f}", kind_col)
    if args.dump_scores:
        args.dump_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_scores.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["set", "parent", "label", "label_kind", "score"])
            w.writerows(zip(["oof"] * len(y), parents, y.astype(int), kind_col, np.round(oof, 5)))

    if args.out or args.eval:
        model, mean, scale = fit(x, y, args.epochs, args.lr, args.weight_decay, args.seed, pos_weight, args.hidden)
    for path in args.eval:
        xe, ye, se, re_, pe, ke = load([path])
        s_e = predict(model, mean, scale, xe)
        report(ye, s_e, re_, pe, thresholds, f"held-out {path.name} ({len(set(se))} movies, model trained on all --candidates)", ke)
        if args.dump_scores:
            with args.dump_scores.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(zip([path.stem] * len(ye), pe, ye.astype(int), ke, np.round(s_e, 5)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                    "features": FEATURES, "hidden": args.hidden}, args.out)
        args.out.with_suffix(".json").write_text(json.dumps({"n": len(y), "positives": n_pos, "negatives": n_neg,
                                                            "movies": uniq, "pos_weight": pos_weight, "epochs": args.epochs,
                                                            "oof_average_precision": average_precision(y, oof)}, indent=2))
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
