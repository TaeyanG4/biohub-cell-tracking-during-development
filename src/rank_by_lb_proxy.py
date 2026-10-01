#!/usr/bin/env python3
"""Rank post-processing / head variants by the calibrated local -> LB proxy.

Reads per-movie CSVs written by src/eval_pp_variants_local.py, aggregates the 6bba
subgroup with the notebook's formula (size-weighted adjusted edge Jaccard + 0.1 x
micro division Jaccard) and applies the calibration fitted on six LB anchors
(HANDOFF.md section 18): predicted LB = 0.298 x local_6bba + 0.668 (max residual
0.0004 on the anchors). Also reports how many 6bba movies improve over a reference
variant, because the 6bba subset holds only 6 movies and 11 GT divisions.

    python src/rank_by_lb_proxy.py reports/pp_replay/e2e_head_v1_ppknobs.csv \
        reports/pp_replay/e2e_head_v1_a125_all12.csv --reference c012_as_is
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SLOPE, INTERCEPT = 0.298, 0.668
DIVISION_WEIGHT = 0.1


def aggregate(rows: list[dict]) -> dict:
    w = sum(float(r["weight"]) for r in rows) or 1.0
    adj = sum(float(r["adjusted_edge_jaccard"]) * float(r["weight"]) for r in rows) / w
    tp = sum(int(r["div_tp"]) for r in rows); fp = sum(int(r["div_fp"]) for r in rows); fn = sum(int(r["div_fn"]) for r in rows)
    div = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"adj": adj, "div": div, "tp": tp, "fp": fp, "fn": fn, "total": adj + DIVISION_WEIGHT * div}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--reference", default=None, help="variant label to compare per-movie against")
    parser.add_argument("--prefix", default="6bba")
    args = parser.parse_args()

    by_variant: dict[str, list[dict]] = {}
    for path in args.csvs:
        for r in csv.DictReader(path.open(encoding="utf-8")):
            label = r["config"] if len(args.csvs) == 1 or r["config"] != "x138" else path.stem.replace("e2e_", "").replace("_all12", "")
            by_variant.setdefault(label, []).append(r)
    ref = by_variant.get(args.reference) if args.reference else None
    ref_by_stem = {r["stem"]: r for r in ref} if ref else {}

    results = []
    for label, rows in by_variant.items():
        sub = [r for r in rows if r["stem"].startswith(args.prefix)]
        if not sub:
            continue
        agg = aggregate(sub)
        agg["label"] = label
        agg["pred"] = SLOPE * agg["total"] + INTERCEPT
        agg["all"] = aggregate(rows)["total"]
        if ref_by_stem:
            better = sum(float(r["adjusted_edge_jaccard"]) > float(ref_by_stem[r["stem"]]["adjusted_edge_jaccard"]) + 1e-9 for r in sub if r["stem"] in ref_by_stem)
            worse = sum(float(r["adjusted_edge_jaccard"]) < float(ref_by_stem[r["stem"]]["adjusted_edge_jaccard"]) - 1e-9 for r in sub if r["stem"] in ref_by_stem)
            agg["edge_better"], agg["edge_worse"] = better, worse
        results.append(agg)
    results.sort(key=lambda a: a["pred"], reverse=True)
    print(f"{'variant':22s} {'pred LB':>8s} {args.prefix + ' total':>11s} {'adj-edge':>9s} {'divJ':>6s} div tp/fp/fn  all12   edge better/worse vs ref")
    for a in results:
        cmp = f"{a['edge_better']}/{a['edge_worse']}" if "edge_better" in a else ""
        print(f"{a['label']:22s} {a['pred']:8.4f} {a['total']:11.4f} {a['adj']:9.4f} {a['div']:6.4f} {a['tp']:>3d}/{a['fp']:>2d}/{a['fn']:>2d}   {a['all']:.4f}   {cmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
