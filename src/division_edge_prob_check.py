#!/usr/bin/env python3
"""Does the edge transformer itself see the far daughter of a division?

For every strict positive division candidate on the held-out movies (P matched to the GT divider, D1 and D2 to
its daughters; from candidates_heldout12.csv) read, from a dense edge-probability dump (inference re-run with
BIOHUB_CACHE_EDGE_THRESHOLD=0.02 so `edge_src/edge_tgt/edge_prob` hold every pair above 0.02):
  * p(P->D2): the transformer's probability that P is D2's parent (softmax over D2's parent candidates),
  * the best competing parent Q of D2 and p(Q->D2),
  * p(P->D1) for reference and the rank of P among D2's parents.
If p(P->D2) is typically far below the competitor, the model itself rejects far daughters and only retraining
could change that; if it is close (say >= 0.3 or within 2x of the competitor), inference-time changes or a light
fine-tune have room.

    python src/division_edge_prob_check.py --dense-dir experiments/candidates/c016_division_scorer/e2e/head_v1_dense/edge_cache
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VOX = np.array([1.625, 0.40625, 0.40625])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=REPO / "experiments/candidates/c016_division_scorer/candidates_heldout12.csv")
    parser.add_argument("--kinds", default="strict", help="positive label kinds to inspect")
    args = parser.parse_args()
    kinds = set(args.kinds.split(","))
    rows = [r for r in csv.DictReader(args.candidates.open(encoding="utf-8")) if r["label"] == "1" and r.get("label_kind", "strict") in kinds]
    by_stem = defaultdict(list)
    for r in rows:
        by_stem[r["stem"]].append(r)
    print(f"{len(rows)} positive candidates ({args.kinds}) on {len(by_stem)} movies")
    summary = []
    for stem, cands in sorted(by_stem.items()):
        path = args.dense_dir / f"{stem}.npz"
        if not path.exists():
            print(f"{stem}: no dense dump"); continue
        d = np.load(path)
        src, tgt, prob = d["edge_src"].astype(int), d["edge_tgt"].astype(int), d["edge_prob"].astype(float)
        coords = d["coords"].astype(float)
        parents = defaultdict(list)
        for s, t, p in zip(src, tgt, prob):
            parents[t].append((p, s))
        pair = {(s, t): p for s, t, p in zip(src, tgt, prob)}
        print(f"\n== {stem}: dense pairs {len(prob)} (>0.02), nodes {len(coords)}")
        seen = set()
        for c in cands:
            P, D1, D2 = int(c["P"]), int(c["D1"]), int(c["D2"])
            if (P, D2) in seen:
                continue
            seen.add((P, D2))
            if max(P, D1, D2) >= len(coords):
                print(f"  P{P} D1 {D1} D2 {D2}: synthetic node id (post-processing) - not in the ILP graph, skipped"); continue
            p_d2 = pair.get((P, D2), 0.0); p_d1 = pair.get((P, D1), 0.0)
            comp = sorted(parents.get(D2, []), reverse=True)
            best = [(p, s) for p, s in comp if s != P][:1]
            rank = 1 + sum(1 for p, s in comp if s != P and p > p_d2)
            dist_pd2 = float(np.linalg.norm((coords[P, 1:] - coords[D2, 1:]) * VOX))
            q_txt = ""
            if best:
                q = best[0][1]; dq = float(np.linalg.norm((coords[q, 1:] - coords[D2, 1:]) * VOX))
                q_txt = f"best competitor Q{q}: p={best[0][0]:.3f} at {dq:.1f} um"
            print(f"  P{P} -> D2 {D2} ({dist_pd2:.1f} um): p={p_d2:.3f} (rank {rank} of {len(comp)} parents) | p(P->D1)={p_d1:.3f} | {q_txt}")
            summary.append((p_d2, best[0][0] if best else 0.0, rank, dist_pd2))
    if summary:
        a = np.array(summary)
        print(f"\nSUMMARY over {len(a)} division events: p(P->D2) median {np.median(a[:, 0]):.3f}, >=0.1: {(a[:, 0] >= 0.1).sum()}, >=0.3: {(a[:, 0] >= 0.3).sum()}, "
              f">=0.48 (ILP-admitted): {(a[:, 0] >= 0.48).sum()}; competitor median {np.median(a[:, 1]):.3f}; P ranked first for D2 in {(a[:, 2] == 1).sum()} events; "
              f"ratio p(P->D2)/p(Q->D2) median {np.median(a[:, 0] / np.maximum(a[:, 1], 1e-3)):.2f}; P-D2 distance median {np.median(a[:, 3]):.1f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
