#!/usr/bin/env python3
"""Compare V1284 heads (and their average) on capture pairs: mean |GT - refined centre| per movie and pooled.

    python src/compare_v1284_heads_on_pairs.py --pairs-dir <dir> --stems-file <stems> --head ours=<pt> --head x138=<pt>
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tmp/c011_output/tracking_repo/scripts"))
from v1284_coordinate_refinement import make_head, bounded  # noqa: E402


def load(p):
    s = torch.load(p, map_location="cpu", weights_only=True); h = make_head(); h.load_state_dict(s["state_dict"]); h.eval()
    return h, s["mean"].float(), s["scale"].float()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs-dir", type=Path, required=True); ap.add_argument("--stems-file", type=Path, required=True)
    ap.add_argument("--head", action="append", default=[], metavar="NAME=PATH"); ap.add_argument("--label", default="set")
    a = ap.parse_args()
    heads = {n: load(p) for n, p in (h.split("=", 1) for h in a.head)}
    names = list(heads); combos = names + (["avg(" + "+".join(names) + ")"] if len(names) > 1 else [])
    stems = [s for s in a.stems_file.read_text().split() if (a.pairs_dir / f"{s}.npz").exists()]
    print(f"{a.label:8s} {'movie':14s} {'n':>5s} {'base':>6s} " + " ".join(f"{c[:10]:>10s}" for c in combos) + "   mean |GT - centre| um (lower is better)")
    sums = {c: 0.0 for c in ["base"] + combos}; n_all = 0; wins = {c: 0 for c in combos}
    for stem in stems:
        d = np.load(a.pairs_dir / f"{stem}.npz"); x = torch.from_numpy(d["features"].astype(np.float32)); off = d["offset_um"].astype(np.float32)
        if len(off) == 0: continue
        with torch.no_grad():
            sh = {n: bounded(h, (x - m) / s).numpy() for n, (h, m, s) in heads.items()}
        if len(names) > 1: sh[combos[-1]] = sum(sh[n] for n in names) / len(names)
        err = {"base": np.linalg.norm(off, axis=1)}; err.update({c: np.linalg.norm(off - sh[c], axis=1) for c in combos})
        for c in err: sums[c] += err[c].sum()
        n_all += len(off)
        best = min(combos, key=lambda c: err[c].mean()); wins[best] += 1
        print(f"{a.label:8s} {stem:14s} {len(off):5d} {err['base'].mean():6.3f} " + " ".join(f"{err[c].mean():10.3f}" for c in combos))
    print(f"{a.label:8s} {'POOLED':14s} {n_all:5d} {sums['base'] / n_all:6.3f} " + " ".join(f"{sums[c] / n_all:10.3f}" for c in combos) +
          "   vs base: " + ", ".join(f"{c} {100 * (sums[c] / sums['base'] - 1):+.1f}%" for c in combos) + " | per-movie wins: " + str(wins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
