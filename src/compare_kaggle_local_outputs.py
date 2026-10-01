#!/usr/bin/env python3
"""Compare a Kaggle run's low-detection dumps with a local run of the same notebook code.

x138-lineage kernels write ``/kaggle/working/edge_cache/<movie>.npz`` (final detector
coordinates, admitted edges with probabilities, sub-threshold peaks); so does
src/run_kaggle_predict_local.py. Matching detections by position (per frame, nearest
neighbour) makes the comparison robust to the handful of borderline peaks that flip
between GPUs, and reports:

  * detections: counts and the fraction matched within --tol-um,
  * coordinate differences of matched detections (um) - with a V1284 head these are the
    refined centres, so this checks that the head behaves the same on the T4,
  * edge-probability differences on edges whose both endpoints matched.

    python src/compare_kaggle_local_outputs.py --kaggle tmp/c012_output/edge_cache \
        --local experiments/candidates/c012_v1284_head/e2e/head_v1/edge_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

VOXEL_UM = np.array([1.625, 0.40625, 0.40625])


def match(k_coords: np.ndarray, l_coords: np.ndarray, tol_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Index into l for every k detection (-1 if none within tol) and the distance in um."""
    idx = np.full(len(k_coords), -1, dtype=np.int64)
    dist = np.full(len(k_coords), np.inf)
    for t in np.unique(k_coords[:, 0]):
        ks = np.nonzero(k_coords[:, 0] == t)[0]
        ls = np.nonzero(l_coords[:, 0] == t)[0]
        if not len(ls):
            continue
        d, j = cKDTree(l_coords[ls, 1:] * VOXEL_UM).query(k_coords[ks, 1:] * VOXEL_UM, k=1)
        ok = d <= tol_um
        idx[ks[ok]] = ls[j[ok]]
        dist[ks] = d
    return idx, dist


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--tol-um", type=float, default=0.05)
    args = parser.parse_args()

    all_d, all_p = [], []
    for kp in sorted(args.kaggle.glob("*.npz")):
        lp = args.local / kp.name
        if not lp.exists():
            print(f"{kp.stem}: no local dump")
            continue
        k, l = np.load(kp), np.load(lp)
        kc, lc = k["coords"].astype(np.float64), l["coords"].astype(np.float64)
        idx, dist = match(kc, lc, args.tol_um)
        matched = idx >= 0
        all_d.append(dist[matched])
        k_edges = {(int(a), int(b)): p for a, b, p, _ in k["admitted"]}
        l_edges = {(int(a), int(b)): p for a, b, p, _ in l["admitted"]}
        diffs = [abs(p - l_edges[(idx[a], idx[b])]) for (a, b), p in k_edges.items()
                 if idx[a] >= 0 and idx[b] >= 0 and (idx[a], idx[b]) in l_edges]
        all_p.append(np.array(diffs))
        print(f"{kp.stem}: detections kaggle {len(kc)} local {len(lc)} matched {matched.mean():.5f}; "
              f"coord |diff| median {np.median(dist[matched]):.1e} um, p99 {np.percentile(dist[matched], 99):.1e}, "
              f"max {dist[matched].max():.1e}; edges compared {len(diffs)}/{len(k_edges)}, "
              f"prob |diff| median {np.median(diffs):.1e} p99 {np.percentile(diffs, 99):.1e}")
    if all_d:
        d, p = np.concatenate(all_d), np.concatenate(all_p)
        print(f"ALL: coord |diff| median {np.median(d):.2e} um, p99 {np.percentile(d, 99):.2e} um; "
              f"edge prob |diff| median {np.median(p):.2e}, p99 {np.percentile(p, 99):.2e}, frac>0.01 {(p > 0.01).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
