#!/usr/bin/env python3
"""Frozen frames and global jumps in the train movies (forum: hengck23 724283, g john rao 729082).

For every consecutive frame pair of every train movie:
  * frozen: the two uint16 volumes are byte-identical;
  * global shift: 3D phase correlation of the (z, y/2, x/2) volumes, in micrometres;
  * GT motion: median and spread of the annotated edge displacements (t -> t+1) in micrometres, and how
    coherent they are (median |d - median d|).
A "global jump" is a pair whose GT edges move coherently by more than JUMP_UM (most of the displacement is a
whole-field translation). Output: per-pair CSV + summary (how common, how large, whether phase correlation
recovers the GT shift, which movies).

    python src/frame_motion_audit.py --out reports/frame_motion_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VOX = np.array([1.625, 0.40625, 0.40625])
JUMP_UM = 3.0


def read_frame(zarr_path: Path, t: int, shape, dtype):
    import blosc2
    raw = (zarr_path / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes()
    return np.frombuffer(blosc2.decompress(raw), dtype=dtype).reshape(shape[1:])


def phase_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer-voxel translation (z, y, x) that maps a onto b, via normalised cross-power spectrum."""
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    c = np.fft.irfftn(r, s=a.shape)
    idx = np.array(np.unravel_index(np.argmax(c), c.shape), dtype=float)
    size = np.array(a.shape, dtype=float)
    idx[idx > size / 2] -= size[idx > size / 2]
    return idx


def gt_edges(geff: Path):
    import zarr
    g = zarr.open(str(geff), mode="r")
    ids = np.asarray(g["nodes"]["ids"])
    props = {k: np.asarray(g["nodes"]["props"][k]["values"]) for k in ("t", "z", "y", "x")}
    pos = {int(i): (int(props["t"][j]), np.array([props["z"][j], props["y"][j], props["x"][j]], float) * VOX) for j, i in enumerate(ids)}
    e = np.asarray(g["edges"]["ids"])
    by_t: dict[int, list] = {}
    for s, d in e:
        ts, ps = pos[int(s)]; td, pd = pos[int(d)]
        if td == ts + 1:
            by_t.setdefault(ts, []).append(pd - ps)
    return by_t


def audit_movie(stem: str) -> list[dict]:
    zp = REPO / "data" / "train" / f"{stem}.zarr"
    meta = json.loads((zp / "0" / "zarr.json").read_text())
    shape, dtype = tuple(meta["shape"]), np.dtype(meta["data_type"])
    motion = gt_edges(REPO / "data" / "train" / f"{stem}.geff")
    rows = []
    prev = read_frame(zp, 0, shape, dtype)
    for t in range(shape[0] - 1):
        cur = read_frame(zp, t + 1, shape, dtype)
        frozen = bool(np.array_equal(prev, cur))
        a = prev[:, ::2, ::2].astype(np.float32); b = cur[:, ::2, ::2].astype(np.float32)
        a -= a.mean(); b -= b.mean()
        sh = phase_shift(a, b) * np.array([1.0, 2.0, 2.0]) * VOX
        d = np.array(motion.get(t, []))
        row = {"stem": stem, "t": t, "frozen": int(frozen), "pc_dz": sh[0], "pc_dy": sh[1], "pc_dx": sh[2],
               "pc_um": float(np.linalg.norm(sh)), "n_gt_edges": len(d)}
        if len(d):
            med = np.median(d, axis=0)
            row.update(gt_med_um=float(np.linalg.norm(med)), gt_dz=med[0], gt_dy=med[1], gt_dx=med[2],
                       gt_step_med=float(np.median(np.linalg.norm(d, axis=1))),
                       gt_incoherence=float(np.median(np.linalg.norm(d - med, axis=1))))
        rows.append(row)
        prev = cur
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "frame_motion_audit.csv")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    stems = sorted(p.stem for p in (REPO / "data" / "train").glob("*.geff"))
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(audit_movie, stems):
            rows.extend(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("stem", "t"), k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    n = len(rows)
    frozen = [r for r in rows if r["frozen"]]
    with_gt = [r for r in rows if r.get("n_gt_edges", 0) >= 3]
    jumps = [r for r in with_gt if r["gt_med_um"] >= JUMP_UM and r["gt_incoherence"] < 0.5 * r["gt_med_um"]]
    print(f"{len(stems)} movies, {n} frame pairs; frozen pairs {len(frozen)} ({100 * len(frozen) / n:.2f} %) in {len({r['stem'] for r in frozen})} movies")
    print(f"pairs with >= 3 GT edges: {len(with_gt)}; coherent global jumps (median GT shift >= {JUMP_UM} um, incoherence < half): "
          f"{len(jumps)} ({100 * len(jumps) / max(len(with_gt), 1):.2f} %) in {len({r['stem'] for r in jumps})} movies")
    if jumps:
        err = [np.linalg.norm(np.array([r['pc_dz'], r['pc_dy'], r['pc_dx']]) - np.array([r['gt_dz'], r['gt_dy'], r['gt_dx']])) for r in jumps]
        print(f"  jump size median {np.median([r['gt_med_um'] for r in jumps]):.2f} um (max {max(r['gt_med_um'] for r in jumps):.2f}); "
              f"phase-correlation error vs GT median shift: median {np.median(err):.2f} um")
        for r in sorted(jumps, key=lambda r: -r["gt_med_um"])[:12]:
            print(f"    {r['stem']} t={r['t']}: GT shift {r['gt_med_um']:.2f} um (incoherence {r['gt_incoherence']:.2f}), phase corr {r['pc_um']:.2f} um, frozen {r['frozen']}")
    if frozen:
        fg = [r for r in frozen if r.get("n_gt_edges", 0) >= 1]
        print(f"  frozen pairs with GT edges: {len(fg)}; GT step median on frozen pairs {np.median([r['gt_step_med'] for r in fg]) if fg else float('nan'):.2f} um "
              f"(all pairs: {np.median([r['gt_step_med'] for r in with_gt]):.2f} um)")
    big = [r for r in rows if r["pc_um"] >= JUMP_UM]
    print(f"pairs whose phase-correlation shift >= {JUMP_UM} um: {len(big)} ({100 * len(big) / n:.2f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
