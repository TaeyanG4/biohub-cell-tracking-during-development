#!/usr/bin/env python3
"""Image crops for an appearance-based division (mitosis) classifier (HANDOFF section 24).

For every train movie: positives = GT parents with two children (the division frame t); negatives = GT nodes
with exactly one child that are not within +-2 frames of a division along their lineage (random sample per
movie) plus, optionally, the pipeline's false forks from the C016 candidate CSVs (label 0 rows, parent P).
Each sample = frames t-1, t, t+1 (clamped) cropped to (8 z, 32 y, 32 x) full-resolution voxels around the
node (13 x 13 x 13 um), normalised with the movie's 0.001 / 0.999 intensity quantiles and clamped to [0, 3].

    python src/division_crops_extract.py --out experiments/candidates/c031_division_cnn/crops [--neg-per-movie 40] [--stems a,b]
Writes <out>/<stem>.npz with crops (N, 3, 8, 32, 32) float16, label (1 division / 0 not), t, z, y, x, kind.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval_pp_variants_local import build_namespace  # noqa: E402

CROP = (8, 32, 32)
FRAMES = (-1, 0, 1)


def crop_at(vol, t_idx, z, y, x):
    T = vol.shape[0]
    out = np.zeros((len(FRAMES),) + CROP, dtype=np.float32)
    hz, hy, hx = CROP[0] // 2, CROP[1] // 2, CROP[2] // 2
    for k, dt in enumerate(FRAMES):
        t = min(max(t_idx + dt, 0), T - 1)
        z0, y0, x0 = z - hz, y - hy, x - hx
        zs, ys, xs = slice(max(z0, 0), min(z0 + CROP[0], vol.shape[1])), slice(max(y0, 0), min(y0 + CROP[1], vol.shape[2])), slice(max(x0, 0), min(x0 + CROP[2], vol.shape[3]))
        block = np.asarray(vol[t, zs, ys, xs], dtype=np.float32)
        oz, oy, ox = zs.start - z0, ys.start - y0, xs.start - x0
        out[k, oz:oz + block.shape[0], oy:oy + block.shape[1], ox:ox + block.shape[2]] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "train")
    ap.add_argument("--stems", default="")
    ap.add_argument("--neg-per-movie", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = [s for s in args.stems.split(",") if s] or sorted(p.stem for p in args.data_dir.glob("*.geff"))
    with contextlib.redirect_stdout(io.StringIO()):
        ns = build_namespace((REPO_ROOT / "experiments/candidates/c011_x138_zero/biohub-c011-x138-zero.ipynb").resolve(), {}, None)
    rng = np.random.default_rng(args.seed)
    total_pos = total_neg = 0
    for stem in stems:
        dest = args.out / f"{stem}.npz"
        if dest.exists():
            continue
        gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](args.data_dir / f"{stem}.geff"))
        children: dict[int, list[int]] = {}
        parent: dict[int, int] = {}
        for s, d in gt_edges:
            children.setdefault(s, []).append(d); parent[d] = s
        parents2 = [n for n, ch in children.items() if len(ch) >= 2]
        near_div = set()
        for p in parents2:  # lineage neighbours of a division are ambiguous: exclude from negatives
            a = p
            for _ in range(2):
                a = parent.get(a)
                if a is None: break
                near_div.add(a)
            for c in children[p]:
                near_div.add(c)
                for gc in children.get(c, []): near_div.add(gc)
        neg_pool = [n for n, ch in children.items() if len(ch) == 1 and n not in near_div and n not in parents2]
        neg = list(rng.choice(neg_pool, size=min(args.neg_per_movie, len(neg_pool)), replace=False)) if neg_pool else []
        g = zarr.open_group(str(args.data_dir / f"{stem}.zarr"), mode="r")
        q = g.attrs["image_statistics"]["quantiles"]; lo, hi = float(q["0.001"]), float(q["0.999"])
        vol = g["0"]
        crops, labels, meta, kinds = [], [], [], []
        for label, ids, kind in ((1, parents2, "gt_division"), (0, neg, "gt_single_child")):
            for n in ids:
                t, z, y, x = (int(round(float(v))) for v in gt_nodes[n])
                c = crop_at(vol, t, z, y, x)
                c = np.clip((c - lo) / (hi - lo + 1e-6), 0.0, 3.0)
                crops.append(c.astype(np.float16)); labels.append(label); meta.append((t, z, y, x)); kinds.append(kind)
        if crops:
            np.savez_compressed(dest, crops=np.stack(crops), label=np.array(labels, dtype=np.int8), tzyx=np.array(meta, dtype=np.int32), kind=np.array(kinds))
        total_pos += len(parents2); total_neg += len(neg)
        print(f"{stem}: divisions {len(parents2)}, negatives {len(neg)}", flush=True)
    print(f"done: positives {total_pos}, negatives {total_neg}, movies {len(stems)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
