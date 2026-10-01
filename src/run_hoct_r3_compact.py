from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from extract_hoct_r3_compact import extract_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard-k", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    submission = pd.read_csv(args.submission)
    model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
    buckets = {k: [] for k in (
        "features", "labels", "groups", "edge_ids", "source_ids", "target_ids",
        "base_scores", "views", "datasets",
    )}
    stats = []
    for name in args.datasets:
        x, y, g, e, s, t, b, v, st = extract_one(
            name, submission, args.image_root, args.gt_root, model, args.hard_k
        )
        values = (x, y, g, e, s, t, b, v, np.full(len(y), name, dtype="U64"))
        for key, value in zip(buckets, values):
            buckets[key].append(value)
        stats.append(st)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in buckets.items()}
    np.savez_compressed(args.out, **arrays)
    args.out.with_suffix(".stats.json").write_text(
        json.dumps({"hard_k": args.hard_k, "movies": stats}, indent=2), encoding="utf-8"
    )
    print("saved", args.out, "rows", len(arrays["labels"]), "shape", arrays["features"].shape)


if __name__ == "__main__":
    main()
