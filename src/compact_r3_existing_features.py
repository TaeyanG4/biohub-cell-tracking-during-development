from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_r3_target_ranker import aggregate_unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--hoct", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard-k", type=int, default=8)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    model = torch.jit.load(str(args.hoct), map_location="cpu").eval()
    w0 = model.head.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(model.head.bias.detach().cpu().numpy().reshape(-1)[0])
    meta, feat = aggregate_unique(d, w0, b0)
    base = feat @ w0 + b0

    keep: list[int] = []
    targets = positives = 0
    for (_, target_id), g in meta.groupby(["dataset", "target_id"], sort=False):
        idx = g.index.to_numpy(dtype=np.int64)
        pos = idx[meta.loc[idx, "label"].to_numpy() > 0]
        neg = idx[meta.loc[idx, "label"].to_numpy() == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        targets += 1
        positives += len(pos)
        k = min(args.hard_k, len(neg))
        hard = neg[np.argpartition(base[neg], -k)[-k:]] if k < len(neg) else neg
        keep.extend(pos.tolist())
        keep.extend(hard.tolist())

    keep_arr = np.asarray(sorted(set(keep)), dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=feat[keep_arr].astype(np.float16),
        labels=meta.loc[keep_arr, "label"].to_numpy(np.int8),
        groups=meta.loc[keep_arr, "group"].astype(str).to_numpy(dtype="U32"),
        edge_ids=meta.loc[keep_arr, "edge_id"].to_numpy(np.int64),
        source_ids=meta.loc[keep_arr, "source_id"].to_numpy(np.int64),
        target_ids=meta.loc[keep_arr, "target_id"].to_numpy(np.int64),
        datasets=meta.loc[keep_arr, "dataset"].astype(str).to_numpy(dtype="U64"),
        base_scores=base[keep_arr].astype(np.float32),
    )
    report = {
        "source_rows": int(len(d["labels"])),
        "unique_edges": int(len(meta)),
        "kept_edges": int(len(keep_arr)),
        "supervised_targets": int(targets),
        "positive_edges": int(positives),
        "hard_k": int(args.hard_k),
    }
    args.out.with_suffix(".stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
