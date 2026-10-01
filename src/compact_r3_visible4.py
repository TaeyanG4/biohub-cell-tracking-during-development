from __future__ import annotations

import argparse
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
    w0 = model.head.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b0 = float(model.head.bias.detach().cpu().numpy().reshape(-1)[0])
    meta, feat = aggregate_unique(d, w0, b0)
    base = feat @ w0 + b0

    keep: list[int] = []
    supervised_targets = 0
    positive_targets = 0
    meta = meta.reset_index(drop=True)
    for (_, _), g in meta.groupby(["dataset", "target_id"], sort=False):
        idx = g.index.to_numpy(dtype=np.int64)
        labels = meta.loc[idx, "label"].to_numpy(dtype=np.int8)
        pos = idx[labels > 0]
        neg = idx[labels == 0]
        if len(pos) == 0:
            continue
        positive_targets += 1
        if len(neg) == 0:
            continue
        supervised_targets += 1
        hard = neg[np.argsort(base[neg])[-min(args.hard_k, len(neg)):]]
        keep.extend(pos.tolist())
        keep.extend(hard.tolist())

    keep_arr = np.asarray(sorted(set(keep)), dtype=np.int64)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=feat[keep_arr].astype(np.float16),
        labels=meta.loc[keep_arr, "label"].to_numpy(dtype=np.int8),
        groups=meta.loc[keep_arr, "group"].astype(str).to_numpy(dtype="U32"),
        edge_ids=meta.loc[keep_arr, "edge_id"].to_numpy(dtype=np.int64),
        source_ids=meta.loc[keep_arr, "source_id"].to_numpy(dtype=np.int64),
        target_ids=meta.loc[keep_arr, "target_id"].to_numpy(dtype=np.int64),
        datasets=meta.loc[keep_arr, "dataset"].astype(str).to_numpy(dtype="U64"),
        variants=np.full(len(keep_arr), "pred", dtype="U8"),
        base_scores=base[keep_arr].astype(np.float32),
        head_weight=w0,
        head_bias=np.asarray([b0], dtype=np.float32),
    )
    print(
        "saved", out,
        "unique_edges", len(meta),
        "positive_targets", positive_targets,
        "supervised_targets", supervised_targets,
        "kept", len(keep_arr),
        "shape", (len(keep_arr), feat.shape[1]),
    )


if __name__ == "__main__":
    main()
