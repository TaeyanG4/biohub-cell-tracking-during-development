from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_model(path: Path, device: str):
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def extract(model, node_features, node_pos, edge_pos, edge_indices, node_mask, edge_mask):
    with torch.inference_mode():
        logits, node_repr, raw_edge_repr, aux = model(
            node_features,
            node_pos,
            edge_pos,
            edge_indices,
            node_mask,
            edge_mask,
        )
        norm_edge_repr = model.head_norm(raw_edge_repr)
        reconstructed_logits = model.head(model.head_dropout(norm_edge_repr))

    max_diff = float((reconstructed_logits - logits).abs().max().detach().cpu())
    if max_diff > 1e-6:
        raise RuntimeError(f"head reconstruction drifted: max_diff={max_diff}")

    return {
        "logits": logits.detach().cpu().numpy(),
        "node_repr": node_repr.detach().cpu().numpy(),
        "raw_edge_repr": raw_edge_repr.detach().cpu().numpy(),
        "norm_edge_repr": norm_edge_repr.detach().cpu().numpy(),
        "aux": aux.detach().cpu().numpy(),
        "max_logit_diff": np.asarray([max_diff], dtype=np.float32),
    }


def synthetic(model, device: str):
    batch, nodes, edges = 1, 4, 3
    node_features = torch.randn(batch, nodes, 19, device=device)
    node_pos = torch.randn(batch, nodes, 3, device=device)
    edge_indices = torch.tensor([[[0, 1], [1, 2], [2, 3]]], dtype=torch.long, device=device)
    idx0 = edge_indices[:, :, 0, None].expand(-1, -1, 3)
    idx1 = edge_indices[:, :, 1, None].expand(-1, -1, 3)
    edge_pos = (node_pos.gather(1, idx0) + node_pos.gather(1, idx1)) / 2
    node_mask = torch.ones(batch, nodes, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch, edges, dtype=torch.bool, device=device)
    return extract(model, node_features, node_pos, edge_pos, edge_indices, node_mask, edge_mask)


def from_npz(model, input_path: Path, device: str):
    arr = np.load(input_path)
    names = [
        "node_features",
        "node_pos",
        "edge_pos",
        "edge_indices",
        "node_mask",
        "edge_mask",
    ]
    missing = [name for name in names if name not in arr]
    if missing:
        raise KeyError(f"missing arrays: {missing}")

    tensors = []
    for name in names:
        x = torch.from_numpy(arr[name])
        if name == "edge_indices":
            x = x.long()
        elif name in {"node_mask", "edge_mask"}:
            x = x.bool()
        else:
            x = x.float()
        tensors.append(x.to(device))
    return extract(model, *tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    print("head_weight_shape", tuple(model.head.weight.shape))

    if args.input_npz is None:
        out = synthetic(model, args.device)
        print("synthetic_shapes", {k: v.shape for k, v in out.items()})
        print("max_logit_diff", float(out["max_logit_diff"][0]))
    else:
        out = from_npz(model, args.input_npz, args.device)
        print("feature_shapes", {k: v.shape for k, v in out.items()})
        if args.output_npz is not None:
            args.output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output_npz, **out)
            print("wrote", args.output_npz)


if __name__ == "__main__":
    main()
