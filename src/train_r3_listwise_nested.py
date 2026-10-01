from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class TargetPack:
    edge_idx: np.ndarray
    valid: np.ndarray
    positive: np.ndarray
    dataset: np.ndarray
    group: np.ndarray
    variant: np.ndarray
    target_id: np.ndarray


def parse_csv_numbers(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def load_data(paths: list[Path], head_contract: Path | None):
    arrays: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "features",
            "labels",
            "groups",
            "target_ids",
            "datasets",
            "variants",
            "base_scores",
        )
    }
    embedded_head = None
    for path in paths:
        d = np.load(path, allow_pickle=True)
        n = len(d["labels"])
        arrays["features"].append(np.asarray(d["features"]))
        arrays["labels"].append(np.asarray(d["labels"], dtype=np.int8))
        arrays["groups"].append(d["groups"].astype(str))
        arrays["target_ids"].append(np.asarray(d["target_ids"], dtype=np.int64))
        arrays["datasets"].append(d["datasets"].astype(str))
        arrays["variants"].append(
            d["variants"].astype(str)
            if "variants" in d.files
            else np.full(n, "unknown", dtype="U16")
        )
        if "base_scores" in d.files:
            arrays["base_scores"].append(np.asarray(d["base_scores"], dtype=np.float32))
        else:
            arrays["base_scores"].append(np.full(n, np.nan, dtype=np.float32))
        if embedded_head is None and "head_weight" in d.files and "head_bias" in d.files:
            embedded_head = (
                np.asarray(d["head_weight"], dtype=np.float32).reshape(-1),
                float(np.asarray(d["head_bias"], dtype=np.float32).reshape(-1)[0]),
            )

    x = np.concatenate(arrays["features"], axis=0)
    y = np.concatenate(arrays["labels"], axis=0)
    groups = np.concatenate(arrays["groups"], axis=0)
    target_ids = np.concatenate(arrays["target_ids"], axis=0)
    datasets = np.concatenate(arrays["datasets"], axis=0)
    variants = np.concatenate(arrays["variants"], axis=0)
    base_scores = np.concatenate(arrays["base_scores"], axis=0)

    if head_contract is not None:
        h = np.load(head_contract, allow_pickle=True)
        w0 = np.asarray(h["head_weight"], dtype=np.float32).reshape(-1)
        b0 = float(np.asarray(h["head_bias"], dtype=np.float32).reshape(-1)[0])
    elif embedded_head is not None:
        w0, b0 = embedded_head
    else:
        raise RuntimeError("Need --head-contract or embedded head_weight/head_bias")

    if x.shape[1] != len(w0):
        raise RuntimeError(f"feature/head mismatch {x.shape} vs {w0.shape}")
    missing_base = ~np.isfinite(base_scores)
    if missing_base.any():
        base_scores[missing_base] = x[missing_base].astype(np.float32) @ w0 + b0
    return x, y, groups, target_ids, datasets, variants, base_scores, w0, b0


def build_targets(
    labels: np.ndarray,
    groups: np.ndarray,
    target_ids: np.ndarray,
    datasets: np.ndarray,
    variants: np.ndarray,
) -> TargetPack:
    meta = pd.DataFrame(
        {
            "edge_idx": np.arange(len(labels), dtype=np.int64),
            "label": labels,
            "group": groups,
            "target_id": target_ids,
            "dataset": datasets,
            "variant": variants,
        }
    )
    chunks: list[np.ndarray] = []
    rows: list[tuple[str, str, str, int]] = []
    max_k = 0
    for (dataset, variant, target), g in meta.groupby(
        ["dataset", "variant", "target_id"], sort=False
    ):
        idx = g["edge_idx"].to_numpy(dtype=np.int64)
        yy = labels[idx]
        if not np.any(yy > 0) or not np.any(yy == 0):
            continue
        chunks.append(idx)
        rows.append((str(dataset), str(g["group"].iloc[0]), str(variant), int(target)))
        max_k = max(max_k, len(idx))
    if not chunks:
        raise RuntimeError("No supervised target groups")

    n = len(chunks)
    edge_idx = np.full((n, max_k), -1, dtype=np.int64)
    valid = np.zeros((n, max_k), dtype=bool)
    positive = np.zeros((n, max_k), dtype=bool)
    for i, idx in enumerate(chunks):
        edge_idx[i, : len(idx)] = idx
        valid[i, : len(idx)] = True
        positive[i, : len(idx)] = labels[idx] > 0

    return TargetPack(
        edge_idx=edge_idx,
        valid=valid,
        positive=positive,
        dataset=np.asarray([r[0] for r in rows], dtype="U96"),
        group=np.asarray([r[1] for r in rows], dtype="U32"),
        variant=np.asarray([r[2] for r in rows], dtype="U24"),
        target_id=np.asarray([r[3] for r in rows], dtype=np.int64),
    )


def deterministic_movie_split(names: np.ndarray, val_frac: float) -> tuple[set[str], set[str]]:
    unique = sorted(set(map(str, names)))
    if len(unique) < 2:
        return set(unique), set()
    scored = sorted(
        unique,
        key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest(),
    )
    n_val = max(1, min(len(scored) - 1, int(round(len(scored) * val_frac))))
    val = set(scored[:n_val])
    train = set(scored[n_val:])
    return train, val


def fit_listwise(
    x: np.ndarray,
    pack: TargetPack,
    target_indices: np.ndarray,
    w0: np.ndarray,
    b0: float,
    *,
    epochs: int,
    lr: float,
    anchor: float,
    batch_targets: int,
    device: str,
    seed: int,
) -> np.ndarray:
    if len(target_indices) == 0:
        raise RuntimeError("No targets selected for training")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    weight = torch.nn.Parameter(torch.as_tensor(w0, dtype=torch.float32, device=dev).clone())
    ref = torch.as_tensor(w0, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([weight], lr=lr, weight_decay=0.0)
    x_cpu = torch.from_numpy(x)
    edge_idx = torch.from_numpy(pack.edge_idx[target_indices])
    valid = torch.from_numpy(pack.valid[target_indices])
    positive = torch.from_numpy(pack.positive[target_indices])

    for epoch in range(epochs):
        order = torch.randperm(len(target_indices))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_targets):
            q = order[start : start + batch_targets]
            ids = edge_idx[q]
            vb = valid[q]
            pb = positive[q]
            safe = ids.clamp_min(0)
            xb = x_cpu[safe].to(dev, dtype=torch.float32, non_blocking=True)
            vb = vb.to(dev, non_blocking=True)
            pb = pb.to(dev, non_blocking=True)
            scores = torch.einsum("bkd,d->bk", xb, weight) + float(b0)
            neg_inf = torch.finfo(scores.dtype).min
            all_scores = scores.masked_fill(~vb, neg_inf)
            pos_scores = scores.masked_fill(~pb, neg_inf)
            loss = (
                torch.logsumexp(all_scores, dim=1)
                - torch.logsumexp(pos_scores, dim=1)
            ).mean()
            loss = loss + anchor * (weight - ref).square().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(q)
            seen += len(q)
        if epoch in {0, epochs - 1}:
            print(
                "TRAIN",
                "epoch", epoch + 1,
                "epochs", epochs,
                "loss", total_loss / max(seen, 1),
                "drift", float(torch.linalg.vector_norm(weight - ref).detach().cpu()),
                flush=True,
            )
    return weight.detach().cpu().numpy().astype(np.float32)


def score_edges(x: np.ndarray, weight: np.ndarray, b0: float, device: str) -> np.ndarray:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    w = torch.as_tensor(weight, dtype=torch.float32, device=dev)
    out = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), 131072):
            stop = min(start + 131072, len(x))
            xb = torch.from_numpy(x[start:stop]).to(dev, dtype=torch.float32)
            out[start:stop] = (xb @ w + float(b0)).cpu().numpy()
    return out


def target_metrics(pack: TargetPack, labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    ids = pack.edge_idx[mask]
    valid = pack.valid[mask]
    positive = pack.positive[mask]
    if len(ids) == 0:
        return {
            "n_targets": 0,
            "top1": float("nan"),
            "top2": float("nan"),
            "mrr": float("nan"),
        }
    safe = np.maximum(ids, 0)
    ss = scores[safe].astype(np.float64)
    ss[~valid] = -np.inf
    order = np.argsort(-ss, axis=1)
    pos_sorted = np.take_along_axis(positive, order, axis=1)
    ranks = np.argmax(pos_sorted, axis=1) + 1
    return {
        "n_targets": int(len(ids)),
        "top1": float(np.mean(ranks == 1)),
        "top2": float(np.mean(ranks <= 2)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def compare_winners(
    pack: TargetPack,
    labels: np.ndarray,
    base: np.ndarray,
    probe: np.ndarray,
    mask: np.ndarray,
) -> dict:
    ids = pack.edge_idx[mask]
    valid = pack.valid[mask]
    if len(ids) == 0:
        return {"winner_changes": 0, "fixes": 0, "breaks": 0}
    safe = np.maximum(ids, 0)
    bs = base[safe].astype(np.float64)
    ps = probe[safe].astype(np.float64)
    bs[~valid] = -np.inf
    ps[~valid] = -np.inf
    bpos = np.argmax(bs, axis=1)
    ppos = np.argmax(ps, axis=1)
    bw = safe[np.arange(len(safe)), bpos]
    pw = safe[np.arange(len(safe)), ppos]
    bc = labels[bw] > 0
    pc = labels[pw] > 0
    changed = bw != pw
    return {
        "winner_changes": int(changed.sum()),
        "fixes": int(((~bc) & pc & changed).sum()),
        "breaks": int((bc & (~pc) & changed).sum()),
        "both_correct_changed": int((bc & pc & changed).sum()),
    }


def eval_partition(
    pack: TargetPack,
    labels: np.ndarray,
    base: np.ndarray,
    probe: np.ndarray,
    mask: np.ndarray,
) -> dict:
    result: dict[str, object] = {}
    variants = sorted(set(pack.variant[mask].tolist()))
    for variant in ["all", *variants]:
        vm = mask if variant == "all" else mask & (pack.variant == variant)
        b = target_metrics(pack, labels, base, vm)
        p = target_metrics(pack, labels, probe, vm)
        result[variant] = {
            "base": b,
            "probe": p,
            "delta_top1": p["top1"] - b["top1"] if b["n_targets"] else float("nan"),
            "delta_mrr": p["mrr"] - b["mrr"] if b["n_targets"] else float("nan"),
            **compare_winners(pack, labels, base, probe, vm),
        }
    return result


def robust_inner_value(evaluation: dict) -> tuple[float, float]:
    per_variant = [v for k, v in evaluation.items() if k != "all" and v["base"]["n_targets"] > 0]
    if not per_variant:
        per_variant = [evaluation["all"]]
    top1 = [float(v["delta_top1"]) for v in per_variant]
    mrr = [float(v["delta_mrr"]) for v in per_variant]
    return min(top1), float(np.mean(mrr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, nargs="+", required=True)
    ap.add_argument("--head-contract", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", default="20")
    ap.add_argument("--lrs", default="0.0003,0.001")
    ap.add_argument("--anchors", default="0.02,0.05,0.1")
    ap.add_argument("--inner-val-frac", type=float, default=0.2)
    ap.add_argument("--batch-targets", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    epochs_grid = parse_csv_numbers(args.epochs, int)
    lr_grid = parse_csv_numbers(args.lrs, float)
    anchor_grid = parse_csv_numbers(args.anchors, float)
    configs = [
        {"epochs": e, "lr": lr, "anchor": a}
        for e, lr, a in itertools.product(epochs_grid, lr_grid, anchor_grid)
    ]

    x, labels, groups, target_ids, datasets, variants, base, w0, b0 = load_data(
        args.npz, args.head_contract
    )
    pack = build_targets(labels, groups, target_ids, datasets, variants)
    print(
        "DATA",
        "edges", len(x),
        "features", x.shape,
        "targets", len(pack.edge_idx),
        "groups", sorted(set(pack.group.tolist())),
        "variants", sorted(set(pack.variant.tolist())),
        "device", args.device,
        flush=True,
    )

    outer_results = []
    inner_by_config: dict[str, list[dict[str, float]]] = {
        json.dumps(c, sort_keys=True): [] for c in configs
    }
    head_weights: dict[str, np.ndarray] = {}

    for fold_idx, holdout in enumerate(sorted(set(pack.group.tolist()))):
        outer_train = pack.group != holdout
        outer_val = pack.group == holdout
        train_movies, inner_val_movies = deterministic_movie_split(
            pack.dataset[outer_train], args.inner_val_frac
        )
        if not inner_val_movies:
            raise RuntimeError(f"Need >=2 training movies for nested split, holdout={holdout}")
        inner_train = outer_train & np.isin(pack.dataset, list(train_movies))
        inner_val = outer_train & np.isin(pack.dataset, list(inner_val_movies))
        print(
            "FOLD", holdout,
            "outer_train_targets", int(outer_train.sum()),
            "outer_val_targets", int(outer_val.sum()),
            "inner_train_movies", len(train_movies),
            "inner_val_movies", len(inner_val_movies),
            flush=True,
        )

        candidates = []
        for cfg_idx, cfg in enumerate(configs):
            weight = fit_listwise(
                x,
                pack,
                np.flatnonzero(inner_train),
                w0,
                b0,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                anchor=cfg["anchor"],
                batch_targets=args.batch_targets,
                device=args.device,
                seed=args.seed + fold_idx * 1000 + cfg_idx,
            )
            pred = score_edges(x, weight, b0, args.device)
            ev = eval_partition(pack, labels, base, pred, inner_val)
            robust_top1, mean_mrr = robust_inner_value(ev)
            drift = float(np.linalg.norm(weight - w0))
            row = {
                "config": cfg,
                "robust_delta_top1": robust_top1,
                "mean_delta_mrr": mean_mrr,
                "weight_l2_drift": drift,
                "evaluation": ev,
            }
            candidates.append((robust_top1, mean_mrr, -drift, row))
            inner_by_config[json.dumps(cfg, sort_keys=True)].append(
                {
                    "robust_delta_top1": robust_top1,
                    "mean_delta_mrr": mean_mrr,
                    "weight_l2_drift": drift,
                }
            )
            print("INNER", holdout, json.dumps(row, allow_nan=True), flush=True)

        candidates.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
        selected = candidates[0][3]
        use_identity = selected["robust_delta_top1"] < 0.0
        if use_identity:
            outer_pred = base.copy()
            outer_weight = w0.copy()
            selected_config = None
        else:
            selected_config = selected["config"]
            outer_weight = fit_listwise(
                x,
                pack,
                np.flatnonzero(outer_train),
                w0,
                b0,
                epochs=selected_config["epochs"],
                lr=selected_config["lr"],
                anchor=selected_config["anchor"],
                batch_targets=args.batch_targets,
                device=args.device,
                seed=args.seed + 10000 + fold_idx,
            )
            outer_pred = score_edges(x, outer_weight, b0, args.device)
        outer_eval = eval_partition(pack, labels, base, outer_pred, outer_val)
        fold_result = {
            "holdout": holdout,
            "selected_config": selected_config,
            "selected_inner": selected,
            "used_identity": use_identity,
            "outer_evaluation": outer_eval,
            "outer_weight_l2_drift": float(np.linalg.norm(outer_weight - w0)),
            "inner_train_movies": sorted(train_movies),
            "inner_val_movies": sorted(inner_val_movies),
        }
        outer_results.append(fold_result)
        head_weights[f"holdout_{holdout}"] = outer_weight
        print("OUTER", json.dumps(fold_result, allow_nan=True), flush=True)

    aggregate_candidates = []
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        rows = inner_by_config[key]
        robust_values = [r["robust_delta_top1"] for r in rows]
        mrr_values = [r["mean_delta_mrr"] for r in rows]
        drifts = [r["weight_l2_drift"] for r in rows]
        aggregate_candidates.append(
            (
                min(robust_values),
                float(np.mean(robust_values)),
                float(np.mean(mrr_values)),
                -float(np.mean(drifts)),
                cfg,
            )
        )
    aggregate_candidates.sort(reverse=True, key=lambda x: x[:4])
    best = aggregate_candidates[0]
    final_config = best[4] if best[0] >= 0.0 else None
    if final_config is None:
        final_weight = w0.copy()
    else:
        final_weight = fit_listwise(
            x,
            pack,
            np.arange(len(pack.edge_idx), dtype=np.int64),
            w0,
            b0,
            epochs=final_config["epochs"],
            lr=final_config["lr"],
            anchor=final_config["anchor"],
            batch_targets=args.batch_targets,
            device=args.device,
            seed=args.seed + 20000,
        )
    head_weights["final"] = final_weight

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data": {
            "npz": [str(p) for p in args.npz],
            "edges": int(len(x)),
            "targets": int(len(pack.edge_idx)),
            "groups": sorted(set(pack.group.tolist())),
            "variants": sorted(set(pack.variant.tolist())),
        },
        "configs": configs,
        "outer_folds": outer_results,
        "aggregate_inner_selection": {
            "ranking": [
                {
                    "min_robust_delta_top1": r[0],
                    "mean_robust_delta_top1": r[1],
                    "mean_delta_mrr": r[2],
                    "neg_mean_drift": r[3],
                    "config": r[4],
                }
                for r in aggregate_candidates
            ],
            "final_config": final_config,
        },
        "final_weight_l2_drift": float(np.linalg.norm(final_weight - w0)),
    }
    (args.out_dir / "nested_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    np.savez_compressed(
        args.out_dir / "heads.npz",
        original_weight=w0,
        bias=np.asarray([b0], dtype=np.float32),
        **head_weights,
    )
    print("SAVED", args.out_dir, flush=True)
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
