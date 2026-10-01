#!/usr/bin/env python3
"""Local GPU training pipeline for UNet3D + SimpleNodeTransformer edge predictor.

Optimized for local NVIDIA RTX 4070 Ti SUPER (16GB VRAM, Ada Lovelace, SM 8.9)
on Windows CUDA 12.1.
Utilizes the official dataset in data/train (approx 80GB, 199 movies with .zarr and .geff).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_REPO = REPO_ROOT / "artifacts" / "pilkwang_support50" / "repo"
sys.path.insert(0, str(SUPPORT_REPO / "src"))
sys.path.insert(0, str(SUPPORT_REPO / "scripts"))

# Critical configuration for Ada Lovelace / Windows PyTorch 2.5.1+cu121:
# Disable FlashAttention/MemEfficient SDP on small window lengths to prevent CUDA error
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True

from biohub_tracking.models import TemporalUNet3D
from train_unet_transformer import (
    DEFAULT_AUGMENTATIONS,
    FrameWindowDataset,
    UNetNodeTransformer,
    VideoMeta,
    evaluate,
    load_dataset_windows,
    train_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_local_unet")

# Standard 8 held-out validation movies established in the competition
DEFAULT_VAL_STEMS = [
    "44b6_12dfb391",
    "44b6_267148e4",
    "44b6_2a2eff9f",
    "44b6_341df25f",
    "6bba_062c8d37",
    "6bba_07e24132",
    "6bba_085bf656",
    "6bba_09961292",
]


def load_cached_video_data(
    stems: List[str],
    data_dir: Path,
    cache_dir: Optional[Path],
    downsample: Tuple[int, int, int] = (1, 4, 4),
    window_size: int = 2,
    desc: str = "Metadata",
) -> List[Tuple[Any, Any]]:
    ds_str = "_".join(str(x) for x in downsample)
    results: List[Tuple[Any, Any]] = []
    to_load: List[str] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for s in stems:
            cfile = cache_dir / f"{s}_ds{ds_str}_w{window_size}.pkl"
            if cfile.exists():
                try:
                    with open(cfile, "rb") as f:
                        results.append(pickle.load(f))
                    continue
                except Exception:
                    pass
            to_load.append(s)
    else:
        to_load = list(stems)

    if to_load:
        logger.info(f"Extracting {desc} for {len(to_load)} movies ({len(results)} loaded from cache)...")
        for s in tqdm(to_load, desc=desc):
            p = data_dir / s
            try:
                vm, ws = load_dataset_windows(p, window_size=window_size, downsample=downsample)
                if ws:
                    pair = (vm, ws)
                    results.append(pair)
                    if cache_dir is not None:
                        cfile = cache_dir / f"{s}_ds{ds_str}_w{window_size}.pkl"
                        try:
                            with open(cfile, "wb") as f:
                                pickle.dump(pair, f, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as ce:
                            logger.debug(f"Cache write error for {s}: {ce}")
            except Exception as e:
                logger.warning(f"Could not load {p.name}: {e}")
    else:
        logger.info(f"Loaded all {len(results)} {desc} records directly from cache.")

    return results


def setup_datasets(
    train_stems: List[str],
    val_stems: List[str],
    data_dir: Path,
    cache_dir: Optional[Path] = None,
    downsample: Tuple[int, int, int] = (1, 4, 4),
    window_size: int = 2,
    augment: bool = True,
) -> Tuple[FrameWindowDataset, FrameWindowDataset]:
    train_video_data = load_cached_video_data(
        stems=train_stems,
        data_dir=data_dir,
        cache_dir=cache_dir,
        downsample=downsample,
        window_size=window_size,
        desc="Train metadata",
    )

    val_video_data = load_cached_video_data(
        stems=val_stems,
        data_dir=data_dir,
        cache_dir=cache_dir,
        downsample=downsample,
        window_size=window_size,
        desc="Val metadata",
    )

    all_windows = [w for _, ws in train_video_data + val_video_data for w in ws]
    if not all_windows:
        raise RuntimeError("No valid frame windows extracted from datasets!")
    max_nodes = max(max(w.node_counts) for w in all_windows)
    logger.info(f"Extracted {sum(len(w) for _, w in train_video_data)} train windows, "
                f"{sum(len(w) for _, w in val_video_data)} val windows. Max nodes per window: {max_nodes}")

    train_ds = FrameWindowDataset(
        train_video_data,
        max_nodes=max_nodes,
        augmentations=DEFAULT_AUGMENTATIONS if augment else [],
    )
    val_ds = FrameWindowDataset(val_video_data, max_nodes=max_nodes)
    return train_ds, val_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPU Training on RTX 4070 Ti SUPER")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "train"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "experiments" / "local_training" / "checkpoints"))
    parser.add_argument("--pretrained-weights", type=str,
                        default=str(REPO_ROOT / "artifacts" / "pilkwang_support50" / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-movies", type=int, default=None,
                        help="Limit number of train movies (useful for testing / quick training cycles)")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Limit iterations per epoch for quick smoke testing")
    parser.add_argument("--downsample", type=str, default="1,4,4")
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--det-neg-weight", type=float, default=0.01)
    parser.add_argument("--pool-kernel-um", type=float, default=5.0)
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "experiments" / "local_training" / "metadata_cache"))
    parser.add_argument("--no-cache", action="store_true", help="Disable metadata caching")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    downsample = tuple(int(x) for x in args.downsample.split(","))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"VRAM Capacity: {total_vram:.2f} GB")

    all_zarrs = sorted([p.stem for p in data_dir.glob("*.zarr") if (data_dir / f"{p.stem}.geff").exists()])
    logger.info(f"Found {len(all_zarrs)} complete dataset pairs (.zarr + .geff) in {data_dir}")

    val_stems = [s for s in DEFAULT_VAL_STEMS if s in all_zarrs]
    train_stems = [s for s in all_zarrs if s not in val_stems]

    if args.max_train_movies and args.max_train_movies < len(train_stems):
        logger.info(f"Selecting subset of {args.max_train_movies} movies for training cycle...")
        train_stems = train_stems[:args.max_train_movies]

    logger.info(f"Partition: {len(train_stems)} train movies, {len(val_stems)} val movies.")

    train_ds, val_ds = setup_datasets(
        train_stems=train_stems,
        val_stems=val_stems,
        data_dir=data_dir,
        cache_dir=cache_dir,
        downsample=downsample,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=False,
    )

    # Initialize UNet3D + NodeTransformer
    unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    model = UNetNodeTransformer(unet=unet, unet_out_channels=32, pos_feat_dim=32).to(device)

    pretrained_path = Path(args.pretrained_weights)
    if pretrained_path.exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}...")
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights ({len(missing)} missing, {len(unexpected)} unexpected keys).")
    else:
        logger.info("No pretrained weights found; training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = 0.0
    history = []

    logger.info(f"Starting training on {device} for {args.epochs} epochs (batch_size={args.batch_size})...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        edge_loss, det_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            det_loss_weight=args.det_loss_weight,
            det_neg_weight=args.det_neg_weight,
            max_iters=args.max_iters,
            pool_kernel_um=args.pool_kernel_um,
        )
        lr_scheduler.step()
        train_elapsed = time.time() - t0

        t1 = time.time()
        val_loss, val_acc, val_recall = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            pool_kernel_um=args.pool_kernel_um,
        )
        val_elapsed = time.time() - t1

        score = val_acc * val_recall
        is_best = score > best_score
        if is_best:
            best_score = score
            save_path = output_dir / "best_local_unet_transformer.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"*** New best score: {best_score:.4f} (Acc: {val_acc:.4f}, Recall: {val_recall:.4f}) -> Saved {save_path.name}")

        epoch_record = {
            "epoch": epoch,
            "edge_loss": float(edge_loss),
            "det_loss": float(det_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_recall": float(val_recall),
            "val_score": float(score),
            "best_score": float(best_score),
            "train_sec": round(train_elapsed, 1),
            "val_sec": round(val_elapsed, 1),
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | Edge Loss: {edge_loss:.4f} | Det Loss: {det_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f} | Score: {score:.4f} | "
            f"Time: {train_elapsed:.1f}s train, {val_elapsed:.1f}s val"
        )

        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    logger.info(f"Training completed! Best val score: {best_score:.4f}. Model checkpoints saved in {output_dir}")


if __name__ == "__main__":
    main()
