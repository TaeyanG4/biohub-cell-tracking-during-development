#!/usr/bin/env python3
"""Stage and upload local 4070 Ti SUPER checkpoint weights as a Kaggle private dataset.

Dataset slug: biohub-local-4070ti-weights
Owner: taeyangg4
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("upload_local_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "experiments" / "local_training" / "checkpoints"
STAGING_DIR = REPO_ROOT / "experiments" / "local_training" / "dataset_export"
DATASET_SLUG = "biohub-local-4070ti-weights"
DATASET_TITLE = "Biohub Local 4070Ti UNet Transformer Weights"


def main() -> int:
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username") or "taeyangg4"
    full_dataset_ref = f"{user}/{DATASET_SLUG}"

    best_weights = CHECKPOINTS_DIR / "best_local_unet_transformer.pth"
    history_file = CHECKPOINTS_DIR / "training_history.json"

    if not best_weights.exists():
        logger.error(f"Checkpoint not found at {best_weights}")
        return 1

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    target_weights = STAGING_DIR / "best_local_unet_transformer.pth"
    shutil.copy2(best_weights, target_weights)
    logger.info(f"Staged {best_weights.name} ({target_weights.stat().st_size / (1024*1024):.2f} MB)")

    if history_file.exists():
        shutil.copy2(history_file, STAGING_DIR / "training_history.json")
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                hist = json.load(f)
            last_rec = hist[-1] if hist else {}
            logger.info(
                f"History summary: {len(hist)} epochs, best_score: {last_rec.get('best_score', 0):.4f}, "
                f"val_acc: {last_rec.get('val_acc', 0):.4f}, val_recall: {last_rec.get('val_recall', 0):.4f}"
            )
        except Exception:
            pass

    metadata_path = STAGING_DIR / "dataset-metadata.json"
    metadata = {
        "title": DATASET_TITLE,
        "id": full_dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Local NVIDIA RTX 4070 Ti SUPER trained UNet3D + SimpleNodeTransformer checkpoint "
            "for Biohub Cell Tracking during Development. Trained on the full 199 .zarr movies "
            "with NVMe metadata caching."
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset metadata prepared at {metadata_path}")

    # Check if dataset already exists on Kaggle
    dataset_exists = False
    try:
        existing = api.dataset_status(full_dataset_ref)
        if existing:
            dataset_exists = True
            logger.info(f"Existing dataset found on Kaggle: {full_dataset_ref} (status: {existing})")
    except Exception:
        dataset_exists = False

    if dataset_exists:
        logger.info(f"Creating new version for {full_dataset_ref}...")
        api.dataset_create_version(
            folder=str(STAGING_DIR),
            version_notes="Updated weights trained on local RTX 4070 Ti SUPER",
            delete_old_versions=False,
            dir_mode="zip",
            quiet=False,
        )
        logger.info(f"Successfully created new version of {full_dataset_ref}!")
    else:
        logger.info(f"Creating new private dataset {full_dataset_ref}...")
        api.dataset_create_new(
            folder=str(STAGING_DIR),
            dir_mode="zip",
            public=False,
            quiet=False,
        )
        logger.info(f"Successfully created new private dataset {full_dataset_ref}!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
