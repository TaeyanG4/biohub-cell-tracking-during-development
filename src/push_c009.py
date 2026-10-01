#!/usr/bin/env python3
"""Push Candidate C009 (Boost Geometric Fusion) to Kaggle as a GPU kernel.

Kernel: taeyangg4/biohub-c009-boost-geometric-fusion
Attached Datasets:
- pilkwang/biohub-deepcenter-unet3d-center-prior-v1
- pilkwang/biohub-tracking-support-pack-50ep-v1
- taeyangg4/biohub-local-4070ti-weights
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("push_c009")

REPO_ROOT = Path(__file__).resolve().parents[1]
C009_DIR = REPO_ROOT / "experiments" / "candidates" / "c009_boost_geometric_fusion"


def main() -> int:
    logger.info(f"Checking C009 candidate folder at {C009_DIR}...")
    nb_path = C009_DIR / "biohub-c009-boost-geometric-fusion.ipynb"
    meta_path = C009_DIR / "kernel-metadata.json"

    if not nb_path.exists():
        logger.error(f"Missing notebook: {nb_path}")
        return 1
    if not meta_path.exists():
        logger.error(f"Missing metadata: {meta_path}")
        return 1

    logger.info("Running offline verification suite before push...")
    res = subprocess.run([sys.executable, "src/verify_c009.py"], cwd=str(REPO_ROOT))
    if res.returncode != 0:
        logger.error("C009 offline verification failed! Aborting push.")
        return res.returncode

    logger.info("Pushing Candidate C009 to Kaggle...")
    cmd = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(C009_DIR),
    ]
    push_res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if push_res.returncode == 0:
        logger.info("Candidate C009 successfully pushed to Kaggle!")
    else:
        logger.error(f"Kaggle kernel push failed with code {push_res.returncode}")
    return push_res.returncode


if __name__ == "__main__":
    sys.exit(main())
