#!/usr/bin/env python3
"""Push Candidate C010 (x138 Flow Fusion) to Kaggle as a GPU kernel.

Kernel: taeyangg4/biohub-c010-x138-flow-fusion
Attached Datasets:
- pilkwang/biohub-deepcenter-unet3d-center-prior-v1
- pilkwang/biohub-temporal-unet3d-seed314159-v1
- pilkwang/biohub-tracking-support-pack-50ep-v1

CRITICAL USER CONSTRAINT: Only run when explicitly authorized by the user.
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
logger = logging.getLogger("push_c010")

REPO_ROOT = Path(__file__).resolve().parents[1]
C010_DIR = REPO_ROOT / "experiments" / "candidates" / "c010_x138_flow_fusion"


def main() -> int:
    logger.info(f"Checking C010 candidate folder at {C010_DIR}...")
    nb_path = C010_DIR / "biohub-c010-x138-flow-fusion.ipynb"
    meta_path = C010_DIR / "kernel-metadata.json"

    if not nb_path.exists():
        logger.error(f"Missing notebook: {nb_path}")
        return 1
    if not meta_path.exists():
        logger.error(f"Missing metadata: {meta_path}")
        return 1

    logger.info("Running offline verification suite before push...")
    res = subprocess.run([sys.executable, "src/verify_c010.py"], cwd=str(REPO_ROOT))
    if res.returncode != 0:
        logger.error("C010 offline verification failed! Aborting push.")
        return res.returncode

    logger.info("Pushing Candidate C010 to Kaggle...")
    cmd = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(C010_DIR),
    ]
    push_res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if push_res.returncode == 0:
        logger.info("Candidate C010 successfully pushed to Kaggle!")
    else:
        logger.error(f"Kaggle kernel push failed with code {push_res.returncode}")
    return push_res.returncode


if __name__ == "__main__":
    sys.exit(main())
