from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from extract_hoct_r3_compact import SCALE_ZYX, extract_one, load_gt


ROOT = Path(__file__).resolve().parents[1]
DATASET = "6bba_05b6850b"
IMAGE_ROOT = ROOT / "data" / "visible_test" / "test"
GT_ROOT = ROOT / "data" / "visible_gt" / "train"
CKPT = ROOT / "artifacts" / "hoct_general_v1_research" / "general_v1.pt"
JITTER_STD_UM = np.asarray([1.76, 0.96, 1.00], dtype=np.float32)


def gt_nodes() -> pd.DataFrame:
    gt = load_gt(GT_ROOT / f"{DATASET}.geff")
    frame = gt.node_attrs().to_pandas()[["node_id", "t", "z", "y", "x"]].copy()
    frame["dataset"] = DATASET
    frame["row_type"] = "node"
    return frame


def jitter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(20260914)
    noise_um = rng.normal(0.0, JITTER_STD_UM, size=(len(out), 3)).astype(np.float32)
    norm = np.linalg.norm(noise_um, axis=1)
    over = norm > 6.0
    if over.any():
        noise_um[over] *= (6.0 / norm[over])[:, None]
    noise_px = noise_um / np.asarray(SCALE_ZYX, dtype=np.float32)
    out.loc[:, ["z", "y", "x"]] = out[["z", "y", "x"]].to_numpy(dtype=np.float32) + noise_px
    return out


def main() -> None:
    model = torch.jit.load(str(CKPT), map_location="cpu").eval()
    clean = gt_nodes()
    print("gt_nodes", len(clean))
    for variant, frame in [("clean", clean), ("jitter", jitter(clean))]:
        t0 = time.perf_counter()
        result = extract_one(DATASET, frame, IMAGE_ROOT, GT_ROOT, model, hard_k=8)
        dt = time.perf_counter() - t0
        print(
            variant,
            "seconds", round(dt, 3),
            "rows", len(result[1]),
            "positives", int(result[1].sum()),
            "stats", result[-1],
        )


if __name__ == "__main__":
    main()
