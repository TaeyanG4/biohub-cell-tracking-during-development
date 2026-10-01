from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visible_test" / "test"
MODEL_DIR = ROOT / "artifacts" / "hengck_point_detector"
ANCHOR = ROOT / "experiments" / "exp_edge_tta_det096" / "output_v2" / "submission.csv"
OUT = ROOT / "experiments" / "strongunet_visible_probe"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))
from model_v5 import StrongUNet3D3Level  # noqa: E402


STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
if os.environ.get("PROBE_STEMS"):
    STEMS = tuple(x.strip() for x in os.environ["PROBE_STEMS"].split(",") if x.strip())
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
if os.environ.get("PROBE_THRESHOLDS"):
    THRESHOLDS = tuple(float(x.strip()) for x in os.environ["PROBE_THRESHOLDS"].split(",") if x.strip())
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)


def peaks(logits: torch.Tensor, threshold: float) -> list[np.ndarray]:
    prob = torch.sigmoid(logits.float()).unsqueeze(1)
    pooled = F.max_pool3d(prob, kernel_size=3, stride=1, padding=1)
    keep = (prob >= pooled) & (prob >= threshold)
    result = []
    for b in range(len(prob)):
        idx = keep[b, 0].nonzero(as_tuple=False)
        if len(idx):
            p = prob[b, 0, idx[:, 0], idx[:, 1], idx[:, 2]]
            result.append(
                np.column_stack(
                    [idx.detach().cpu().numpy(), p.detach().cpu().numpy()]
                ).astype(np.float32)
            )
        else:
            result.append(np.empty((0, 4), dtype=np.float32))
    return result


def anchor_nodes(df: pd.DataFrame, stem: str) -> dict[int, np.ndarray]:
    g = df[(df.dataset == stem) & (df.row_type == "node")]
    return {
        int(t): part[["z", "y", "x"]].to_numpy(np.float32)
        for t, part in g.groupby("t")
    }


device = torch.device("cuda")
model = StrongUNet3D3Level(
    in_channels=1,
    channels=(64, 128, 256),
    node_channels=1,
    gradient_checkpointing=False,
).to(device)
ckpt = torch.load(MODEL_DIR / "00000030.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print("checkpoint", {k: ckpt.get(k) for k in ("epoch", "best_valid_f1", "valid_kaggle_clean_recall")})

anchor_df = pd.read_csv(ANCHOR)
summary = []
all_points = []

for stem in STEMS:
    zarr_path = DATA / f"{stem}.zarr"
    group = zarr.open_group(str(zarr_path), mode="r")
    arr = group["0"]
    anchors = anchor_nodes(anchor_df, stem)
    started = time.perf_counter()
    counts = {thr: 0 for thr in THRESHOLDS}
    matched = {thr: 0 for thr in THRESHOLDS}
    novel = {thr: 0 for thr in THRESHOLDS}
    anchor_total = {thr: 0 for thr in THRESHOLDS}
    anchor_covered = {thr: 0 for thr in THRESHOLDS}

    for start in range(0, int(arr.shape[0]), 10):
        stop = min(start + 10, int(arr.shape[0]))
        vol = np.asarray(arr[start:stop, ::1, ::4, ::4], dtype=np.float32)
        q_low = float(np.quantile(vol, 0.001))
        q_high = float(np.quantile(vol, 0.999))
        vol = np.clip((vol - q_low) / (q_high - q_low + 1e-6), 0.0, None)
        x = torch.from_numpy(vol).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _, logits = model(x)
        for thr in THRESHOLDS:
            batch = peaks(logits, thr)
            for offset, rows in enumerate(batch):
                t = start + offset
                counts[thr] += len(rows)
                if not len(rows):
                    continue
                # Detector runs on YX/4 data. Restore native XY coordinates.
                coords = rows[:, :3].copy()
                coords[:, 1:] *= 4.0
                probs = rows[:, 3]
                ref = anchors.get(t, np.empty((0, 3), dtype=np.float32))
                anchor_total[thr] += len(ref)
                if len(ref):
                    tree = cKDTree(ref * SCALE_UM)
                    dist, _ = tree.query(coords * SCALE_UM, k=1)
                    is_match = dist <= 7.0
                    det_tree = cKDTree(coords * SCALE_UM)
                    anchor_dist, _ = det_tree.query(ref * SCALE_UM, k=1)
                    anchor_covered[thr] += int((anchor_dist <= 7.0).sum())
                else:
                    dist = np.full(len(coords), np.inf)
                    is_match = np.zeros(len(coords), dtype=bool)
                matched[thr] += int(is_match.sum())
                novel[thr] += int((~is_match).sum())
                if thr == 0.20:
                    for xyz, prob, d in zip(coords, probs, dist):
                        all_points.append(
                            {
                                "dataset": stem,
                                "t": t,
                                "z": float(xyz[0]),
                                "y": float(xyz[1]),
                                "x": float(xyz[2]),
                                "prob": float(prob),
                                "nearest_anchor_um": float(d),
                            }
                        )
        del x, logits
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    for thr in THRESHOLDS:
        summary.append(
            {
                "dataset": stem,
                "threshold": thr,
                "detections": counts[thr],
                "within_7um_of_anchor": matched[thr],
                "novel_gt7um": novel[thr],
                "novel_fraction": novel[thr] / max(counts[thr], 1),
                "anchor_nodes": anchor_total[thr],
                "anchor_covered_7um": anchor_covered[thr],
                "anchor_coverage_7um": anchor_covered[thr] / max(anchor_total[thr], 1),
                "seconds": elapsed,
            }
        )
    print(stem, "seconds", round(elapsed, 2), "counts", counts, "novel", novel, flush=True)

pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(all_points).to_csv(OUT / "threshold020_points.csv", index=False)
(OUT / "checkpoint_metrics.json").write_text(
    json.dumps(
        {
            k: ckpt.get(k)
            for k in (
                "epoch",
                "valid_loss",
                "valid_clean_f1",
                "valid_aug_f1",
                "valid_kaggle_clean_recall",
                "valid_kaggle_aug_recall",
                "best_valid_f1",
            )
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("STRONGUNET_PROBE_COMPLETE", OUT, flush=True)
