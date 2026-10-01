from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visible_test" / "test"
MODEL_DIR = ROOT / "artifacts" / "hengck_point_detector"
CKPT = MODEL_DIR / "00000030.pth"
MODEL_PY = MODEL_DIR / "model_v5.py"
BASELINE = ROOT / "artifacts" / "reyhan_0946_output" / "submission.csv"
OUTDIR = ROOT / "experiments" / "strongunet_probe"
OUTDIR.mkdir(parents=True, exist_ok=True)

STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
MATCH_UM = 7.0
BATCH = 10


def load_model_class():
    spec = importlib.util.spec_from_file_location("hengck_model_v5", MODEL_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load model_v5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StrongUNet3D3Level


def metadata_quantiles(sample_dir: Path) -> tuple[float, float]:
    meta = json.loads((sample_dir / "zarr.json").read_text(encoding="utf-8"))
    q = meta["attributes"]["image_statistics"]["quantiles"]
    return float(q["0.001"]), float(q["0.999"])


@torch.inference_mode()
def infer_sample(model, sample_dir: Path, device: torch.device) -> pd.DataFrame:
    arr = zarr.open_group(str(sample_dir), mode="r")["0"]
    q_low, q_high = metadata_quantiles(sample_dir)
    rows = []
    total_t = int(arr.shape[0])
    for t0 in range(0, total_t, BATCH):
        t1 = min(total_t, t0 + BATCH)
        vol = arr[t0:t1, :, ::4, ::4].astype(np.float32)
        vol = (vol - q_low) / (q_high - q_low + 1e-6)
        vol = np.clip(vol, 0.0, None)
        x = torch.from_numpy(vol).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _, logits = model(x)
        prob = torch.sigmoid(logits.float()).unsqueeze(1)
        pooled = F.max_pool3d(prob, kernel_size=3, stride=1, padding=1)
        keep = (prob >= pooled) & (prob >= min(THRESHOLDS))
        for b in range(t1 - t0):
            idx = keep[b, 0].nonzero(as_tuple=False)
            if len(idx) == 0:
                continue
            p = prob[b, 0, idx[:, 0], idx[:, 1], idx[:, 2]]
            idx_np = idx.cpu().numpy().astype(np.float32)
            p_np = p.cpu().numpy().astype(np.float32)
            # Model works on Y/X downsampled by 4. Convert back to original voxels.
            idx_np[:, 1:] *= 4.0
            for (z, y, x0), score in zip(idx_np, p_np, strict=True):
                rows.append((t0 + b, z, y, x0, score))
        print(f"{sample_dir.stem}: {t1}/{total_t} frames", flush=True)
        del x, prob, pooled, keep, vol
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["t", "z", "y", "x", "p"])


def match_counts(pred: pd.DataFrame, ref: pd.DataFrame) -> dict[str, float | int]:
    matched = 0
    distances = []
    for t in sorted(set(pred["t"].astype(int)) | set(ref["t"].astype(int))):
        p = pred[pred["t"].astype(int).eq(t)][["z", "y", "x"]].to_numpy(np.float32)
        r = ref[ref["t"].astype(int).eq(t)][["z", "y", "x"]].to_numpy(np.float32)
        if len(p) == 0 or len(r) == 0:
            continue
        p_um = p * SCALE_UM[None, :]
        r_um = r * SCALE_UM[None, :]
        d = np.linalg.norm(r_um[:, None, :] - p_um[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(d)
        accepted = d[rr, cc] <= MATCH_UM
        matched += int(accepted.sum())
        distances.extend(d[rr[accepted], cc[accepted]].tolist())
    return {
        "pred_nodes": int(len(pred)),
        "ref_nodes": int(len(ref)),
        "matched": int(matched),
        "pred_only": int(len(pred) - matched),
        "ref_only": int(len(ref) - matched),
        "pred_match_fraction": float(matched / max(len(pred), 1)),
        "ref_coverage": float(matched / max(len(ref), 1)),
        "mean_match_um": float(np.mean(distances)) if distances else float("nan"),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    Model = load_model_class()
    model = Model(
        in_channels=1,
        channels=(64, 128, 256),
        node_channels=1,
        gradient_checkpointing=False,
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(
        "checkpoint",
        {k: ckpt.get(k) for k in (
            "epoch", "valid_clean_f1", "valid_aug_f1",
            "valid_kaggle_clean_recall", "valid_kaggle_aug_recall", "best_valid_f1"
        )},
        flush=True,
    )

    base = pd.read_csv(BASELINE)
    summaries = []
    for stem in STEMS:
        sample_dir = DATA_ROOT / f"{stem}.zarr"
        pred_all = infer_sample(model, sample_dir, device)
        pred_all.to_parquet(OUTDIR / f"{stem}_peaks.parquet", index=False)
        ref = base[(base["dataset"] == stem) & (base["row_type"] == "node")][
            ["t", "z", "y", "x"]
        ].copy()
        for threshold in THRESHOLDS:
            pred = pred_all[pred_all["p"] >= threshold]
            stats = match_counts(pred, ref)
            summaries.append({"dataset": stem, "threshold": threshold, **stats})
            print(stem, threshold, stats, flush=True)

    result = pd.DataFrame(summaries)
    result.to_csv(OUTDIR / "threshold_vs_public0946.csv", index=False)
    agg = (
        result.groupby("threshold", as_index=False)
        .agg(
            pred_nodes=("pred_nodes", "sum"),
            ref_nodes=("ref_nodes", "sum"),
            matched=("matched", "sum"),
            pred_only=("pred_only", "sum"),
            ref_only=("ref_only", "sum"),
        )
    )
    agg["pred_match_fraction"] = agg["matched"] / agg["pred_nodes"]
    agg["ref_coverage"] = agg["matched"] / agg["ref_nodes"]
    agg.to_csv(OUTDIR / "aggregate_vs_public0946.csv", index=False)
    print("AGGREGATE")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
