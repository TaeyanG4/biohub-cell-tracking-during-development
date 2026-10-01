from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visible_test" / "test"
MODEL_DIR = ROOT / "artifacts" / "hengck_point_detector"
CKPT = MODEL_DIR / "00000030.pth"
MODEL_PY = MODEL_DIR / "model_v5.py"
OUTDIR = ROOT / "experiments" / "candidates" / "c002_node_rescue" / "gpu_peaks"

STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)

MIN_THRESHOLD = 0.10
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
    rows: list[tuple[int, float, float, float, float]] = []
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
        keep = (prob >= pooled) & (prob >= MIN_THRESHOLD)

        for b in range(t1 - t0):
            idx = keep[b, 0].nonzero(as_tuple=False)
            if len(idx) == 0:
                continue
            p = prob[b, 0, idx[:, 0], idx[:, 1], idx[:, 2]]
            idx_np = idx.cpu().numpy().astype(np.float32)
            p_np = p.cpu().numpy().astype(np.float32)
            idx_np[:, 1:] *= 4.0
            for (z, y, x0), score in zip(idx_np, p_np, strict=True):
                rows.append((t0 + b, z, y, x0, score))

        print(f"{sample_dir.stem}: {t1}/{total_t} frames peaks={len(rows)}", flush=True)
        del x, prob, pooled, keep, vol

    return pd.DataFrame(rows, columns=["t", "z", "y", "x", "p"])


def main() -> None:
    # Keep CPU pressure minimal while the workstation is already CPU-saturated.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    OUTDIR.mkdir(parents=True, exist_ok=True)
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

    print("device", torch.cuda.get_device_name(0), flush=True)
    print("checkpoint_epoch", ckpt.get("epoch"), flush=True)
    print("min_threshold", MIN_THRESHOLD, "batch", BATCH, flush=True)

    summary = []
    for stem in STEMS:
        sample_dir = DATA_ROOT / f"{stem}.zarr"
        pred = infer_sample(model, sample_dir, device)
        out = OUTDIR / f"{stem}_peaks.parquet"
        pred.to_parquet(out, index=False)
        summary.append(
            {
                "dataset": stem,
                "frames": int(pred["t"].nunique()) if len(pred) else 0,
                "peaks": int(len(pred)),
                "min_p": float(pred["p"].min()) if len(pred) else None,
                "max_p": float(pred["p"].max()) if len(pred) else None,
                "output": str(out.relative_to(ROOT)),
            }
        )
        print("saved", out, "rows", len(pred), flush=True)

    pd.DataFrame(summary).to_csv(OUTDIR / "summary.csv", index=False)
    print("saved", OUTDIR / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
