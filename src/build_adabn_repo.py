#!/usr/bin/env python3
"""AdaBN test-time adaptation for the detection UNets (HANDOFF section 24).

The hidden test movies come from other embryos than the two the public UNets were trained on. Before
inference on each movie, this patch recomputes the BatchNorm3d running statistics of the primary and the
secondary UNet from that movie's own frames (weights untouched; cumulative average over all windows), then
runs the unchanged pipeline. Only BatchNorm modules are touched (the node transformer uses LayerNorm and
Dropout, which stay in eval mode). Per movie the original statistics are restored before adapting again.

  BIOHUB_ADABN=1             enable (default 0 = byte-identical behaviour)
  BIOHUB_ADABN_MAX_WINDOWS=N adapt on at most N evenly spaced windows (0 = every window)
  BIOHUB_ADABN_SECONDARY=1   also adapt the secondary (seed 314159) UNet (default 1)

    python src/build_adabn_repo.py [--source tmp/c022_output/tracking_repo] [--dest tmp/c030_adabn/tracking_repo]
    python src/run_kaggle_predict_local.py --repo tmp/c030_adabn/tracking_repo ... --env BIOHUB_ADABN=1
The same two text patches are what `src/build_c030_candidate.py` applies inside the notebook (cell 4).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODULE_ANCHOR = "_LOWDET: list = []\n"
MODULE_BLOCK = """_LOWDET: list = []
_ADABN = os.environ.get('BIOHUB_ADABN', '0') != '0'
_ADABN_MAX_WINDOWS = int(os.environ.get('BIOHUB_ADABN_MAX_WINDOWS', '0') or 0)
_ADABN_SECONDARY = os.environ.get('BIOHUB_ADABN_SECONDARY', '1') != '0'
_ADABN_ORIG: dict = {}


def _adabn_adapt(model, label, zarr_arr, window_starts, W, target_shape, downsample, q_low, q_high, device):
    \"\"\"Recompute the BatchNorm running statistics of `model`'s UNet on this movie (AdaBN).\"\"\"
    import torch.nn as _nn
    bns = [m for m in model.modules() if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d))]
    if not bns:
        print(f'ADABN[{label}]: no BatchNorm layers, nothing to adapt', flush=True)
        return
    key = id(model)
    if key not in _ADABN_ORIG:
        _ADABN_ORIG[key] = [(m.running_mean.detach().clone(), m.running_var.detach().clone(),
                             m.num_batches_tracked.detach().clone(), m.momentum) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the adaptation windows
        m.train()
    starts = list(window_starts)
    if _ADABN_MAX_WINDOWS > 0 and len(starts) > _ADABN_MAX_WINDOWS:
        step = len(starts) / float(_ADABN_MAX_WINDOWS)
        starts = [starts[int(i * step)] for i in range(_ADABN_MAX_WINDOWS)]
    with torch.no_grad():
        for ws in starts:
            imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample) for t in range(ws, ws + W)])
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
            model.encode(imgs)
            del imgs
    for m, (_rm, _rv, _nb, mom) in zip(bns, _ADABN_ORIG[key]):
        m.momentum = mom
        m.eval()
    mean_shift = float(sum((m.running_mean - rm).abs().mean() for m, (rm, _rv, _nb, _mom) in zip(bns, _ADABN_ORIG[key])) / len(bns))
    print(f'ADABN[{label}]: adapted {len(bns)} BatchNorm layers on {len(starts)} windows; mean |running_mean shift| {mean_shift:.4f}', flush=True)


def _adabn_restore(model):
    key = id(model)
    if key not in _ADABN_ORIG:
        return
    import torch.nn as _nn
    bns = [m for m in model.modules() if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d))]
    for m, (rm, rv, nb, mom) in zip(bns, _ADABN_ORIG[key]):
        m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb); m.momentum = mom; m.eval()
"""

LOOP_ANCHOR = "    for ws in tqdm(\n        window_starts,\n"
LOOP_BLOCK = """    if _ADABN:
        _adabn_adapt(model, 'primary', zarr_arr, window_starts, W, target_shape, downsample, q_low, q_high, device)
        if secondary_model is not None and _ADABN_SECONDARY:
            _adabn_adapt(secondary_model, 'secondary', zarr_arr, window_starts, W, target_shape, downsample, q_low, q_high, device)

    for ws in tqdm(
        window_starts,
"""

RETURN_ANCHOR = "    return coords, all_edges\n"
RETURN_BLOCK = """    if _ADABN:
        _adabn_restore(model)
        if secondary_model is not None:
            _adabn_restore(secondary_model)
    return coords, all_edges
"""

PATCHES = [("module", MODULE_ANCHOR, MODULE_BLOCK), ("loop", LOOP_ANCHOR, LOOP_BLOCK), ("return", RETURN_ANCHOR, RETURN_BLOCK)]


def patch_text(text: str) -> str:
    for label, old, new in PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "tmp" / "c022_output" / "tracking_repo")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "tmp" / "c030_adabn" / "tracking_repo")
    args = ap.parse_args()
    if args.dest.exists():
        shutil.rmtree(args.dest)
    for part in ("scripts", "src"):
        shutil.copytree(args.source / part, args.dest / part, ignore=shutil.ignore_patterns("__pycache__"))
    script = args.dest / "scripts" / "predict_unet_transformer.py"
    text = patch_text(script.read_text(encoding="utf-8"))
    script.write_text(text, encoding="utf-8")
    compile(text, str(script), "exec")
    print(f"wrote {script} ({len(PATCHES)} patches)")


if __name__ == "__main__":
    main()
