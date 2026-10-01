"""C061 known-point axial localization; no image IO, graph edits, or fitting.

Every input contains one supplied known point during supervised learning. Nine
classes describe that point's z coordinate relative to the integer crop center;
they are not detection/background labels for the other cells in the image.
The caller owns whole-embryo folds, direct GEFF provenance, real crop support,
and uniform-source-movie then uniform-known-point sampling.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RECIPE = {
    "version": 1,
    "seed": 6101,
    "input_channels": 1,
    "crop_zyx": [13, 49, 49],
    "stored_crop_zyx": [21, 49, 49],
    "voxel_um_zyx": [1.625, 0.40625, 0.40625],
    "z_offsets_vox": list(range(-4, 5)),
    "shift_geometry": "slice expanded z at 4 + shift; integer-center target = -shift",
    "image_normalization": "existing movie quantiles 0.001/0.999; clip [0,3]",
    "widths": [8, 16, 24],
    "strides_zyx": [[1, 2, 2], [2, 2, 2], [2, 2, 2]],
    "kernel_zyx": [3, 3, 3],
    "padding_zyx": [1, 1, 1],
    "group_norm_groups": 4,
    "spatial_pool_zyx": [4, 7, 7],
    "hidden": 64,
    "last_layer_init": "zero weights and bias; no center prior; exact all-class tie",
    "target": "known GEFF point z minus integer input-crop-center z, in native voxels",
    "loss": "conditional cross entropy; fractional z uses linear adjacent-bin soft labels",
    "unknown_cells": "never detection/background negatives",
    "decode": "unique argmax z class -4..4; exact ties return zero and tied flag",
    "augmentation": "uniform independent y/x reflections per example; no z reflection",
    "steps": 1200,
    "batch_size": 36,
    "examples_per_offset_per_batch": 4,
    "batch_offsets": "exactly four of each of nine offsets, randomly permuted",
    "training_sampler": "uniform source movie, then uniform eligible known GEFF point",
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "schedule": "CosineAnnealingLR over fixed 1200 steps",
    "checkpoint": "final step only; no evaluation-based selection",
    "precision": "FP32; AMP and TF32 off",
}
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _check_images(images: torch.Tensor, *, expanded: bool = False):
    shape = RECIPE["stored_crop_zyx"] if expanded else RECIPE["crop_zyx"]
    expected = (1, *shape)
    if images.ndim != 5 or tuple(images.shape[1:]) != expected or len(images) == 0:
        raise ValueError(f"Expected nonempty (N,{expected}), got {tuple(images.shape)}")
    if images.dtype != torch.float32:
        raise ValueError("C061 images must be float32")


def _check_logits(logits: torch.Tensor):
    if logits.ndim != 2 or logits.shape[1] != 9 or len(logits) == 0:
        raise ValueError("Expected nonempty (N,9) conditional axial logits")
    if logits.dtype != torch.float32:
        raise ValueError("C061 logits must be float32")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite C061 logits")


class AxialLocalizer(nn.Module):
    """Spatial CNN and flattened whole-crop head, with nine conditional z logits.

    The exact convolution geometry is 13x25x25 -> 7x13x13 -> 4x7x7.
    Pooling preserves that final lattice; the dense head mixes its full spatial
    field. There is no BatchNorm, dropout, coordinate prior, or output scaling.
    """

    def __init__(self):
        super().__init__()
        layers = []
        previous = 1
        for width, stride in zip(RECIPE["widths"], RECIPE["strides_zyx"]):
            layers.extend([
                nn.Conv3d(previous, width, 3, stride=tuple(stride), padding=1),
                nn.GroupNorm(RECIPE["group_norm_groups"], width),
                nn.SiLU(),
            ])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(tuple(RECIPE["spatial_pool_zyx"]))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(24 * 4 * 7 * 7, RECIPE["hidden"]), nn.SiLU(),
            nn.Linear(RECIPE["hidden"], 9),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        _check_images(crops)
        features = self.features(crops)
        if tuple(features.shape[1:]) != (24, 4, 7, 7):
            raise AssertionError("C061 convolution geometry drift")
        return self.head(self.pool(features))


@torch.no_grad()
def shifted_batch(expanded: torch.Tensor, shifts: torch.Tensor):
    """Return (real cropped images, integer known-point target_z_vox).

    Expanded crops are centered at a known integer GEFF point at array z=10.
    A positive shift moves the input center toward increasing image z, so the
    same point is at offset -shift. The caller must verify real image support;
    this function only slices existing pixels and never interpolates or pads.
    For fractional GEFF z, add its separately recorded center residual to the
    returned target and let loss validate support; never clip that target.
    """
    _check_images(expanded, expanded=True)
    if shifts.shape != (len(expanded),) or shifts.device != expanded.device:
        raise ValueError("Shifts must have shape (N,) and share the image device")
    if shifts.dtype not in _INTEGER_DTYPES:
        raise ValueError("Shifts must be integer native z voxels")
    shifts = shifts.to(torch.int64)
    if ((shifts < -4) | (shifts > 4)).any():
        raise ValueError("C061 z shifts must be within [-4,4]")
    starts = 4 + shifts
    cropped = torch.stack([expanded[i, :, start:start + 13, :, :]
                           for i, start in enumerate(starts.tolist())])
    return cropped, -shifts


def balanced_shifts(*, generator: torch.Generator | None = None, device=None):
    """Exactly four of each offset in a randomly permuted fixed 36-row batch."""
    offsets = torch.arange(-4, 5, dtype=torch.int64, device=device).repeat_interleave(4)
    order = torch.randperm(RECIPE["batch_size"], generator=generator, device=offsets.device)
    return offsets[order]


@torch.no_grad()
def reflect_xy_batch(crops: torch.Tensor, *, generator: torch.Generator | None = None):
    """Return (images, N-by-2 y/x reflection flags); the z targets are unchanged."""
    _check_images(crops)
    reflected = crops.clone()
    masks = torch.randint(0, 2, (len(crops), 2), device=crops.device,
                          generator=generator).bool()
    for column, tensor_axis in enumerate((3, 4)):
        selected = masks[:, column]
        if selected.any():
            reflected[selected] = torch.flip(reflected[selected], (tensor_axis,))
    return reflected, masks


def target_distribution(target_z_vox: torch.Tensor, *, n: int | None = None):
    """Linear adjacent-class interpolation for finite known targets in [-4,4].

    Integer targets become exact one-hot labels. Fractional points are encoded
    explicitly; out-of-support values, including fractional endpoint overflow,
    are rejected instead of clipped or silently rounded to an integer label.
    """
    if target_z_vox.ndim != 1 or len(target_z_vox) == 0:
        raise ValueError("Known axial targets must have nonempty shape (N,)")
    if n is not None and len(target_z_vox) != n:
        raise ValueError("Known target count must equal logit count")
    if target_z_vox.dtype not in (*_INTEGER_DTYPES, torch.float32, torch.float64):
        raise ValueError("Known axial targets must be integer or full-precision floats")
    if not torch.isfinite(target_z_vox).all() or ((target_z_vox < -4) | (target_z_vox > 4)).any():
        raise ValueError("Known axial targets must be finite and within [-4,4]; no clipping")
    coordinate = target_z_vox.to(torch.float32) + 4
    lower = coordinate.floor().to(torch.int64)
    upper = coordinate.ceil().to(torch.int64)
    upper_weight = coordinate - lower
    target = torch.zeros((len(coordinate), 9), device=coordinate.device, dtype=torch.float32)
    target.scatter_add_(1, lower[:, None], (1 - upper_weight)[:, None])
    target.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return target


def loss(logits: torch.Tensor, target_z_vox: torch.Tensor) -> torch.Tensor:
    """Known-point conditional CE; no claim that other image cells are absent."""
    _check_logits(logits)
    if target_z_vox.device != logits.device:
        raise ValueError("Known targets and logits must share a device")
    target = target_distribution(target_z_vox, n=len(logits))
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def decode(logits: torch.Tensor):
    """Unique axial mode with explicit zero-on-tie, not a confidence threshold."""
    _check_logits(logits)
    best = logits.topk(2, dim=1)
    tied = best.values[:, 0] == best.values[:, 1]
    class_index = logits.argmax(dim=1)
    modal_z = class_index - 4
    return {
        "z_vox": torch.where(tied, torch.zeros_like(modal_z), modal_z),
        "tied": tied,
        "class_index": class_index,
        "probability": logits.softmax(dim=1),
    }


def make_optimizer(model: AxialLocalizer):
    """Construct the fixed optimizer/schedule; this does not perform any fit."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                 weight_decay=RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE["steps"])
    return optimizer, scheduler
