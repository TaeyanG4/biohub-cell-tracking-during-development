"""C060 conditional spatial posterior for one supplied predicted trajectory.

All three channels are real crops centered on their own predecessor/current/
successor prediction. This module does not choose tracks, label unknown cells,
read images, select folds, change graphs, or apply the caller's ownership guard.
Two valid convolutions retain spatial alternatives and use only real support.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F


RECIPE = {
    "version": 1,
    "seed": 6001,
    "crop_zyx": [13, 49, 49],
    "stored_crop_zyx": [15, 57, 57],
    "input_channels": 3,
    "channels": "own-centered predicted predecessor, current, successor",
    "voxel_um_zyx": [1.625, 0.40625, 0.40625],
    "widths": [8, 16],
    "valid_kernels_zyx": [[3, 7, 7], [3, 7, 7]],
    "normalization": "LayerNorm over channels independently at each voxel",
    "image_normalization": "existing movie quantiles 0.001/0.999; clip [0,3]",
    "output_zyx": [9, 37, 37],
    "max_shift_um": 7.0,
    "prior_sigma_um": 7.0 / 3.0,
    "target_sigma_um_zyx": [1.625, 0.40625, 0.40625],
    "target": "conditional known current-point Gaussian, one native voxel sigma per axis",
    "loss": "conditional distribution cross entropy inside physical 7um sphere",
    "unknown_cells": "never detection/background negatives; caller selects known point targets",
    "last_layer_init": "zero weights, no bias; exact centered symmetric prior",
    "jitter_vox_zyx": [1, 4, 4],
    "jitter": "one shared integer translation of all three real views; reject outside7um targets",
    "reflections": "uniform joint z/y/x reflections; odd-grid target d -> -d",
    "inference": "average eight inverse-reflected logits, then conditional softmax",
    "abstention": "all eight aligned maps must have same unique integer mode; caller adds strict Voronoi ownership",
    "photometric_augmentation": False,
    "steps": 1200,
    "batch_size": 32,
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "schedule": "CosineAnnealingLR over fixed 1200 steps",
    "checkpoint": "final step only; no evaluation-based selection",
    "precision": "FP32; AMP and TF32 off",
}
REFLECTIONS = tuple(tuple(a for a, enabled in enumerate(bits) if enabled)
                    for bits in itertools.product((False, True), repeat=3))


def _geometry(device=None, dtype=torch.float32):
    axes = [torch.arange(-(n // 2), n // 2 + 1, device=device, dtype=dtype)
            for n in RECIPE["output_zyx"]]
    vox = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    um = vox * vox.new_tensor(RECIPE["voxel_um_zyx"])
    support = um.square().sum(-1) <= RECIPE["max_shift_um"] ** 2
    prior = -um.square().sum(-1) / (2 * RECIPE["prior_sigma_um"] ** 2)
    return vox, um, support, prior


def _check_crops(crops, expanded=False):
    expected = (3, *(RECIPE["stored_crop_zyx"] if expanded else RECIPE["crop_zyx"]))
    if crops.ndim != 5 or tuple(crops.shape[1:]) != expected:
        raise ValueError(f"Expected (N,{expected}), got {tuple(crops.shape)}")
    if crops.dtype != torch.float32:
        raise ValueError("C060 requires FP32 crops")


def _check_targets(target_um, n=None):
    if target_um.ndim != 2 or target_um.shape[1] != 3 or (n is not None and len(target_um) != n):
        raise ValueError("Known point targets must have shape (N,3)")
    if not torch.isfinite(target_um).all() or (target_um.norm(dim=1) > RECIPE["max_shift_um"] + 1e-5).any():
        raise ValueError("Known targets must be finite and within the fixed 7um identity gate")


class ChannelNorm(nn.Module):
    """Normalize only channels, never pool across positions or samples."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class SpatialLocalizer(nn.Module):
    """Fully convolutional conditional location logits on a valid 9x37x37 grid."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 8, (3, 7, 7), padding=0), ChannelNorm(8), nn.SiLU(),
            nn.Conv3d(8, 16, (3, 7, 7), padding=0), ChannelNorm(16), nn.SiLU(),
        )
        self.head = nn.Conv3d(16, 1, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        vox, um, support, prior = _geometry()
        self.register_buffer("offset_vox", vox)
        self.register_buffer("offset_um", um)
        self.register_buffer("support", support)
        self.register_buffer("log_prior", prior)

    def forward(self, crops):
        _check_crops(crops)
        logits = self.head(self.features(crops))[:, 0]
        if tuple(logits.shape[1:]) != tuple(RECIPE["output_zyx"]):
            raise AssertionError("Valid-convolution geometry drift")
        return (logits + self.log_prior).masked_fill(~self.support, -torch.inf)

    @torch.no_grad()
    def predict(self, crops, *, eight_reflections=True):
        """Propose integer voxels; caller MUST additionally enforce spatial ownership.

        The accepted flag is only the registered unique-mode agreement guard,
        not evidence of calibrated biological confidence. No posterior mean is
        used as a correction. A tied/discordant mode returns exact zero shift.
        eight_reflections=False is available for controls/diagnostics only.
        """
        averaged, aligned = group_average_logits(self, crops, eight_reflections=eight_reflections)
        result = decode_modes(averaged, aligned)
        result["eight_reflections"] = bool(eight_reflections)
        return result


def gaussian_target(target_um):
    """Normalized point-location targets; this is not a cell/background map."""
    _check_targets(target_um)
    _, grid_um, support, _ = _geometry(target_um.device, target_um.dtype)
    scaled = ((grid_um[None] - target_um[:, None, None, None, :]) /
              target_um.new_tensor(RECIPE["target_sigma_um_zyx"]))
    score = (-0.5 * scaled.square().sum(-1)).masked_fill(~support, -torch.inf)
    return score.flatten(1).softmax(1).reshape_as(score)


def conditional_loss(logits, target_um):
    """Known trajectory point cross entropy, with explicit masked arithmetic."""
    _check_targets(target_um, len(logits))
    if tuple(logits.shape[1:]) != tuple(RECIPE["output_zyx"]):
        raise ValueError("Unexpected logit lattice")
    if logits.device != target_um.device:
        raise ValueError("Logits and target coordinates must share a device")
    _, _, support, _ = _geometry(logits.device, logits.dtype)
    if not torch.isfinite(logits[:, support]).all():
        raise FloatingPointError("Nonfinite supported logits")
    log_probability = logits.masked_fill(~support, -torch.inf).flatten(1).log_softmax(1).reshape_as(logits)
    # Avoid the undefined zero * -inf operation outside the support sphere.
    log_probability = log_probability.masked_fill(~support, 0.0)
    target = gaussian_target(target_um)
    return -(target * log_probability).flatten(1).sum(1).mean()


def reflect_crops(crops, axes):
    """Jointly reflect spatial axes0/1/2 across every temporal channel."""
    axes = tuple(axes)
    if len(set(axes)) != len(axes) or any(a not in (0, 1, 2) for a in axes):
        raise ValueError("Reflection axes must be distinct members of z/y/x = 0/1/2")
    return torch.flip(crops, tuple(a + 2 for a in axes)) if axes else crops


def reflect_targets(target_um, axes):
    """Odd centered crops require exact sign inversion, with no half-voxel shift."""
    sign = target_um.new_ones(3)
    for axis in axes:
        sign[axis] = -1
    return target_um * sign


@torch.no_grad()
def reflect_batch(crops, target_um, *, generator=None):
    """One uniformly random reflection-group member per example; returns masks."""
    _check_crops(crops)
    _check_targets(target_um, len(crops))
    if target_um.device != crops.device:
        raise ValueError("Crops and targets must share a device")
    masks = torch.randint(0, 2, (len(crops), 3), device=crops.device, generator=generator).bool()
    output = crops.clone()
    for axes in REFLECTIONS:
        wanted = masks.new_tensor([axis in axes for axis in range(3)])
        selected = (masks == wanted).all(1)
        if selected.any():
            output[selected] = reflect_crops(crops[selected], axes)
    return output, target_um * torch.where(masks, -1.0, 1.0), masks


@torch.no_grad()
def jitter_batch(expanded, target_um, *, generator=None, enabled=True):
    """Shared integer centre translation of three real expanded crops; no padding."""
    _check_crops(expanded, expanded=True)
    _check_targets(target_um, len(expanded))
    if expanded.device != target_um.device:
        raise ValueError("Crops and targets must share a device")
    shift = torch.zeros((len(expanded), 3), dtype=torch.int64, device=expanded.device)
    if enabled:
        for axis, radius in enumerate(RECIPE["jitter_vox_zyx"]):
            shift[:, axis] = torch.randint(-radius, radius + 1, (len(expanded),),
                                          device=expanded.device, generator=generator)
    shifted_target = target_um - shift * target_um.new_tensor(RECIPE["voxel_um_zyx"])
    rejected = shifted_target.norm(dim=1) > RECIPE["max_shift_um"]
    shift[rejected] = 0
    shifted_target[rejected] = target_um[rejected]
    origins = shift + shift.new_tensor(RECIPE["jitter_vox_zyx"])
    dz, dy, dx = RECIPE["crop_zyx"]
    cropped = torch.stack([expanded[i, :, z:z + dz, y:y + dy, x:x + dx]
                           for i, (z, y, x) in enumerate(origins.tolist())])
    return cropped, shifted_target, shift


def group_average_logits(model, crops, *, eight_reflections=True):
    """Eight group-aligned maps; no probability/coordinate averaging alternative."""
    members = REFLECTIONS if eight_reflections else ((),)
    aligned = []
    for axes in members:
        prediction = model(reflect_crops(crops, axes))
        if axes:
            prediction = torch.flip(prediction, tuple(a + 1 for a in axes))
        aligned.append(prediction)
    aligned = torch.stack(aligned)
    # -inf is intentional and common to every support mask; average remains -inf.
    return aligned.mean(0), aligned


def decode_modes(averaged_logits, aligned_logits):
    """Mode agreement, exact no-op fallback, and descriptive posterior diagnostics."""
    if aligned_logits.ndim != 5 or averaged_logits.shape != aligned_logits.shape[1:]:
        raise ValueError("Expected aligned (views,N,Z,Y,X) and averaged (N,Z,Y,X)")
    vox, um, support, _ = _geometry(averaged_logits.device, averaged_logits.dtype)
    if tuple(averaged_logits.shape[1:]) != tuple(RECIPE["output_zyx"]):
        raise ValueError("Unexpected lattice shape")
    if not torch.isfinite(aligned_logits[..., support]).all():
        raise FloatingPointError("Nonfinite supported logits")
    per_view = aligned_logits.masked_fill(~support, -torch.inf).flatten(2).topk(2, dim=2)
    unique = per_view.values[:, :, 0] > per_view.values[:, :, 1]
    view_modes = per_view.indices[:, :, 0]
    agree = unique.all(0) & (view_modes == view_modes[0]).all(0)
    flat_logits = averaged_logits.masked_fill(~support, -torch.inf).flatten(1)
    top = flat_logits.topk(2, dim=1)
    unique_average = top.values[:, 0] > top.values[:, 1]
    agree &= unique_average & (top.indices[:, 0] == view_modes[0])
    modal_vox = vox.reshape(-1, 3)[top.indices[:, 0]].to(torch.int64)
    proposal_vox = torch.where(agree[:, None], modal_vox, 0)
    probability = flat_logits.softmax(1)
    log_probability = flat_logits.log_softmax(1).masked_fill(~support.reshape(1, -1), 0)
    probability_grid = probability.reshape_as(averaged_logits)
    # Opposite-bin cancellation returns bit-exact zero for the symmetric prior.
    mean_um = torch.stack([
        ((probability_grid - probability_grid.flip(axis + 1)) * um[..., axis]).flatten(1).sum(1) / 2
        for axis in range(3)
    ], dim=1)
    return {
        "proposal_vox": proposal_vox,
        "accepted": agree,
        "modal_vox": modal_vox,
        "view_mode_vox": vox.reshape(-1, 3)[view_modes].to(torch.int64),
        "unique_each_view": unique,
        "mode_probability": probability.gather(1, top.indices[:, :1])[:, 0],
        "mode_logit_gap": top.values[:, 0] - top.values[:, 1],
        "posterior_mean_um": mean_um,
        "posterior_entropy": -(probability * log_probability).sum(1),
        "ownership_guard_applied": False,
    }


def make_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE["steps"])
    return optimizer, scheduler


@torch.no_grad()
def augment_batch(expanded, target_um, *, generator=None):
    """Driver adapter: shared real-pixel jitter followed by a joint reflection."""
    crops, translated, _ = jitter_batch(expanded, target_um, generator=generator)
    crops, translated, _ = reflect_batch(crops, translated, generator=generator)
    return crops, translated


@torch.no_grad()
def predict(model, crops):
    """Driver adapter for the registered eight-reflection inference recipe."""
    return model.predict(crops, eight_reflections=True)
