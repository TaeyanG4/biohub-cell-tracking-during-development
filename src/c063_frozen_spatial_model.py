"""C063 shared conditional location scorer on frozen production UNet fields.

Inputs are exact FP32 production feature cubes at real, fractional-centered
C023 final integer anchors. This module neither extracts features nor selects
known pairs, folds, movie sampling, graph edits, checkpoints or acceptance gates.
Every candidate uses the same local scorer; no position-specific parameter or
synthetic displacement is used. All dimensions and coordinates are z, y, x.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn


RECIPE = {
    "version": 1,
    "seed": 6301,
    "input_channels": 32,
    "crop_zyx": [13, 13, 13],
    "input_offsets": [-6, 6],
    "output_zyx": [11, 11, 11],
    "candidate_offsets": [-5, 5],
    "feature_pitch_um_zyx": [1.625, 1.625, 1.625],
    "native_voxel_um_zyx": [1.625, 0.40625, 0.40625],
    "feature_geometry": "native(z,y,x)->(z,y/4,x/4), origin0, preserve fractional anchor",
    "feature_source": "frozen C023 primary32-channel eight-XY-view averaged full-frame field",
    "feature_context": "first-seen frame0:[0,1]/index0; t>=1:[t-1,t]/index1",
    "feature_normalization": "unchanged production movie quantiles; no fitted feature statistics",
    "support": "full real input13cube; caller rejects boundary-clamped or synthetic context",
    "representation": "raw32 concatenated with raw32 minus center32",
    "widths": [32, 16],
    "valid_kernels_zyx": [[3, 3, 3], [1, 1, 1]],
    "normalization": "LayerNorm over channels independently at each candidate",
    "last_layer_init": "zero weights, no bias",
    "position_parameters": False,
    "max_shift_um": 7.0,
    "target": "exact trilinear mass on all11cube vertices of original known-pair displacement",
    "target_support": "no sphere mask; retain every vertex for targets within7um",
    "loss": "conditional location cross entropy; known paired point only",
    "decoder": "temperature1 softmax physical mean; radial projection to7um; exact uniform fallback0",
    "augmentation": "none: no synthetic offsets, balancing, reflections, jitter or TTA",
    "training_sampler": "uniform source movie then uniform eligible original known pair",
    "unknown_cells": "never biological background/negative labels",
    "steps": 1200,
    "batch_size": 32,
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "schedule": "CosineAnnealingLR over fixed1200 steps",
    "checkpoint": "final step only; no evaluated decoder/checkpoint/strength selection",
    "precision": "FP32; AMP and TF32 off in caller",
}


def candidate_grid_um(*, device=None, dtype=torch.float32):
    """All11^3 vertices, including vertices outside the7um target sphere."""
    axis = torch.arange(-5, 6, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    return grid * grid.new_tensor(RECIPE["feature_pitch_um_zyx"])


def _check_cubes(cubes):
    expected = (RECIPE["input_channels"], *RECIPE["crop_zyx"])
    if cubes.ndim != 5 or tuple(cubes.shape[1:]) != expected or len(cubes) == 0:
        raise ValueError(f"Expected nonempty (N,{expected}), got {tuple(cubes.shape)}")
    if cubes.dtype != torch.float32:
        raise ValueError("C063 captured feature cubes must be FP32")
    if not torch.isfinite(cubes).all():
        raise FloatingPointError("Nonfinite frozen feature cube")


def _check_targets(target_um, n=None):
    if (target_um.ndim != 2 or target_um.shape[1] != 3 or len(target_um) == 0
            or (n is not None and len(target_um) != n)):
        raise ValueError("Known original-pair target coordinates must have shape (N,3)")
    if target_um.dtype != torch.float32:
        raise ValueError("C063 original-pair targets must be FP32 physical micrometres")
    if not torch.isfinite(target_um).all():
        raise FloatingPointError("Nonfinite original-pair target")
    if (target_um.norm(dim=1) > RECIPE["max_shift_um"] + 1e-5).any():
        raise ValueError("Original known-pair displacement exceeds the fixed7um identity gate")


def _check_logits(logits):
    if logits.ndim != 4 or tuple(logits.shape[1:]) != tuple(RECIPE["output_zyx"]) or len(logits) == 0:
        raise ValueError("Expected nonempty (N,11,11,11) candidate logits")
    if logits.dtype != torch.float32:
        raise ValueError("C063 logits must be FP32")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite conditional candidate logits")


class ChannelNorm(nn.Module):
    """Shared affine channel normalization, independent for each spatial query."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, value):
        return self.norm(value.movedim(1, -1)).movedim(-1, 1)


class FrozenSpatialLocalizer(nn.Module):
    """One local3cube shared scorer with explicit original-anchor contrast.

    The raw input is unaltered. The contrast channel subtracts only the center
    vector of this same real cube, never a fitted source/target-domain mean.
    Biases/normalization are shared spatially; the final score has no bias.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=0), ChannelNorm(32), nn.SiLU(),
            nn.Conv3d(32, 16, 1, padding=0), ChannelNorm(16), nn.SiLU(),
        )
        self.head = nn.Conv3d(16, 1, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, cubes):
        _check_cubes(cubes)
        anchor = cubes[:, :, 6:7, 6:7, 6:7]
        query = torch.cat((cubes, cubes - anchor), dim=1)
        logits = self.head(self.features(query))[:, 0]
        if tuple(logits.shape[1:]) != tuple(RECIPE["output_zyx"]):
            raise AssertionError("Shared valid-convolution candidate geometry drift")
        return logits

    @torch.no_grad()
    def predict(self, cubes):
        """Return fixed continuous proposals; caller owns original-pair audit."""
        return decode_logits(self(cubes))


SpatialLocalizer = FrozenSpatialLocalizer


def trilinear_target(target_um):
    """Exact fractional coordinate distribution, without rounding or clipping.

    Targets lie within7um (4.308 feature cells), hence all eight enclosing
    lattice vertices lie within[-5,5]^3. A sphere mask would incorrectly
    discard some of these vertices and move labels inward; none is applied.
    """
    _check_targets(target_um)
    grid_position = target_um / target_um.new_tensor(RECIPE["feature_pitch_um_zyx"]) + 5.0
    lower = grid_position.floor().to(torch.int64)
    fraction = grid_position - lower.to(target_um.dtype)
    result = target_um.new_zeros((len(target_um), 11 * 11 * 11))
    for corner in itertools.product((0, 1), repeat=3):
        high = lower.new_tensor(corner)
        indices = lower + high
        if ((indices < 0) | (indices > 10)).any():
            raise AssertionError("Full trilinear target support was not retained")
        weights = torch.where(high.bool()[None], fraction, 1.0 - fraction).prod(dim=1)
        flat = (indices[:, 0] * 11 + indices[:, 1]) * 11 + indices[:, 2]
        result.scatter_add_(1, flat[:, None], weights[:, None])
    return result.reshape(len(target_um), 11, 11, 11)


def conditional_loss(logits, target_um):
    """Point-conditional cross entropy; zero-head inference fallback is absent."""
    _check_logits(logits)
    _check_targets(target_um, len(logits))
    if logits.device != target_um.device:
        raise ValueError("Candidate logits and original-pair targets must share a device")
    labels = trilinear_target(target_um)
    return -(labels.flatten(1) * logits.flatten(1).log_softmax(dim=1)).sum(dim=1).mean()


@torch.no_grad()
def decode_logits(logits):
    """Fixed expectation decoder with an exact no-op for uniform logits.

    The7um radial projection is the original physical domain limit. There is
    no confidence threshold, fitted temperature, shrink scalar or prior.
    Partial ties/multiple peaks retain the declared distribution mean. Exact
    all-location ties return exact zero independently of float summation.
    """
    _check_logits(logits)
    flat = logits.flatten(1)
    uniform = (flat == flat[:, :1]).all(dim=1)
    probability = flat.softmax(dim=1)
    grid_um = candidate_grid_um(device=logits.device).reshape(-1, 3)
    mean_um = probability @ grid_um
    mean_um = torch.where(uniform[:, None], torch.zeros_like(mean_um), mean_um)
    norm = mean_um.norm(dim=1, keepdim=True)
    scale = (RECIPE["max_shift_um"] / norm.clamp_min(torch.finfo(logits.dtype).tiny)).clamp(max=1.0)
    shift_um = mean_um * scale
    return {
        "shift_um": shift_um,
        "raw_mean_um": mean_um,
        "uniform": uniform,
        "projected": norm[:, 0] > RECIPE["max_shift_um"],
    }


def make_optimizer(model):
    """Fixed source-only training recipe; no fitting, sampling or device change."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE["steps"])
    return optimizer, scheduler


def fit_step(model, optimizer, scheduler, cubes, target_um):
    """One FP32 step; caller samples a source movie then a real known pair."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = conditional_loss(model(cubes), target_um)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite C063 conditional loss")
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())
