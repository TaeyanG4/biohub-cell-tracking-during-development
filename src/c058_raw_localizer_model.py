"""C058 fixed spatial raw-image coordinate learner; no graph or image IO.

The input is one normalized full-resolution crop centered on an integer output
node. The output is a correction in micrometres, ordered (z, y, x), relative to
that same integer center. All three spatial axes survive into the flattened
head; global average pooling is deliberately absent. This module does not
select labelled examples, folds, checkpoints, nodes, or deployment thresholds.

For the even-sized crop, an array reflection is NOT simple offset negation:
with crop origin center - shape//2, flipping axis a maps d[a] to
-d[a] - voxel_um[a]. The caller owns exact joint image/target augmentation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RECIPE = {
    "version": 1,
    "seed": 5801,
    "crop_zyx": [16, 64, 64],
    "stored_crop_zyx": [18, 72, 72],
    "jitter_vox_zyx": [1, 4, 4],
    "jitter_policy": "integer crop translation; use zero jitter if target norm would exceed 7 um; no flips",
    "voxel_um_zyx": [1.625, 0.40625, 0.40625],
    "normalization": "movie quantiles 0.001/0.999; clip [0,3]",
    "input_channels": 1,
    "widths": [8, 16, 24],
    "strides_zyx": [[1, 2, 2], [2, 2, 2], [2, 2, 2]],
    "group_norm_groups": 4,
    "spatial_pool_zyx": [4, 8, 8],
    "hidden": 64,
    "max_shift_um": 7.0,
    "bound": "raw / sqrt(1 + sum(raw**2)/max_shift_um**2)",
    "last_layer_init": "all zeros; exact zero initial correction",
    "target": "known GT zyx minus integer crop-center zyx, in micrometres",
    "steps": 1200,
    "batch_size": 32,
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "schedule": "CosineAnnealingLR over fixed 1200 steps",
    "loss": "SmoothL1 in micrometres; beta 1; mean over examples and axes",
    "loss_beta_um": 1.0,
    "precision": "FP32; AMP and TF32 off",
    "checkpoint": "final step only; no target-embryo selection",
    "training_sampler": "uniform training movie, then uniform eligible known match; no label-count cap",
}


class RawLocalizer(nn.Module):
    """Small position-preserving CNN with a smooth 7-um radial output bound.

    No BatchNorm or dropout: predictions do not depend on other crop samples or
    train/eval state. The radial map has unit Jacobian at zero, so the zero head
    receives ordinary regression gradients and cannot hit a zero-norm divide.
    """

    def __init__(self):
        super().__init__()
        layers = []
        previous = RECIPE["input_channels"]
        for width, stride in zip(RECIPE["widths"], RECIPE["strides_zyx"]):
            layers.extend([
                nn.Conv3d(previous, width, 3, stride=tuple(stride), padding=1),
                nn.GroupNorm(RECIPE["group_norm_groups"], width),
                nn.SiLU(),
            ])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(tuple(RECIPE["spatial_pool_zyx"]))
        flattened = previous
        for size in RECIPE["spatial_pool_zyx"]:
            flattened *= size
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened, RECIPE["hidden"]), nn.SiLU(),
            nn.Linear(RECIPE["hidden"], 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        expected = (RECIPE["input_channels"], *RECIPE["crop_zyx"])
        if crops.ndim != 5 or tuple(crops.shape[1:]) != expected:
            raise ValueError(f"Expected (N, {expected}), got {tuple(crops.shape)}")
        raw_um = self.head(self.pool(self.features(crops)))
        limit = float(RECIPE["max_shift_um"])
        return raw_um / torch.sqrt(1.0 + raw_um.square().sum(dim=1, keepdim=True) / (limit * limit))


def localization_loss(predicted_um: torch.Tensor, target_um: torch.Tensor) -> torch.Tensor:
    """Known paired coordinate targets only; caller must exclude unknown nodes."""
    if predicted_um.shape != target_um.shape or target_um.ndim != 2 or target_um.shape[1] != 3:
        raise ValueError("Expected matching (N, 3) predicted and target coordinates")
    return F.smooth_l1_loss(predicted_um, target_um, beta=RECIPE["loss_beta_um"], reduction="mean")


def make_optimizer(model: RawLocalizer):
    """Fixed optimizer and step-count schedule; no fitting or device changes."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE["steps"])
    return optimizer, scheduler


@torch.no_grad()
def jitter_batch(expanded: torch.Tensor, target_um: torch.Tensor, *,
                 generator: torch.Generator | None = None, enabled: bool = True):
    """Crop integer jitter from a real expanded image; translate targets exactly.

    Accepts (N,1,18,72,72) and (N,3) on the same device. A seeded generator must
    be on that device too. Returns cropped images, translated targets, and the
    *accepted* integer center displacement in zyx voxels. Rejected jitter uses
    the original center and original label, preserving every matched tail case.
    Expanded images must contain real support; the caller determines eligibility
    from image bounds before this function and never synthesizes padded support.
    """
    expected = (1, *RECIPE["stored_crop_zyx"])
    if expanded.ndim != 5 or tuple(expanded.shape[1:]) != expected:
        raise ValueError(f"Expected (N, {expected}), got {tuple(expanded.shape)}")
    if target_um.shape != (len(expanded), 3) or target_um.device != expanded.device:
        raise ValueError("Targets must have shape (N,3) and share the crop device")
    if not torch.isfinite(target_um).all() or (target_um.norm(dim=1) > RECIPE["max_shift_um"] + 1e-5).any():
        raise ValueError("Known targets must be finite and within the fixed 7-um matching gate")
    shifts = torch.zeros((len(expanded), 3), dtype=torch.int64, device=expanded.device)
    if enabled:
        for axis, radius in enumerate(RECIPE["jitter_vox_zyx"]):
            shifts[:, axis] = torch.randint(-radius, radius + 1, (len(expanded),),
                                            generator=generator, device=expanded.device)
    voxel = target_um.new_tensor(RECIPE["voxel_um_zyx"])
    shifted_target = target_um - shifts.to(target_um.dtype) * voxel
    rejected = shifted_target.norm(dim=1) > RECIPE["max_shift_um"]
    shifts[rejected] = 0
    shifted_target[rejected] = target_um[rejected]
    origin = shifts + shifts.new_tensor(RECIPE["jitter_vox_zyx"])
    dz, dy, dx = RECIPE["crop_zyx"]
    # Each slice references real source pixels; there is no interpolation,
    # resampling, image shift padding, or target clipping.
    crops = torch.stack([expanded[i, :, z:z+dz, y:y+dy, x:x+dx]
                         for i, (z, y, x) in enumerate(origin.tolist())])
    return crops, shifted_target, shifts


def fit_step(model: RawLocalizer, optimizer, scheduler, crops: torch.Tensor,
             target_um: torch.Tensor) -> float:
    """One fixed FP32 regression step; caller owns sampling/device/augmentation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = localization_loss(model(crops), target_um)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite raw-localizer loss")
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())
