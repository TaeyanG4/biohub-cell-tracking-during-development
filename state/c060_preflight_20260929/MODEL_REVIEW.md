# C060 spatial model preflight

`src/c060_spatial_localizer_model.py` is implemented; no fitted experiment
checkpoint or queue was created by this subtask. Parent owns extraction,
whole-embryo training, actual writer/metric checks, and the full-movie benchmark.

The model has22,432 parameters. Three own-centered13x49x49 image views pass
through two valid3x7x7 convolutions (8/16 channels), independent per-voxel channel
normalization, and a zero-initialized scalar head. Output9x37x37 offsets cover
5,381 lattice locations in the physical7um sphere. The symmetric prior has
sigma7/3um, a fixed three-sigma support scale; it can suppress useful tail
corrections and must not be tuned to evaluation outcomes.

API:

- `SpatialLocalizer()(crops)` returns `(N,9,37,37)` masked conditional logits.
- `conditional_loss(logits,target_um)` is Gaussian point-target cross entropy.
  Target sigma is one native voxel on each axis. Unknown cells are not cell or
  background labels; input point identity and source supervision are caller duties.
- `jitter_batch(expanded,target_um)` returns crop, translated target, accepted
  shared integer shifts. `reflect_batch(crop,target)` returns reflected crop,
  target, and reflection masks. `augment_batch` combines these and returns two.
- `model.predict(crop)` or `predict(model,crop)` returns `proposal_vox:int[N,3]`
  and `accepted:bool[N]`, plus modal/mean/entropy/agreement diagnostics. The
  proposal is the integer mode only when all eight inverse-reflected maps have
  that same unique mode. Otherwise it is exactly zero. Logits, not posterior
  probabilities or coordinates, are group-averaged.
- `accepted` is **only** mode agreement. The driver must additionally require
  strict nearest-center ownership among **all current-frame predicted nodes**.
  No GT-derived runtime gate is implemented in this module.

`model_controls.py` uses two actual C023 predicted three-node paths in
44b6_12dfb391, their original raw images/normalization/crop function, and fixed
disposable target offsets. It does not select inputs or targets using GT. It
passed exact zero initial proposal and mean, all eight reflection transforms,
shared jitter sign/pixel support, rejected-jitter fallback, random reflection,
finite nonzero head gradients and feature gradients after one disposable
optimizer update, unique-mode agreement, tied-map fallback, and cross-view
disagreement fallback. No checkpoint was saved.

Gaussian reflection error was at most1.49e-8. Group-averaged supported-logit
reflection error after the disposable update was at most9.54e-7. Native RTX4070
Ti SUPER FP32 smoke separately passed zero output and finite loss/gradients.
`model_controls.json` records code/input hashes and actual measured values.

The learned image-evidence logits also passed one-voxel common-valid-support
translation in all three axes on those real crops. The anchor-centred prior
and finite search sphere do not claim invariance when changing the anchor.

These are implementation controls, not evidence of useful localization or
biological confidence. The all-node ownership guard remains a separate
caller-level check. The final graph's IDs/edges remain
unchanged in the planned component study; upstream association improvement
would require a separate dependency-correct replay after a successful component.
