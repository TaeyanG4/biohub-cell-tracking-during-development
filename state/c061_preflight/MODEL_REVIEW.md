# C061 model implementation and bounded controls

`src/c061_axial_model.py` implements the registered 315,889-parameter model:
float32 `(N,1,13,49,49)` images; three Conv3d/GroupNorm4/SiLU blocks with
widths8/16/24 and strides `(1,2,2)/(2,2,2)/(2,2,2)`; spatial field4x7x7;
flattened64-unit hidden head; nine logits for native z offsets -4 through4.
The last layer starts at exact zero. There is no center prior, normalization
state learned across batches, or confidence threshold in this module.

Interfaces:

- `AxialLocalizer(crops)` returns `(N,9)` conditional logits.
- `shifted_batch(expanded, shifts)` takes float32 `(N,1,21,49,49)` and integer
  `(N,)` shifts in[-4,4]; returns `(cropped, target_z_vox)`, using slice start
  `4+shift` and integer-known-point target `-shift`.
- `balanced_shifts(generator=..., device=...)` returns36 shuffled shifts with
  exactly four of each offset. Caller samples source movie then known point.
- `reflect_xy_batch(crops, generator=...)` returns `(reflected, masks)`;
  masks has shape `(N,2)` ordered y/x. Axial targets are unchanged. No z flips.
- `loss(logits, target_z_vox)` supports integer labels and explicit adjacent-bin
  linear soft labels for fractional z. Nonfinite/out-of-range labels raise;
  no target clipping. For fractional GEFF points, caller must add the actual
  integer-anchor residual to shifted targets before passing them to loss.
- `decode(logits)` returns a dict with `z_vox`, `tied`, `class_index`,
  `probability`; unique argmax maps directly to[-4,4], exact ties return z0.
  `class_index` on a tie is descriptive only; callers must use `z_vox/tied`.
- `make_optimizer(model)` constructs AdamW lr3e-4/wd1e-4 and the1200-step
  cosine schedule. It does not train. Full recipe and precision policy are
  recorded in `RECIPE`; caller must enforce FP32 and disable AMP/TF32.

`model_controls.py` ran on CUDA and passed. It uses only disposable synthetic
images, one optimizer step and a second gradient calculation; there is no
source fit or saved checkpoint. A coordinate ramp verifies every shifted crop
pixel, target sign and XY reflection. All class decodes and tied fallback,
integer CE equivalence, fractional target mass/coordinate and invalid input
rejections pass. The initial loss is log9=2.197224617; after the disposable
step2.195908070. Initial final-head gradient norm0.2332885; subsequent first
convolution gradient norm0.0043902, both finite/nonzero. This checks trainability
of the implementation, not learnability or biological performance.

Exact source/control hashes, class counts, device, timing, recipe and control
results are in `model_controls.json`. Only the new C061 model and these
preflight files were created by this subtask. Old candidates are untouched.
