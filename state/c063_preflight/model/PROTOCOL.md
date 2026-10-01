# C063 fixed shared scorer for frozen production spatial features

This is a new component protocol, not a claim of measured improvement.
Only the new shared scoring head is fitted on the opposite whole embryo.
The frozen public backbone already saw both embryos. The caller retains
original known C023 node/GT pairs and reports excluded pairs unchanged.

## Capture contract

`cubes` is FP32 `[N,32,13,13,13]`, ordered channels,z,y,x. The original
C023 final integer anchor maps to feature `(z,y/4,x/4)` without an origin
shift; fractional XY phases remain fractional. Capture offsets -6 through6
using the exact production trilinear sampler on its real primary32-channel
eight-view averaged field. Physical lattice pitch is1.625um in every axis.
Required full real feature support excludes boundary clamping. Each cube is
281,216bytes. Capture agent owns real context/window and no-op proof.

The first-seen context is frame0 `[0,1]`, index0; every`t>=1` `[t-1,t]`,
index1. Preserve production full-frame quantiles, FP32 TTA addition order,
and no upper clip. No detached small-image encoder or old224/64-vector
cache can substitute for the dense field. See
`state/c062_preflight/geometry/REVIEW.md` for the source geometry proof.

## Fixed head and labels

Concatenate each feature cube with its feature-minus-original-anchor
contrast. Apply a shared valid3cube Conv64->32, per-position channel
LayerNorm/SiLU, shared1cube Conv32->16/LayerNorm/SiLU, and a zero-initialized
biasless shared1cube score. Output `[N,11,11,11]` represents query offsets
-5 through5. There are no coordinate channels, position embeddings,
per-location parameters, global fitted normalization, embryo-prefix inputs
or separate classification intercepts. All internal affine parameters are
shared across query locations. Architecture is frozen before fit results.

For each actual fixed original prediction/known-GT pair within7um, retain
the exact GT-minus-anchor physical residual. Its trilinear target mass goes
to eight surrounding feature-grid vertices. No rounding, synthetic offset,
label clipping or target standardization occurs. Candidate vertices extend
to8.125um per axis, ensuring complete support for every fractional target
within7um. **Do not mask the7um sphere:** such a mask discards vertices near
its boundary and changes their expected physical coordinate. Conditional
cross entropy is over the complete11cube. This location distribution refers
only to the supplied known point; other cells are not background labels.

## Fixed decoder, zero baseline and fit

Use temperature1 softmax expectation in physical micrometres, then project
only values beyond the geometric7um radial limit onto that sphere. Exact
uniform logits return exact zero, including the zero-initialized head.
Nonuniform multiple peaks use the same mean. There is no tuned confidence,
temperature, shrink scale, mode variant, prior, augmentation or TTA.
The zero fallback is inference-only; it cannot suppress the initial
cross-entropy gradient. The decoder is fixed before any result inspection.

Fit each opposite-embryo source with uniform source movie, then uniform
eligible real known pair sampling. Preserve good and tail frequencies;
no class balancing, synthetic displacement, jitter or reflection. Fixed
1200steps, batch32, AdamW lr3e-4, weight decay1e-4, cosine schedule, seed6301,
FP32/TF32off, final checkpoint only. No source/target result selects a step,
decoder or strength. The driver owns deterministic seed and frozen backbone.

Good-point protection comes from training on real deployment residuals,
not a label-based inference mask. It is a required measured gate, never
guaranteed by this architecture. Exact uniform fallback preserves original
anchors before fitting and any later uniform cases. The conditional mean
can fall between plausible cells, and features can retain embryo bias.

## Required diagnostics before any integration

On fixed actual original pairs in both opposite directions, require lower
mean3D and abs-z errors overall, >3.5um3D tails and |dz|>3.5um tails, with
no worsening of original<=2.5um good means. Include all excluded/no-op pairs,
signed bias, per-movie effects, and ownership conflicts. At least two movies
per embryo must improve tails. Do not rematch labels to make residuals look
better. Report source-fit behavior separately from opposite-embryo transfer.

These component diagnostics alone are not graph or leaderboard gains.
Actual all-node inference must preserve good known identities under the
original matcher and pass exact original-writer/official zero controls.
Pre-association changes need fresh indexing, candidates, Transformer and
ILP; stale association caches cannot validate changed geometry. No local
GPU capture, fitting or queue is launched by this model-only work.

## Interfaces

- `RECIPE` is the full immutable module contract.
- `FrozenSpatialLocalizer` (`SpatialLocalizer` alias): `forward(cubes)`
  returns logits; `predict(cubes)` returns the fixed decoder dictionary.
- `candidate_grid_um`, `trilinear_target(target_um)` and
  `conditional_loss(logits,target_um)` expose physical target proof.
- `decode_logits(logits)` returns `shift_um`, `raw_mean_um`, `uniform`,
  and `projected` tensors. Shifts are continuous physical z,y,x.
- `make_optimizer(model)` and `fit_step(...)` expose the fixed recipe.

Bounded CPU controls are recorded separately; their synthetic arrays test
arithmetic and gradients only and are never production training examples.

Completed CPU proof in `controls.json`:55,968 parameters;2,008 geometric
cases (including7um axes/sphere and fractional phases), max physical target
expectation discrepancy1.91e-6um. All16,931 original C058 pair-label rows
across22 movies reproduce their physical residuals exactly. Uniform/zero
fallback, physical direction, geometric projection, local-neighborhood swap,
constant-field spatial uniformity, train/eval equality and finite nonzero
head/subsequent-feature gradients pass. No GPU calls or real-data fit ran.
Stable model SHA256:
`fafd3c0deb6a97a9785e12b874fc854cfd93cf18aa6c3b6f6d148d81993a2547`.
