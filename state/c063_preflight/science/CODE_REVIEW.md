# C063 independent preflight code review

Reviewed 2026-09-29 13:09 KST. Verdict: **no blocking issue found for the
three-job passive capture preflight**. This is a source and saved CPU-control
review, not evidence that the unrun live controls have passed or that the
learner improves localization. No GPU, fitting, queue or original-source edit
was performed by this reviewer.

## Capture and original-control integrity

- The active hook calls the original candidate-mode V1284 public head and
  returns that exact result. It verifies the incoming array and complete
  production feature field are unchanged. The public coordinate head remains
  active in both original and passive arms.
- Original final C023 integer node IDs/times/positions and existing official
  GT pairs remain the anchors. Eligibility uses nonsynthetic provenance and
  complete real feature support, not label distance or a GT-derived correction
  mask. All excluded original pairs remain explicitly represented.
- Native `(z,y,x)` maps to `(z,y/4,x/4)` with fractional XY preserved.
  Production trilinear interpolation samples the 13-cube in z,y,x order.
  The live hook asserts real field bounds, exact independent center sampling,
  finite cubes, one visit per production frame and complete row coverage.
- Source inspection confirms window size two, first-seen t0 in [0,1]/index0
  and later t in [t-1,t]/index1, primary eight-view feature averaging, original
  frame quantiles and lower-only clamping. Recording these context values from
  t is valid under the pinned predictor and asserted window size.
- Full original/passive inference is followed by exact complete cache equality
  and semantic raw ILP equality. The passive run also must match the existing
  C023 FP32 reference. Fresh passive graphs/caches then pass through the
  existing C058 baseline helper, original C023 postprocessing, integer
  conversion and actual organizer matching/scoring. Final IDs, integer and
  floating coordinates, edges and synthetic provenance must be exact.
- A final postprocessing anchor may differ from its original detector center
  because C023 linefit smoothing is intentional. Querying the unchanged dense
  image field at that final anchor is coherent; no false equality between
  those distinct positions is required.
- The queue verifies concrete dependencies before/after and its own plan hash,
  runs each movie in a fresh process, and refuses blind overwrite/resume.
  Notification logs are external to candidate outputs. The launcher records
  owned PID and creation time before starting the existing hidden notifier.
  Root still owns the separate scientific heartbeat registration.

## Fixed model and physical targets

The selected model is the actual 55,968-parameter shared 13-cube-to-11-cube
scorer: raw32 plus anchor-contrast32, valid shared 3-cube convolution, channel
LayerNorm, shared 1-cube layers and zero-initialized biasless score. There is
no per-position parameter or fitted global normalization. The earlier
alternative proposal is not part of this review or implementation.

Exact trilinear mass preserves every known physical target within 7um. Full
11-cube vertices are necessary near the sphere boundary; masking vertices to
7um would move some labels inward. Flattening and candidate-grid ordering
agree. The fixed temperature-one physical expectation and radial projection
are internally consistent. Uniform logits return exact zero only at inference;
the conditional cross entropy retains an initial nonzero head gradient.

Saved CPU controls cover 2,008 geometry cases, all 16,931 existing original
pair residuals across 22 movies, physical sign/units, exact zero fallback,
fractional means, radial projection, finite gradients, shared local behavior,
constant-field uniformity and train/eval equality. This reviewer independently
rehashed the model/control source and all 22 referenced label files; saved
receipt hashes match. The interpolation receipt also matches the exact
production sampler source. No repeated training or new arithmetic harness was
needed.

## Scope and remaining gates

Preflight has no model fit and therefore does not need model code to execute.
Before a later fit phase, pin the fixed model and PROTOCOL alongside that
phase's driver, actual feature inputs, split/sampling proof and controls.
Whole-embryo source-only fitting, final-step-only selection and unchanged
excluded-pair reporting are requirements for that later driver, not features
already implemented by this capture-only queue.

The conditional expectation can fall between plausible cells, and the frozen
public backbone already saw both embryos. Those disclosed scientific risks
are measured by the prospective opposite-embryo good/tail/z gates; they are
not reasons to substitute an architecture or tune a decoder after results.
Two movies establish capture parity and cost only. Component diagnostics do
not establish graph/LB gains. Any coordinate intervention still requires
actual all-node identity/official checks and, if upstream, freshly recomputed
indexing, candidates, Transformer outputs and ILP.

## Reviewed source SHA256

| File | SHA256 |
|---|---|
| src/c063_frozen_spatial_capture.py | `5d2b04526cb65ef270b0a2396a833375c3ebb8c3d5c947c91d903c1a5c6cff4b` |
| src/c063_frozen_spatial_study.py | `42bb9c6b65953795f5e2bafa59e0020e29c49c6569c31d3b24e41f38707e8ff0` |
| src/c063_frozen_spatial_model.py | `fafd3c0deb6a97a9785e12b874fc854cfd93cf18aa6c3b6f6d148d81993a2547` |
| state/c063_preflight/launch.ps1 | `95d4a6b6f65233af4bb11d648f9c0852657d7a2fd285ad9bc3fe372f10b35841` |
| state/c063_preflight/geometry/CONTRACT.md | `c3e692d684dd22671d957552133d6aea6a5cb412b5033066f5ebda6434d519ad` |
| state/c063_preflight/model/PROTOCOL.md | `9b72c196761fef7e999e40168996b97e48133664169c51a42fc27c9ef5b911f3` |
| state/c063_preflight/model/controls.py | `94d8e80a9f37d31406cced937856599718af86c7ed4dbbbb99c0d25167c616ca` |
| state/c063_preflight/model/controls.json | `41294346fad7515ed07dfee92a2035877c18cc0affd256671b2e24db27be97e6` |
| state/c063_preflight/geometry/local_controls/controls.json | `f5cff7a574a9d4f6a8c7e7fa18c9603e63da0727f29b911211bb09b902dc4051` |

No code changes requested for this preflight.
