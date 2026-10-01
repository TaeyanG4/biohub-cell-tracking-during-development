# C063 independent primary-prefix review

Verdict: **no blocking issue found for the two live prefix-proof jobs**.
The prefix must pass both actual full-field/cube proofs before the extraction
verb is reused for the remaining movies. No GPU command, model inference,
fit or queue was launched by this review.

Reviewed extractor SHA256:
`635ed5b9ed4cf404e949ebddd7e1ec5160b7045c270dbe46bb9e265137aabec2`.
Production predictor SHA256:
`8140a92d916e0d65877d51ee53cc6bd0438b36538cda3365fa99fffa91d6eb25`.
Generated AST SHA256:
`c28c5d9745ecf29b076b8abebef61c649f033aeb100116dcb7ae7ca9ec9594a3`.

## Findings

- The AST retains all 24 original setup statements, six complete primary
  statements inside the window loop, original arguments/defaults and
  `torch.no_grad` decorator. The sixth statement is the complete primary
  eight-view TTA block, including transforms, inverses, addition order and
  averaging. It stops before any secondary inference. There is no replacement
  normalization, frame reader, encoder or interpolation implementation.
- First-seen context is correct: t0 and t1 use window [0,1] at indices 0 and 1;
  later t uses [t-1,t] at index1. Requested windows are a verified subset of
  the original production window list. SparseSink ignores all other contexts
  and requires each selected frame and cube row exactly once.
- Original `open_dataset(...load_image=False, require_tracks=False)` reads
  image metadata without reopening GT or materializing image data. The exact
  production `_load_frame` reads the requested time frame with native strides.
  All 22 movies have checked full-frame chunks `(1,Z,Y,X)` and slash-separated
  keys. Dependencies include both Zarr metadata files and all requested and
  preceding-context frame chunks. Unsupported layout or missing real chunks
  fails rather than synthesizing data.
- Selection and targets reuse the passive capture contract and preserve all
  original IDs, labels, fractional anchors and excluded pairs. These GT-known
  windows are a component diagnostic, not a deployment eligibility rule.
- `prove` requires the original full-pipeline capture's exact zero/official/ILP
  controls. It then compares every requested complete field hash, shape,
  temporal context and anchor count, complete NPY bytes and all numeric cube
  values, all target arrays and both selected/all-original CSV files. This
  catches contextual, precision, TTA, model and extraction differences; AST
  equality alone is not used as empirical field equivalence.
- `extract` requires both benchmark proofs and rechecks extractor, capture,
  production source, generated AST and all shared source/config/checkpoint
  dependencies. Each movie has before/after input hashes, frozen primary-state
  checks and overwrite rejection. Proof receipts are also pinned inputs to
  the new extraction. Different movies still need their own row/support checks.

## Bounded independent CPU check

`prefix_review_checks.py` and `prefix_cpu/review_checks.json` record an actual
CPU check. It independently recompiles the selected AST, verifies the retained
no-grad decorator, reproduces all 22 saved metadata plans and checks every
concrete dependency exists: 16,931 original pairs, 9,113 eligible pairs,
1,945 windows, 1,978 image chunks and 2,089 dependencies.

A small constant-field fixture exercises the actual SparseSink with the
original production interpolation function, requested times 0/1/4/9 and extra
overlapping windows. Saved cubes and temporal receipts exactly select the
intended first-seen field at each time; duplicate/context frames are ignored.
This proves routing logic only, with no model or GPU use.

No source changes requested. Actual live prefix equality and timing remain
the next finite jobs' responsibility; this review makes no efficacy claim.
