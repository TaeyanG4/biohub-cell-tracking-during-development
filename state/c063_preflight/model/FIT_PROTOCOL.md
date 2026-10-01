# C063 fixed fit and analysis adapter

`src/c063_frozen_spatial_fit.py` consumes the unchanged capture contract.
This implementation has not run fitting, prediction, benchmark or GPU work.
Source AST and the CPU actual two-movie full-capture loader passed. Parent
owns the finite queue and independent training-path controls.

## Commands and paths

All commands take `--capture-dir <C063/capture>` and `--out <phase-output>`.
`fit`, `predict`, `benchmark` additionally require `--source 44b6|6bba`;
`--device cuda|cpu` defaults to CUDA. `analyse` needs neither argument.
Commands refuse to overwrite their completed or partial output targets.

1. `benchmark`: any nonempty passed subset of original22 captures, fixed30
   disposable source-only steps (960 samples), actual feature zero-head
   check on all passed cubes, timings to `benchmark.json`; no checkpoint,
   validation result or strength choice. The caller explicitly schedules it.
2. `fit`: all22 captures required; model trained solely on `--source`, one
   source movie per batch then32 uniform eligible pairs with replacement.
   Fixed1200steps, final checkpoint only. `models/<source>.pt`, `.json` and
   `<source>_sample_counts.csv` retain all actual pair/move sampling counts,
   sequence hash and13 loss-history points. Source actual feature cubes
   all pass the zero-head residual no-op before fitting. Saved checkpoint
   state, actual logits and decoded output must reload exactly.
3. `predict`: final source checkpoint predicts every eligible cube on both
   embryos; all16,931 original pairs remain in `evaluation/pairs_<source>.csv`.
   Excluded points retain exact0 shift. No label rematching, coordinate
   rounding, ownership clipping or diagnostic-based acceptance occurs.
4. `analyse`: validates both saved prediction files, original IDs/labels,
   checkpoint/source-fit proofs, exclusion no-ops and residual arithmetic;
   saves complete paired diagnostics, stratum summary, movie-stratum summary
   and `analysis/decision.json`.

## Input proof

Each loader hashes and verifies its exact source/model/C058 labels/baselines,
capture labels/targets/features/metadata and **every nested pinned input**.
The public `dependencies` function returns the complete verified evidence
chain, and `load_data` returns all its hashes. It accepts either original
full-passive `summary.json`+`receipt.json` or proven prefix `summary.json`.
Full-passive receipts prove unchanged C023 graph/cache/writer/official
behavior; they are checked against the actual original-writer zero graph
and official result. Prefix extraction receives no invented per-movie
official-zero or ILP flag.

Every prefix dataset is bound to **both** original two-movie `proof.json`
files in the proof root, their actual output summaries/features/targets,
all shared and per-movie pinned input hashes, and their original full
pipeline reference summaries/receipts. Exact field digests, complete FP32
cube hashes, target arrays and original identity manifests are independently
compared across the proof and full-reference outputs. Current extractor,
capture, production source, primary checkpoint/config, retained AST,
generated prefix text, frozen model-state digest, first-seen frame windows,
feature geometry and real-image support must bind to the same representation.
The proof movies and additional20 extracted movies may share a prefix root;
`--capture-dir <candidate/prefix>` then consumes all22 without recapturing
the proof movies. Prefix runtime verification requires both proofs complete.
Original row/node/GT/time/residual identities are checked against C058.
Feature support and nonsynthetic status are independently reconstructed from
the original graph and full feature-grid bounds; no GT value controls them.
Each targets.npz row must match its labels.csv cube row and original identity.
Full fit/predict/analysis require exactly the original22 and16,931 pairs.
`targets_grid` is verified from the authoritative original double residuals
divided by1.625 and then cast to FP32, matching capture serialization.

`loader_controls.py/json` records the completed CPU read of the actual two
passive movies:1,997 original pairs,1,526 captured cubes,406 hashed evidence
files. No GPU call, model construction or training occurred. Actual prefix
loader control remains pending the two live prefix proofs; no mock proof
substitutes for that verification.
The first actual prefix movie44b6 also passed `prefix_summary` CPU checks
(588 cubes,145 verified files); this alone does not satisfy the required
two-movie proof gate.

## Reports and fixed gate

All original pairs are the principal common-support denominator. Additional
eligible and excluded-only reports expose support changes. Original good
means3D error<=2.5um,3D tail>3.5um, axial tail absolute-z>3.5um. Every stratum
reports3D and abs-z mean/median changes, improved/worsened counts, signed
z/y/x biases, uniform/projection counts and spatial ownership diagnostics.
Source-fit reports are separate from opposite-embryo reports.

Ownership uses only the original all-node coordinates at the same frame:
the proposed point conflicts if another original node is as near or nearer
than its own anchor. Original duplicates and newly introduced conflicts are
reported separately. This is a diagnostic, not a tuned rejection mask.

For **each opposite embryo**, overall,3D-tail and axial-tail3D **and** abs-z
means must improve; good3D and abs-z means must not worsen (1e-12 arithmetic
tolerance); each tail stratum must improve both errors in more than one movie.
Empty required strata fail. The result is either `hold` or `review_required`.
A pass still requires independent ownership/all-node review before graph
integration. No graph score, missing-edge recovery or leaderboard gain is
claimed, and no GT-based deployment eligibility is created here.

The backbone saw both embryos. Overlapping movies are not independent
embryos. Negative results do not justify a decoder/strength/epoch sweep.
