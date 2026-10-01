# C058 raw-image coordinate localization on frozen C023 output

One fixed new component, authorized by the user's 2026-09-29 instruction to
pursue grounded performance improvements without a small-score target.
No local or hidden gain is known at registration.

The question is whether full-resolution spatial image information can correct
the localization tail that the existing frozen-feature head leaves behind.
The C023 final node IDs, node counts, times and edges stay fixed. Its already
rounded emitted coordinates are the input centers. The original sources and
C023/C024 final picks remain unchanged.

## Recipe

`src/c058_raw_localizer_model.py` contains one position-preserving 3D CNN:
16x64x64 input,8/16/24 channels, spatial4x8x8 pool, flattened64-unit head,
zero-initialized3D micrometre correction with a smooth7um norm bound.
Movie0.001/0.999 normalization and full-resolution image reads reuse existing
infrastructure. AdamW3e-4, weight decay1e-4, SmoothL1 beta1um, batch32,
1200 steps, seed5801, final checkpoint only, FP32 with TF32/AMP off.

Training uses the actual vendored official7um one-to-one matches on22 original
pilot movies. All eligible known residual magnitudes remain represented.
Unmatched cells are unlabelled, never negative/background targets. Expanded
18x72x72 real-support crops permit exact integer jitter of z±1,y/x±4 voxels;
targets translate by the same displacement, and a jitter exceeding7um is
rejected without discarding the original example. No reflections or padded
boundaries. Sampling is uniform movie then uniform eligible known point.
Stored training crops are float16 for space; network math and inference inputs
are float32. This quantization is recorded, not claimed byte-identical to raw.

Train on44b6 and apply to6bba, and vice versa. The two biological embryos include
overlapping movie crops. Frozen public detector/head saw both embryos, so this
is opposite-embryo validation of the new component, not independent validation
of the entire pipeline or an estimate of hidden leaderboard improvement.

Inference applies to every non-gap-synthetic node with complete16x64x64 image
support. The actual `gap_synthetic` flag and unflagged interpolated nodes
inserted by `recover_strict_gap2` are passively observed by replaying the
unchanged C023 pipeline (node-object provenance, without graph edits).
Real readmission/peak nodes remain eligible.
No GT label, ambiguity measure or residual threshold determines runtime use.

## Controls and decision

`src/c058_localizer_baseline.py` reuses the original replay and actual C023 CSV
writer. The replay's final integer graph must equal C055's C023 control exactly.
Every node must have an identified inference-time origin. Official matching
uses tracksdata DistanceMatching and scoring uses the pinned organizer metric;
discrepancies from the old notebook replica are explicitly recorded.

Before the finite queue: two full-movie all-node zero-model extraction/inference
benchmarks establish time and exact graph identity. Label/crop audits report
support, synthetic flags, ambiguous alternatives and nonnearest assignments.
Official matches remain operational labels, not confirmed biological identity.

The finite queue reuses `run_last_days_local.Queue`: capture22, extract22,
label/crop/loss controls, two fixed fits, two opposite-embryo inference jobs,
official graph evaluation, actual writer/CSV/official evaluation verification,
signed analysis. Sources and all raw/cache/GT inputs are hashed before launch.
No Kaggle operations occur in this queue. No active waiting or parameter sweep.

Evaluation includes original-pair residuals (including lost/remapped matches),
known-identity losses/remaps/new matches, official edge/division TP/FP/FN and
adjusted-edge/total scores. IDs/counts/topology and excluded coordinates must
remain exact. Zero and learned graphs both pass the original writer and actual
official CSV evaluator, not just an internal replica.

Stop the recipe unless BOTH opposite embryos improve official adjusted-edge
score, original >3.5um paired residuals and edge TP, each with at least two
movie wins, and all22 total score improves. Passing warrants review for an
unchanged97 extension; no automatic deployment, T4 or submission. Any eventual
deployment requires a single fixed hidden-movie recipe and existing full
portable/writer/T4/source/model/CSV/version/quota/dedup controls.

State: `plan.json`, `benchmark.json`, `status.json`, `launch.json`,
`label_audit.json`, `official_rows.csv`, `paired_residuals.csv`,
`writer_verification.json`, `analysis.json`, `decision.json`, `output_hashes.json`.
Before actual launch there is no owned queue PID or completion ETA.
