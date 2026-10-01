# C052 division-bearing production-input Transformer

Registered 2026-09-28 at the user's request following the full HANDOFF review.

## Fixed question and recipe

Does correcting C037's zero-division training coverage improve real C023 graphs
while preserving ordinary links? Frozen primary UNet/head and secondary model;
unchanged production inputs, known-only labels, AdamW lr1e-5/wd1e-4, cosine600,
seed3701, teacherKL0.1, division weight2, dropout off, one step600 checkpoint.
No probability, ILP, relink or division-threshold changes.

Two WHOLE-EMBRYO folds. A model trained on 44b6 is evaluated only on 6bba and
vice versa. Movie-ID exclusion alone is invalid. Frozen public models already
saw both embryos. These are component transfer diagnostics, not an independent
validation of the whole pipeline. Hidden deployment must use a fixed model.

## Data and sampling fixed before evaluation

- Original C03724 production packets/known-only labels retained in place and hashed.
- Capture all87 division-containing movies using the original C037 capture hook
  and the existing full C023 inference command. All151 annotations are audited;
  actual usable production division windows/targets must be measured after capture.
- Unknown detections stay in attention context and never become GT negatives.
- 300 odd training steps cycle shuffled division-context clusters; 300 even steps
  sample the original24 ordinary windows uniformly by movie then window.
- Exact informative raw3-frame image contexts with +/-1 voxel centre offsets
  identify some shared contexts. Do not deduplicate by reused GT IDs. Merge
  co-occurring event clusters for window sampling. Every usable cluster is sampled.
  Unresolved near-duplicates remain; cluster counts are not independent events.
- Reuse original C037 train() by an asserted sampling-only source transform.
  Objective/optimizer/forward/checkpoint/parity/frozen-teacher checks are unchanged.
  Full600 sampled-batch trace proves that division supervision actually ran.

## Finite queue and evidence

`python src/c052_division_transformer.py prepare`, then `run` hidden, one GPU queue.
14 jobs,5-hour new-job start budget, existing Queue process timeout. Estimated
150minutes from launch; exact launch/ETA in launch.json. No agent active waits.

Real original/capture smoke,87 exact detection+ILP controls, actual label audit,
two fits, whole-embryo checkpoint proof,22 original off replays/final graph parity,
one learned22 inference/replay arm and existing official aggregation. Signed
C023 and C037 late600 comparisons on22/splits/embryos. Passive ILP/motion/
pre-restore/final graph captures execute the unchanged notebook functions.
Primary feature packets and fused edge caches are retained for a division-link
survival audit before deployment. Input/output hashes verified, original sources
immutable. Local audit notebooks contain imports and MUST NOT be pushed.

## Completion and user-authorized continuation

Review all signed results, ordinary links, division TP/FP/FN, graph changes and
effective coverage. Small consistent improvements may justify the fixed75-movie
extension; no old >=5 gate. No checkpoint or threshold search. Compare against
previous late600 to establish added value of the data change. Probability gains
alone do not justify graph edits or submission. If promising, reuse existing
fixed deployment/portable builders, notebook/harness parity and actual T4 checks.
Submit worthwhile verified candidates autonomously after fresh shared5/day quota,
exact-version/hash proof, zero fallback/degradation and submission dedup.
C023/C0240.954 anchors/final picks remain intact. No claim0.955 is achieved.

Latest scheduling request: first follow-up10minutes BEFORE the phase ETA, then
every20minutes while unfinished; update the same heartbeat/ETA for each new phase.
At completion review and continue supported work, report meaningful changes,
failures or required action, and remove the heartbeat when no justified work is
pending. Older first-at-ETA policies are superseded.
