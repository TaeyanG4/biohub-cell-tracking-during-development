# Biohub public baseline review - 2026-09-17

## Decision

Use the exact public 0.947 notebook as the new immutable public baseline:

`reyhanksatria/biohub-cell-tracking-0-947-lb`

Local source:

`kaggle_notebooks/latest_review/reyhan_0947/biohub-cell-tracking-0-947-lb.ipynb`

The previous local reconstruction `exp_dctta_lite_det0965_public0947` scored
0.946 on Kaggle submission `56209758`, so it is retained only for provenance and
comparison.

## Candidate review

### ACCEPT - Reyhan 0.947 exact source

- Explicit verified progression: 0.933 -> 0.934 -> 0.939 -> 0.941 -> 0.946 -> 0.947.
- Keeps the same UNet3D + node-transformer + ILP lineage.
- 0.946 -> 0.947 change is narrow and auditable:
  - secondary association feature TTA
  - secondary feature-TTA blend = 0.75
  - DeepCenter TTA
  - DeepCenter safe-division threshold 0.20
- Clean graph pipeline; no negative-time synthetic hub/fork augmentation found in the reviewed source.

### ACCEPT AS REFERENCE - Igor Harmonic Fusion V30

- Public score 0.947 under the corrected metric.
- Canonical public source for the harmonic bidirectional association lineage.
- Useful as an independent source-level cross-check against the Reyhan reproduction.

### ACCEPT AS VALIDATION REFERENCE - Lineage Forge

- Public score 0.947.
- Adds a held-out train-movie post-process sweep using the official metric formula.
- The selected `motion_relink_tight_um=5.5` improved its local proxy from 0.9490 to 0.9511.
- Better used as a validation/ablation reference than as a different model family.

### REJECT - `muhammaddanyalmalik/cell-tracking`

- Contains synthetic hub/fork augmentation with negative time and coordinates
  (`t=-1000`, coordinates around `-10000`).
- Its apparent score is not a clean representation of tracking quality.
- Do not use as a baseline or source of post-processing logic.

### REJECT - `anvithpothula/biohub-0-95`

- Same negative-time / negative-coordinate hub/fork augmentation pattern.
- This belongs to the metric-exploit lineage and is not an acceptable clean baseline.

### REJECT AS UPGRADE - `raunakdey07/biohub-harmonic-fusion-v3`

- Last run is newer (2026-09-16), but the notebook itself describes a 0.939 public result.
- It is not evidence of a clean score above the established 0.947 frontier.

### WATCH - fast 0.947 / DivNet forks

- Recent notebooks advertise the same 0.947 score with shorter runtime and/or division additions.
- They may be useful for runtime headroom or diversity later, but they do not currently provide a verified score improvement over the clean 0.947 source.

## Why Code-page order is unsafe

The competition metric was patched and all submissions were rescored. Public
discussion records that some public notebook pages retained old, inflated
pre-patch scores afterward. Therefore `sortBy=scoreDescending` can place stale
or exploit-derived notebooks above genuinely better corrected-metric notebooks.

Baseline selection must therefore require:

1. corrected-metric public score evidence,
2. source-code audit,
3. no metric-exploit augmentation,
4. reproducible public artifacts,
5. runtime compatible with the 12-hour code-competition limit.

## Baseline gate B0

Before resuming model research:

1. Preserve the exact pulled Reyhan notebook unchanged.
2. Record notebook SHA256 and exact input dataset refs.
3. Create a private reproduction fork from that exact source only when explicitly authorized.
4. Run it without inference edits.
5. Compare resulting graph statistics and LB against the source's verified 0.947.
6. Diff the exact source against our 0.946 reconstruction to locate the missing 0.001.

## 0.946 reproduction discrepancy audit

The first source-level diff found a concrete difference that is more important than
the originally suspected TTA settings.

### What is already identical

The 0.946 reconstruction already has the public 0.947 source's documented narrow changes:

- `BIOHUB_DET_THRESHOLD=0.965`
- primary edge feature TTA enabled
- secondary edge feature TTA enabled
- secondary feature-TTA blend weight `0.75`
- DeepCenter D4 TTA enabled
- DeepCenter safe-division threshold `0.20`
- bidirectional harmonic association enabled
- motion relink tight radius `6.0 um`

Therefore missing secondary TTA / DeepCenter TTA is not a valid explanation for the
0.001 LB gap.

### Leading discrepancy

The reconstructed notebook explicitly contains:

`BIOHUB_VALIDATOR_ENABLE=0`

The exact pulled Reyhan 0.947 source does not disable the validator; its validator is
enabled by default and can choose among held-out post-processing candidates. This matters
because the Lineage Forge validation lineage previously found a `tight55` candidate
(`motion_relink_tight_um=5.5`) that raised its local proxy from 0.9490 to 0.9511, enough
to pass the source selector's improvement margin.

This does not yet prove that validator selection alone caused the 0.946 -> 0.947 LB
difference, but it is now the highest-priority causal hypothesis. The correct next test is
an exact-source B0 reproduction with the validator left on, followed by a one-factor
validator-on versus validator-off ablation. Do not mix this test with R3/HOCT changes.

No R3/R4 candidate should be called an improvement until it beats this exact-source baseline on the chosen evidence surface.
