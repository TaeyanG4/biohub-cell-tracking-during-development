# Biohub Project Structure

Last updated: 2026-10-01 KST. **The competition is closed** (final private 0.918, rank 515 / 4,017). Start from `README.md` and `docs/`.

## Current state of the folder (cleaned 2026-10-01)

To free disk space, the local working folder was reduced to the contents of the public GitHub repository plus `AGENTS.md`. Everything else was deleted after the repository was pushed and checked.

| Kept (in the repository) | Deleted, and where to get it back |
|---|---|
| `README.md`, `README.ko.md`, `LICENSE`, `NOTICE.md`, `docs/` | `data/`: download the competition data from Kaggle |
| `src/**/*.py`, `tools/` sources | `artifacts/`, candidate model files: every submitted model is in our private Kaggle datasets (`taeyangg4/biohub-c012-v1284-head`, `biohub-v1284-heads`, `biohub-v1284-head-v4`, `biohub-fixed-models-c041`, `biohub-c053-division-transformer`, `biohub-c054-output-ensemble`, `biohub-c065/c067/c068-division-transformer`); study-only models were not kept |
| `experiments/candidates/*/` notebooks, READMEs, reviews, `decision.json`, stem lists; `experiments/submission_log.csv` | `tmp*/`: Kaggle kernel outputs, re-downloadable with `kaggle kernels output` while Kaggle keeps them |
| `reports/**/*.md`, replay score summaries | per-run CSVs, feature dumps, caches, logs, `state/` run state and the notebook-radar database |
| `state/**/*.md` review and decision notes, `intel/**/*.md` | `archive/`, `external/` (papers, third-party repositories), `vendor/` (organisers' metric code, on GitHub), `kaggle_notebooks/` (public notebooks, on Kaggle), `.venv/`, BiohubViewer build output |
| `HANDOFF.md`, this file; `AGENTS.md` stays local only | |

Most scripts in `src/` expect the deleted folders (`data/`, `artifacts/`, `tmp/`, `vendor/`). The rest of this file describes the working folder as it was during the competition (last substantive update 2026-09-29); paths in it may no longer exist.

This document is the canonical filesystem map for the active Biohub competition workspace. `HANDOFF.md` remains the source of truth for scientific status and experiment priority.

Current localization study (2026-09-29): `src/c060_track_localizer_study.py`
orchestrates existing official tools/Queue; `c060_track_localizer_data.py` reads
actual predicted3frame contexts using existing image IO; `c060_spatial_localizer_model.py`
implements the conditional spatial map. Artifacts under
`experiments/candidates/c060_spatial_track_localizer/`, continuation
`state/c060_continuation.md`, prior diagnostics `state/localization_diagnosis_20260929/`.
User-command Windows notifier: `tools/notify_background_completion.ps1`.

## Top-level layout

```text
biohub-cell-tracking-during-development/
├── HANDOFF.md                 # current scientific/competition source of truth
├── AGENTS.md                  # agent operating rules and tool reuse index (READ FIRST)
├── PROJECT_STRUCTURE.md       # this file
├── run_radar.bat              # 1-click launcher for Kaggle Notebook Radar
├── public-radar.html          # web browser launcher for Kaggle Notebook Radar
├── public-radar.vbs           # silent background launcher for Kaggle Notebook Radar
├── data/                      # official/local competition data; cleanup must not modify casually
├── experiments/               # ACTIVE experiment work, candidates, and manifests only
├── reports/                   # ACTIVE/current research reports and large R3 artifacts
├── src/                       # operational research/evaluation scripts
├── artifacts/                 # ACTIVE model/support artifacts
├── kaggle_notebooks/          # current notebook references and development notebooks
├── intel/                     # intelligence registry, source deltas, paper notes, idea cards
├── external/                  # external repositories and immutable paper PDFs
├── tools/                     # local utilities (BiohubViewer, notebook_radar)
│   ├── BiohubViewer/          # 3D trajectory & prediction visualizer GUI
│   └── notebook_radar/        # Kaggle public notebook collector & score dashboard
├── state/                     # persistent state (notebook_radar DB, listings)
├── vendor/                    # vendored dependencies/source needed by the project
├── .venv/                    # current local Python environment
└── archive/                   # historical/closed material, not part of active working set
```

## Active baseline and notebook paths

### B0 - immutable score anchor

Working copy:

`experiments/b0_public0947_exact_source/biohub-cell-tracking-0-947-lb.ipynb`

Immutable pulled source:

`kaggle_notebooks/latest_review/reyhan_0947/biohub-cell-tracking-0-947-lb.ipynb`

Do not edit the immutable source in place. Candidates must be compared back to B0.

### Active submission-candidate workspace

All notebooks that we actively modify with the intent of eventually becoming Kaggle submission candidates live under:

`experiments/candidates/`

Active candidates:
- `c001_r3_temporal/` — temporal-history extension of R3 target-parent association (initialized from B0).
- `c002_node_rescue/` — selective node rescue using StrongUNet low-threshold detector peaks.
- `c003_medal_frontier/` — validated breakthrough candidate targeting >= 0.948 (Rank <= 213, medal range) with tight55, calibrated safe-division expansion, gap-2 recovery, and expanded validator grid.
- `c010_x138_flow_fusion/` — SUPERSEDED (x138 + C004 wide division envelope; the envelope costs -0.0070 in replay).
- `c016_division_scorer/` — NO-GO (2026-09-24): C012 + learned division scorer after x138's rule. Keeps the local C012-config inference of 75 division-rich train movies (`e2e/head_v1_b00..02`), labelled candidate CSVs for 97 movies (`candidates_*.csv`, 48 features, labels mirroring the official `score_divisions()`), training logs (`train_logs/`) and the tooling test area (`smoke/`). Tools: `src/division_scorer_stage.py`, `src/division_candidates_local.py`, `src/division_scorer_train.py`, `src/build_c016_candidate.py`, `src/verify_c016.py` (see AGENTS.md section 5).
- `c011_x138_zero/` — HELD, never submitted: `anvithpothula/biohub-x138` (public 0.953) with the private V1284 head switched off. Its patched `tracking_repo` (`tmp/c011_output/`) is the base of the local inference tools.
- Final phase (2026-09-23 .. 26; every folder has a README; LB in brackets):
  - `c012_v1284_head/` (0.952) — our V1284 head v1 on x138; README, capture pairs (`pairs/`, `pairs_heldout/`), heads v1-v4 (`heads/`), datasets, e2e runs, `heldout_stems.txt` / `confirm_stems.txt`. `c013_v1284_head_a05/` (0.948), `c014_v1284_head_noreadmit/` (0.949), `c015_v1284_head_v1_noreadmit/` (0.951).
  - `c017_jump_stabilized_relink/` (0.952), `c018_frozen_catchup_prior/` (0.952), `c019_ilp_seeded_flow/` (not pushed), `c020_stabilize_all_pairs/` (0.953), `c021_ilp_edge_restore/` (0.953), `c022_stabilize_all_restore/` (0.953).
  - **`c023_x138_head_stabilize_restore/` (0.954) and `c024_head_ensemble/` (0.954) — recommended final picks.**
  - `c025_pp_settings/` (0.952), `c026_head_v4/` (not pushed), `c027_head_ensemble3/` (0.953), `c028_x138head_pp_settings/` (0.951), `c029_ensemble_pp_settings/` (0.950), `c030_adabn/` (not pushed), `c031_division_cnn/` (classifier study, no notebook).
  - Experiment work areas that share a candidate prefix: `c017_ilp_division/` (ILP division-weight runs), `c024_tta_pool/` (TTA-pooling runs), `c025_fusion_weights/` (inference-knob runs). `c016_division_scorer/` also holds the 75-movie e2e runs and `stems_b0{0,1,2}.txt`.

### Post-processing replay

- `src/eval_pp_variants_local.py` — replays a 12-cell HF/x138-lineage notebook's own post-processing on cached ILP graphs (default: C004's Kaggle run, `tmp/c004_log/tracking_repo/predictions`) and scores variants with the official formula; calibrated against C004's Kaggle validator.
- `reports/pp_replay/` — variant JSONs, per-movie CSVs, summaries, logs.
- Harness options added 2026-09-24/25: `--round-coords`, `--pred-root` / `--lowdet-dir`, `--env`, `--reinject-ilp-forks`, `--stabilize-relink`, `--ilp-edge-restore`, `--weak-edge-filter` (AGENTS.md section 4). Validation protocol = 97 movies: held-out 12 + confirm 10 (`experiments/candidates/c012_v1284_head/`) + 75 (`experiments/candidates/c016_division_scorer/stems_b0*.txt`).

### Local inference, heads and final-phase builders

- `src/c037_transformer_study.py`, `src/c037_transformer_runtime.py` — C037 production-input frozen-detector Transformer-only pilot, exact capture/off-control, cross-embryo training, and three predefined actual official replay arms including original/learned parameter blending. Uses the existing inference/replay/queue,24 separate training movies; `experiments/candidates/c037_transformer_finetune/README.md` defines data, limits and small-effect/combination follow-up. No GT-negative relabeling or hidden embryo routing; no direct Kaggle operation.
- `src/c038_complementary_study.py`, `src/c038_complementary_stage.py` — C038 complementary appearance/motion official graph replay, all eligible predictions (no GT source whitelist), protected divisions/gaps and collision-safe reconnect/swap rules. `experiments/candidates/c038_complementary_fusion/README.md`; prepare locally, run after C037 GPU queue. Local replay notebooks are explicitly not portable Kaggle candidates. Reuses existing scorer/implementations; small effects get actual graph verification before closure.
- `src/c038_followup_local.py` — finite C038 extension75 and fixed C037 early150 combination22 queue, using unchanged models/stage and the existing official harness. Passive graph capture,66-row pilot parity,75+22 off-controls, official97 aggregation and descriptive edge overlap; no new scorer or hidden-embryo routing. State in `c038_complementary_fusion/followup_extension/`; all generated notebooks local-only.
- `src/c039_public_salvage.py` — user-requested isolation of AmanatarV6 detector max fusion/calibration and repaired division function on C023; existing inference, official replay and finite queue, two-embryo exact off smoke. Prepared after C038; never contend with its running GPU queue. PublicV6 actual output has repair fallback errors and is not proof of0.965+.

- `src/local_registration_probe.py` — C036 training-free local image-motion diagnostic (`run|analyse`); reuse global phase correlation/frame reader and C035 original-coordinate caches, add two-scale local NCC/cycle confidence. CPU background study, no new scorer or graph edits. Fixed gate/provenance/status/results in `experiments/candidates/c036_local_registration/README.md` and study directory. User requested continuing after C035 closure.
- `experiments/candidates/c036_local_registration/FINAL_REVIEW.md`, `decision.json` — C036 closed: all22 movies completed,958 trusted estimates, conservative diagnostic net+1/0 by embryo; gate failed. Image boundaries excluded52.9% of groups. No graph-score/T4/submission; preserve caches and scored anchors, stop follow-up.

- `src/reid_augmented_local.py` — C035 GT-supervised appearance encoder study (`run|analyse`), reusing C034 replay/recorder and C031 crops/backbone. New training excludes all97 evaluation movies; basic/weak-augmentation controls with cross-embryo diagnostic on original C023 candidate groups. Study plan, source hashes, real crop smoke, patches, models and diagnostics under `experiments/candidates/c035_augmented_reid/`. No replacement graph scorer or automatic Kaggle operation; see its README for fixed gate and bounded background follow-up.
- `experiments/candidates/c035_augmented_reid/FINAL_REVIEW.md`, `decision.json` — C035 closed after four completed fits and all22 baseline checks. Augmented conservative diagnostic net +1/-3 by test embryo; neither learned arm passes. No graph integration, official-score delta, T4 or submission; cache retained and follow-up stopped.

- `src/build_temporal_context_candidate.py` — C032 C023-based temporal-context repo/notebook builder (`experiments/candidates/c032_temporal_context/`), exact encodes with bounded lookahead. `run_kaggle_predict_local.py --t4-fp32` provides an explicit local Ada FP32/math-attention policy; baseline and experiment must use the same policy.
- `src/verify_temporal_context_local.py` — actual four-frame dual-seed/head inference smoke and exact-control checks on both embryos.
- `src/build_structured_candidate.py`, `src/structured_trajectory_stage.py` — C033 public fixed structured-assignment builder/adapter, `experiments/candidates/c033_structured_trajectory/`, public code/model/NOTICE in `artifacts/public_structured_trajectory/`; use the existing replay harness `--structured-trajectory` option.
- `src/run_last_days_local.py` — finite one-GPU C032/C033 experiment queue built on existing inference/replay, state and results in `state/last_days_local_20260926/{status.json,RESULTS.md,logs/,replay/}`. Never pushes/submits; the user handles T4 verification/submission.
- `state/last_days_local_20260926/PILOT_REVIEW.md` — completed 22-movie pilot results and reviewed continuation. C016 comma-list parser fixed; `--extra-extension-arm mean_det` allows a reviewed positive-on-both-sets arm alongside automatic future_det. Failed original run and driver archived in `review_before_resume_20260926/`; exact queue-only hash migration in `resume_review.json`. Extension validates 75 unique movies disjoint from pilots; no completed inference is repeated.
- `state/last_days_local_20260926/FINAL_REVIEW.{md,json}`, `FINAL_REVIEW_summary.csv` — completed C032/C033 decision: neither temporal finalist improves extension75/all97; both lose edge score in each embryo group. Existing official aggregation reused and original split summaries reproduced; no new scorer. No push/submission. Durable follow-up state in `auto_submission_state.json`.
- `src/reid_probe_local.py` — C034 read-only relink candidate capture and cross-embryo appearance diagnostic, reusing arnav170 public descriptor code and existing C023 replay/scoring. `smoke|run|analyse`; results/source hashes/provenance/pair caches under `experiments/candidates/c034_appearance_reid/`. Unknown targets are not negative training labels; diagnostic ranking is not an official score or a submission candidate.
- `experiments/candidates/c034_appearance_reid/FINAL_REVIEW.md`, `decision.json` — C034 closed for insufficient known negatives (6/10 by embryo); all 22 extraction/control checks passed but both model fits skipped. No ReID graph-score result or submission. Retained pairs can be reused only under a separately justified study, not as evidence ReID failed.

- `src/run_kaggle_predict_local.py` (local T4-equivalent inference + ILP; `--v1284-head "a;b"` for ensembles), `src/v1284_capture_local.py`, `src/v1284_head_train.py`, `src/compare_v1284_heads_on_pairs.py`, `src/compare_kaggle_local_outputs.py`, `src/evaluate_local.py`.
- Builders: `src/build_c017_candidate.py` + `src/verify_c017.py`, `src/build_c018_candidate.py`, `src/build_c019_candidate.py`, `src/build_c021_candidate.py`, `src/build_c023_candidate.py`, `src/build_head_ensemble.py`, `src/build_env_variant_candidate.py`, `src/build_c030_candidate.py`; patched local repos: `src/build_tta_pool_repo.py`, `src/build_adabn_repo.py` (usage in AGENTS.md section 5).
- Analyses: `src/frame_motion_audit.py` (frozen frames / jumps), `src/node_budget_*.py`, `src/division_edge_prob_check.py`, `src/division_crops_extract.py` + `src/division_cnn_train.py`.
- Kaggle kernel outputs of every pushed candidate: `tmp/c0NN_output/` (submission.csv, patched tracking_repo, logs where the download worked).

### B1 - research/development notebook

`kaggle_notebooks/sjlee_dctta/biohub-lf-dctta-v020.ipynb`

Use B1 for validator/post-process experimentation when convenient, but do not silently promote it over B0.

## Active R3 research

Primary code is kept in `src/` so existing imports and operational commands are not broken by cosmetic reorganization.

Key R3 paths:

```text
reports/
├── research_20260913/         # R3 normalized-HOCT features/rankers and retained large feature data
├── research_20260914/         # follow-up geometry/listwise/full-train diagnostics
└── public_baseline_review_20260917.md

experiments/
├── candidates/
│   └── c001_r3_temporal/      # editable development/submission candidate initialized from B0
├── r0_pack_full_train_gt/
├── r3_fulltrain_featuregen/
├── r3_gt_fulltrain_featuregen/
├── r3_gt_fulltrain_featuregen_cpu/
├── r3_gt_kaggle_wheels/
├── r3_hoct_v1_dataset/
├── colab_r3_smoke/
├── colab_transfer/
├── exp_train_inventory/
├── metric_public0947_clean/
└── metric_public0947_reyhan/
```

The dated report folder names are retained deliberately because current scripts and handoff references point to them. They should not be renamed during the active competition unless all references are migrated together.

## Active support artifacts

`artifacts/` now contains only artifacts that are relevant to B0, B1, R3, or near-term R4 work:

```text
artifacts/
├── anvithpothula_v1284_head_s075/   # x138's public V1284 head (CC0, sha 625a0d93), used by C023 / C024 / C027
├── pilkwang_support50/
├── pilkwang_temporal_seed314159/
├── pilkwang_deepcenter/
├── hoct_general_v1_research/
├── full_train_gt_pack/
├── hengck_point_detector/
├── public_0947_reyhan/
├── sjlee_dctta_output/
└── lineage_forge_output/
```

## Kaggle notebook organization

The active notebook surface is intentionally small:

```text
kaggle_notebooks/
├── latest_review/             # current audited frontier/reference notebooks
├── sjlee_dctta/               # active B1 development reference
└── harmonic_fusion_v3/        # current frontier reference retained outside latest_review
```

Older pulled notebooks were moved to the cleanup archive.

## Intelligence and papers

```text
external/papers/raw/           # immutable PDF originals
external/papers/manifest.csv   # PDF provenance/hash registry

intel/
├── intelligence_policy.md
├── papers.csv
├── paper_notes/
├── idea_cards/
├── sources/
│   └── legacy_recon/          # older Kaggle recon/raw source snapshots moved out of reports/
├── source_registry.csv
├── daily_delta.md
└── state.json
```

## Archive policy

Historical material is under:

`archive/2026-09-17_cleanup/`

This includes closed experiments, old metric sweeps, superseded notebook pulls, obsolete outputs, snapshots, and legacy environments. Archive material is preserved for provenance but is not part of the active experiment surface.

Do not copy archived candidates back into `experiments/` unless the corresponding hypothesis is explicitly reopened in `HANDOFF.md`.

## Data safety rule

The `data/` directory was excluded from modification during the 2026-09-17 cleanup because the official competition-data download was running. An initial size inventory had already read directory metadata before the exclusion instruction. After that instruction, no further `data/` traversal was performed, and no cleanup command moved, renamed, or removed content beneath `data/`.

Future cleanup of `data/` should be a separate operation after download completion and integrity verification.

## What was intentionally deleted

Only clearly redundant/reproducible storage was deleted during the cleanup:

- `experiments/hoct_linux_venv/` - approximately 6.1 GiB; experiment-local Linux virtual environment, reproducible and not the pinned source of truth.
- `reports/research_20260913/r3_visible4.npz` - approximately 1.08 GiB; SHA256-identical to retained `r3_visible4_tiled.npz` (`38A5C68E58BB883D6A25EE290F8ED914BA547A4F41FAA1A9310412694F05B2C5`).
- eight `experiments/colab_transfer/6bba_05b6850b_smoke.part_*` files - approximately 345 MiB total; retained complete `6bba_05b6850b_smoke.tar` has exactly the same total byte length.

Approximate disk space released: 7.3 GiB.

For move details and provenance, see `archive/2026-09-17_cleanup/CLEANUP.md` and `archive/2026-09-17_cleanup/moved_items.csv`.

## C040 late600 fixed combination

src/c040_late600_combo.py: thin prepare/run/analyse orchestration over the existing C038 graph capture/audit, official replay and Queue; workspace experiments/candidates/c040_late600_combo/. Cached late600 plus fixed appearance/agreement; no new scorer or models.

## C040 appearance extension75

src/c040_extension_local.py: prepare/run/smoke_check/analyse thin orchestration of fixed late600 inference, original replay scorer, two full-movie reproductions and75 BASE/off controls. Workspace experiments/candidates/c040_late600_combo/extension75/.

## C041 fixed global models

src/c041_fixed_models.py: self-contained extraction of existing appearance stage, fixed model packaging, and existing Queue/inference/replay orchestration for22-movie three-arm pilot. Workspace experiments/candidates/c041_fixed_models/.

## C042/C043 portable fixed-model candidates

src/build_fixed_model_candidates.py: portable notebook/dataset packaging and existing Queue/inference/replay/notebook-writer/official-evaluator verification. State state/c042_c043_portable/; notebooks experiments/candidates/c042_fixed_transformer/ and c043_fixed_transformer_appearance/. No Kaggle writes in the script. Dataset mounting is SHA-pinned; no embryo routing or local audit imports in submitted notebooks.

## C044/C045 C024 fixed-model combination

src/c044_c024_fixed_study.py: thin prepare/run/check_smoke/analyse orchestration using the existing Queue/inference/replay/official aggregation and passive graph audit. Study experiments/candidates/c044_c024_fixed_study/; new portable candidates c044_fixed_ensemble_transformer/ and c045_fixed_ensemble_transformer_appearance/. C024's scored head ensemble plus unchanged fixed C041 Transformer, with/without appearance. Original/off two full-movie parity, fresh C02422 controls, two fixed learned22 arms, comparison with C042/C043. No new training or scorer; no Kaggle writes in this script.


## C044 management-process recovery

state/c044_queue_recovery_20260928/recover.py: one-off read-only Windows process-handle adoption and resume of existing C044 Queue after accidental manager exit. Preserves exact successful child work,original sources and original budget. See README/request/archived state; not a scorer or experiment driver.

## C044/C045 portable verification

src/verify_c044_candidates.py:7-job thin orchestration of existing replay,actual notebook writer and official evaluator. State state/c044_c045_portable/. Reuses completed C044 caches,compares embedded predictor/head/runtime against actual study,no new scorer or fitting.


## C046 fixed appearance-only extension

src/c046_fixed_appearance_extension.py: five-job thin orchestration of existing C041 notebook extraction, Queue, official replay/aggregation and C038 graph audit. Workspace experiments/candidates/c046_fixed_appearance_extension/. Exact22 fixed-model reproduction then75 extension using original C023 caches, signed97 results, immutable hashes; no new scorer/inference/training.


## C046 final-analysis recovery

state/c046_analysis_recovery_20260928/recover.py: one-off reuse of original analysis with empty confirm10/44b6 group skipped and explicitly recorded unavailable. Original source/plan and397 replay/graph artifacts remain pinned; no model/replay/scorer replacement. Archives/results in same folder.


## C046 portable candidate

src/build_c046_candidate.py: original-C023+fixed-appearance packaging and four-job verification through existing Queue/replay/actual notebook writer/official evaluator. State state/c046_portable/;notebook experiments/candidates/c046_fixed_appearance/. Existing private C041 asset,no Transformer fine-tuning,no new inference/scorer.

## C047 hard-example data study

src/c047_hard_example_study.py: thin orchestration of existing C035 crop/training functions and C041 fixed runtime with original C023 replay, Queue and C038 graph audit. Adds29 known-label close-competitor training movies outside official97 to24 preserved C035 caches. One fixed data recipe,two1200-step encoders,no new scorer or thresholds. Workspace experiments/candidates/c047_hard_example_appearance/;40 finite background jobs,exact controls,signed C023/C046 comparisons. Feasibility evidence state/improvement_feasibility_20260928/.

## Whole-embryo split correction and C048

state/split_audit_20260928/REVIEW.md: confirmed image-level crop overlap and historical conclusion audit. C047 stopped before training,original files retained. src/c048_embryo_holdout_study.py reuses C035 training/C041 runtime/C038 graph audit/existing replay and Queue for32 finite jobs:190 eligible movies across2 training folds,each evaluated on opposite embryo only,97 cached official controls. Workspace experiments/candidates/c048_embryo_holdout_appearance/. No pooled model in outer-fold evaluation,no new scorer,no deployed embryo routing. Original Kaggle anchors/parity preserved.

## C049 corrected division diagnostic

src/c049_division_embryo_audit.py:4-job orchestration of existing C031 fit/predict and standard sklearn classification metrics,whole-embryo split,known-GT/crop/fold/hash checks. Workspace experiments/candidates/c049_division_embryo_audit/. Corrects historical within-embryo CV and mislabeled precision@recall claim;no new competition scorer,graph integration or automatic deployment. Raw positive-crop coverage audit in state/split_audit_20260928/division_audit/coverage.json.

## C050 corrected C016 candidate-scorer diagnostic

C050 final review: `experiments/candidates/c050_division_candidate_audit/FINAL_REVIEW.md`; scoped NumPy scalar checkpoint-analysis recovery: `state/c050_analysis_recovery_20260928/`. Fixed learner closed without graph integration after corrected whole-embryo evaluation.

src/c050_division_candidate_audit.py:5-job thin orchestration of existing C016 load/fit/predict/per_parent and average_precision,whole-embryo folds on existing v3 candidate CSVs,immutable manifests and predictions. Workspace experiments/candidates/c050_division_candidate_audit/. Fixed recipe/threshold,no new official scorer;C012-derived features cannot directly qualify a C023 submission. Estimated-completion-first monitor scheduling.

## C051 fixed boundary-support registration diagnostic

- Driver: `src/c051_boundary_registration.py prepare|run|smoke|extract|analyse|verify`.
- Evidence: `experiments/candidates/c051_boundary_registration/README.md`, `plan.json`, `status.json`, `support/`, `analysis.json`, `artifact_hashes.json`.
- Reuses C036 NCC, original-coordinate extraction and diagnostics plus existing Queue. Minimum feasible real-image context shift only after an original boundary rejection; no padding, confidence relaxation, training or new scorer. Exact original controls/nonboundary parity; one25-job finite run. No submission qualification from annotated-source diagnostics.

