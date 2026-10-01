# C055 guarded endpoint readmission (Lineage Forge V12 port) on C023

Study of ONE fixed component: V12's motion-predicted endpoint readmission
(score >= 0.94, step <= 3 um, residual <= 1.5 um, separation >= 1.5 um,
ambiguity margin 0.5, bridge preference, budget min(100, 0.2% of nodes))
replacing C023's radius readmission (0.965 / 4 um).  Detector, V1284 head,
ILP (division weight 1.2), stabilized relink, restore and every other stage are
C023's own, replayed by the pinned `src/eval_pp_variants_local.py` on the
existing C023 FP32 caches.  Stage: `src/c055_guarded_readmit_stage.py`;
driver: `src/c055_guarded_readmit.py`.

Arms: `control` (C023 as configured), `readmit_off` (C023 readmission off,
diagnostic only), `v12_guarded` (readmission off + V12 stage).  Each split first
reproduces the stored 2026-09-26 C023 control rows with the harness CLI, then the
in-process loop must equal the CLI rows and the C046 `off` final graphs exactly.

Coordinate semantics: raw peaks belonging to existing (head-refined) nodes are
excluded by identity before V12's separation test (documented in the stage).
Outputs: `replay/`, `study/`, `graphs/`, `cache_semantics.json`, `parity_*.json`,
`analysis.json`, `official_summary.csv`, `decision.json`, `REVIEW.md`.
No Kaggle operation is performed by this queue.
