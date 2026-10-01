# C055 continuation — guarded endpoint readmission (Lineage Forge V12 port) on C023

User direction 2026-09-29 (KST): after C054, apply ONLY V12's motion-predicted
endpoint readmission to C023, replacing C023's own readmission, verify locally
(22 then 75/97), and keep improving. C054 v1 was T4-verified by the agent
(experiments/candidates/c054_output_ensemble/remote/c054_v1/T4_VERIFICATION.json)
and submitted by the USER as 56648523 (02:51 KST); do not poll its score.

## Study (finite, resumable, no Kaggle writes)

- Driver `src/c055_guarded_readmit.py prepare|run|check_cache|study|analyse`,
  stage `src/c055_guarded_readmit_stage.py`, workspace
  `experiments/candidates/c055_guarded_readmit/` (README, plan.json with 5450
  pinned inputs, status.json, launch.json, logs/).
- Inputs: C023 notebook + existing C023 FP32 caches
  (`experiments/candidates/c032_temporal_context/e2e/control_fp32_*`), pinned
  harness `src/eval_pp_variants_local.py` (unchanged; it is a C054 pinned input).
- Arms: `control` (C023 as configured), `readmit_off` (READMIT_RADIUS_UM=0,
  diagnostic), `v12_guarded` (readmit off + V12 stage at V12's insertion point,
  fixed constants 0.94 / 3 um / 1.5 um / 1.5 um / margin 0.5 / bridges /
  min(100, 0.2% nodes)).
- Cache semantics verified on all 97 movies (`cache_semantics.json`): pooled
  detector local-max mask is threshold independent; dump scores are sigmoid at
  the peak; coordinates scaled by [1,4,4]; production node count == peaks>0.965;
  dedup is a no-op. Head-refined nodes sit up to 1.64 um from their raw peak, so
  existing nodes' own peaks are excluded by one-to-one identity (<=2.5 um)
  before V12's separation test (the single documented adaptation).
- Parity: harness CLI control rows == stored 2026-09-26 rows, in-process control
  == CLI rows, control final graphs == C046 `off` graphs (heldout12: 12/12).
- Pre-registered gate for the 75 extension: v12 vs control on all22 total>0,
  edge>0, both embryos >=0, surviving readmitted nodes >0.

## Interpretation guard (HANDOFF sections 12/18/21)

C015 = C012 + readmit OFF scored 0.951 < 0.952 on the LB although local
replays favoured readmit off (+0.0008 / +0.0022). The heldout12 result here
(readmit_off +0.0024, v12_guarded +0.0024 vs control, v12 vs readmit_off
-0.00006) is the same local pattern. Judge V12's OWN contribution as
v12_guarded minus readmit_off; a gain that comes only from disabling C023's
readmission is not new evidence and did not transfer before.

## Schedule

Launched 02:36:43 KST (second launch; first failed at check_cache and is
archived in state/c055_failed_launch_20260929/). ETA ~04:00 KST if the gate
passes (75 extension), else ~03:05 KST. One status/log check per wake; never
os.kill(pid, 0) on Windows. On completion: analysis.json / official_summary.csv
/ decision.json, then REVIEW.md with commands, hashes, tables and the
proceed/hold decision. Portable/T4 only if worthwhile after the guard above.

## Closure 2026-09-28T18:45:32.628255+00:00

C055 completed 9 jobs (03:43 KST), 316 artifacts hashed (manifest `f86e73057c7a3fa2…`). all97 v12_guarded +0.000260 vs
control but extension75 −0.000051 and 44b6_97 −0.002840; readmit_off +0.000431 all97 yet −0.002695 on 44b6. V12's own
effect (v12 − readmit_off) negative on 72/75 extension movies. HOLD; no portable/T4/submission. C056 (StrongUNet
candidates, same guard) failed the 22-movie gate (−0.000339 vs readmit_off, −0.001573 vs control on 44b6). Both
REVIEW.md files are final. No pending local or remote work; heartbeat not needed. C023/C024 0.954 final picks stand.
