# Public notebook freshness audit, 2026-09-27 13:23 KST

User provided a screenshot of five public notebooks, including optimized-biohub-max-score updated about3h earlier. The prior broad audit (2026-09-24) covered269 listings and55 selected distinct sources; the last general refresh (2026-09-26 09:16KST) covered271 listings. That does **not** establish that today's latest versions were reviewed.

All five latest public sources and metadata were pulled again using the existing notebook-radar Kaggle wrapper; no notebooks were executed and no output archives/weights downloaded. Existing score fetch was used. Source comparisons and hashes are in `state/notebook_radar/pulled/review_20260927_user_screenshot/`.

| Notebook | Fresh source comparison | Current visible score / best |
|---|---|---|
| raunakdey07/biohub-harmonic-fusion-v3 | All12 code cells identical to09-26 reviewed copy | .953 / .953 |
| kunaldesale2408/biohub-cell-tracking | All13 code cells identical to09-25 reviewed copy | .953 / .953 |
| anvithpothula/biohub-0-953-lb-original | All12 code cells identical to our x138 source; public head already used by C023/C024 | .953 / .953 |
| amanatar/biohub-geometric-fusion | All12 code cells identical to previously reviewed source | .948 / .948 |
| amanatar/optimized-biohub-max-score | **Changed; latestV6 had not been reviewed previously** | latest score absent / best .953 onV4 |

## Amanatar V6: actual differences

Public view metadata records V6, scriptVersionId353147685, updated2026-09-27 01:39:05UTC (10:39KST). Kaggle CLI status is COMPLETE; linked submission56594457 has no score exposed at inspection. The card's best0.953 belongs to **V4, scriptVersionId352554580**, not proof of V6 performance. An attempt to pull `/4` returned403; no scored-V4 source-equivalence claim is possible. Old records saying current-error/best.910 are historical observations and must not be used as today's result. Notebook text claiming0.965+ is an author label, not measured evidence.

Compared with x138, the substantive new mechanisms are:

1. **Primary-preserving detector fusion:** normalize the secondary detector logits to the primary's per-frame mean/std; compute the original weighted blend with calibrated secondary; then take elementwise `maximum(primary, blended)`. This is not C032's adjacent-window temporal-context averaging. Exact implementation was not found among the recorded prior tests. The lower detector threshold .960 (from .965), retention guard .75 (from .90), and altered short-track rescue are bundled with it, so score attribution would require an isolated matched e2e arm.
2. **Relaxed multi-step division test:** sister distance>=4um bypasses future divergence; otherwise inspect up to3 successor steps and accept either separation growth or absolute distance>=2.8um. Gates/caps also loosen. This exact code differs from old division knob tests, but no V6 LB evidence supports it. The two uppercase horizon globals used by the new branch are not assigned in the pulled cells (only environment values and CONFIG_DISPLAY fields exist); a branch-dependent NameError risk must be checked before executing any adapted version. CLI COMPLETE alone does not prove that branch was exercised. Earlier division precision problems remain relevant.
3. Further existing-knob changes: gap5.8um, motion flowK16/radius48um/z-weight.40, division gates/caps and short-track rescue. Several families were already negative; do not combine all changes and call the result a new validated method.

## Disposition

Four of the screenshot's five sources were already covered and remain unchanged; the latest optimized notebook was a freshness gap, now statically reviewed. The primary-preserving detector fusion is a distinct remaining experimental lead, not an established improvement or an immediate submission. The scoredV4 remains inaccessible. Preserve this lead for a separately registered isolated test after the active C038 extension/combination queue is idle; do not interrupt or edit its pinned sources. No experiment, upload, submission, dataset download or shared slot was used by this audit. C023/C024 remain unchanged.


## Follow-up: actual output and executed error reproduction

After the user's request for deeper investigation, selectedV6 actualT4 output was downloaded. Run log confirms the previously suspected missing-global error occurred on **all4** visible movies, with `repair_fallback=1` and basic ILP output. The official evaluator gives visible4 **0.9250646856464496**, not a hiddenLB score. Both missing names were independently reproduced by executing the published function branch; binding both fixes it. See `experiments/candidates/c039_public_v6_salvage/README.md` and `division_branch_regression.json`. The detector fusion did execute and is queued for isolated testing; broken fullV6 is not evidence that every component fails. Source and actual-output artifact hashes retained under `v6_actual_output/`.
