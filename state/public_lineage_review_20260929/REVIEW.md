# User-linked Lineage Forge V12 review — 2026-09-29 KST

## Decision

**Useful as a bounded component reference, not a verified upgrade or a replacement for the running C054 study.** The genuinely changed component is motion-guarded endpoint readmission from observed detector peaks. C023 already has endpoint readmission, but its admission and reconnection rules differ. Retain this specific mechanism for a possible later isolated replay; no experiment, training, Kaggle push or submission was started by this review.

Keep C054's active queue, immutable inputs and existing schedule unchanged. Its completion review remains the next registered action. Do not import Lineage Forge's ILP division weight or local validator sweep along with the readmission function.

## Source and version evidence

- User URL: https://www.kaggle.com/code/flexonafft/biohub-lineage-forge-precision-tracking
- Queried the existing local radar SQLite record first. It was stale for this update: last score check Sep 28 01:35 UTC, last-run listing Sep 22, current score 0.946 / best 0.947.
- One targeted request using the radar's existing public view-model endpoint found **current version 12**, run **353597016**, status **COMPLETE**, Tesla T4. Kernel `updatedTime`: Sep 28 12:57 UTC / 21:57 KST. Run creation: Sep 28 13:01 UTC / 22:01 KST.
- The linked current submission is `56644638`; its response contains **no displayed score**. The displayed best **0.947 belongs to version 8**, run `348830165`, titled `--temporal accord score up`. Do not assign that score to V12 or call V12 a confirmed improvement.
- Immediately pulled the current source and metadata once using existing `notebook_radar.run_kaggle`. The source itself says this variant's public score is not yet measured.
- Latest notebook SHA256: `0188e9a7e789d70ed0e3195897200362555dca3c62557d4648f708b6a9d88fc3`.
- Previously reviewed local notebook SHA256: `f1c7d777e5c41a937ce4b679f65f2e0d5795aec411e1cccda99487a86b06dbb2`.
- Of 12 code cells, only cells **0, 4 and 5** changed relative to the saved reference. The old reference's precise version is not inferred. Nine cells, including validator / scorer / sweep cells, are byte-identical after source string/list normalization.

Evidence: `view_model.json`, `checked_at.txt`, `source_manifest.json`, `source.diff`, and `source/`. This review read code and small public result artifacts; it did not execute the downloaded notebook, download weights, or reproduce its inference.

## What changed, compared with our C023

| Mechanism | New Lineage Forge V12 | Existing C023 / interpretation |
|---|---|---|
| Candidate source | Capture observed pooled-detector peaks at score >= 0.94, while keeping the normal detection output unchanged | C023 already captures low detections at 0.3 and admits endpoint candidates at >= 0.965. Existing cache availability makes a later replay plausible, but score / coordinate / duplicate semantics must be checked first. |
| Endpoint admission | Requires one adjacent track context, context step <= 3 um, extrapolated-position residual <= 1.5 um, and separation >= 1.5 um from existing nodes | C023 admits free peaks within 4 um of an open start/end and then reruns motion relinking. V12's context / residual / ambiguity rules are a real component difference. |
| Ambiguity / budget | Rejects alternatives with cost margin < 0.5; prefers two-sided bridges; maximum min(100, floor(0.002 * node count)) additions | These guards could control the risk of admitting the extra 0.94–0.965 score range. Benefit remains unmeasured. |
| Edge placement | Directly appends one or two edges around the admitted node after motion relinking, before gap / division processing | C023 reruns its stabilized motion relinker after adding nodes and later restores ILP edges. Interaction with these stages must be tested on final graphs. |
| ILP division weight | 1.2 -> **0.4** compared with the saved reference | C023 uses 1.2. Historical 0.7 / 0.5 trials produced many false forks and degraded final metrics (HANDOFF section 21). The exact 0.4 setting is not claimed to have been tested; lowering this weight has poor prior evidence. |
| DeepCenter division threshold | 0.20 -> 0.25 compared with the saved reference | A setting change, without isolated improvement evidence here. |
| Model / coordinate head | Existing public detector / Transformer assets; no V1284 block in the retrieved source | This is not a newly trained model or the full C023/C054 stack. Copying the whole notebook would replace our scored head and graph improvements. |

The source calls the endpoint stage an independent implementation inspired by discussion `743929`. That attribution is the author's claim; the discussion itself was not audited here. The added nodes are observed detector peaks. No GT deployment whitelist or synthetic negative-time node mechanism was found in the changed component.

## What the public run actually demonstrates

Retrieved only the small `ppsweep_selected.json` and `ppsweep_results.csv` artifacts linked to run 353597016:

- Base eight-movie proxy: **0.9479624763164339**.
- Selected `tight55` proxy: **0.950017415688161**; override `MOTION_RELINK_TIGHT_UM=5.5`.
- Selected adjusted-edge Jaccard: **0.9269404926112379**; division TP / FP / FN: **3 / 1 / 9**.
- Those eight stems are already in our familiar heldout-12 evaluation set. They are not additional independent data.
- This comparison measures the postprocessing selection within V12. All these rows share its readmission / inference configuration; there is **no readmission-off control**. Neither the proxy difference nor the displayed historical 0.947 establishes the new component's gain.
- COMPLETE status and saved sweep artifacts are evidence that a public run completed. This review has not independently verified its final CSV, full runtime logs, fallback behavior or local/T4 parity.

Do not compare the 0.9500 local proxy directly with our 0.954 public LB or C054's future visible-4 score. Do not call sparse-GT unmatched detections biological false positives just because the public table labels them `spurious_pred_nodes`.

## Possible follow-up after C054 review

If a new finite study is justified, use the existing C023 caches and official replay harness to evaluate **one fixed guarded endpoint-readmission component** with unchanged detector, ILP weight, head and all other postprocessing. First establish cache coordinate / score / duplicate semantics and exact off-control graph parity. Preserve C023's original readmission as the control; explicitly define whether the guarded function replaces that stage, so admission is not accidentally applied twice.

Review actual final-graph changes and official edge / division TP, FP and FN on 22 movies and both embryos, with survival through gap repair / division / restore. Added-node counts alone are not gains. Extend only with worthwhile graph evidence; any eventual deployment still requires portable / actual T4 parity, fresh shared quota and deduplication. No threshold or weight sweep is proposed. No such study is currently launched or promised by this source review.
