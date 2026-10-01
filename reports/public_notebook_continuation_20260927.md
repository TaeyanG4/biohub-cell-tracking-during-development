# Public notebook continuation audit, 2026-09-27

Used the existing local SQLite radar first, then one existing crawl cycle at07:34:56UTC/16:34:56KST (limit150 per listing sort,4 workers). Gathered232 unique listings,232 successful score checks,1 new listing,0 score improvements; accumulated DB contains272 entries. This is a bounded refresh, not proof that every public version was inspected. Four new/changed listing records were source-only pulled; no weights, output archives, or notebooks executed.

| Source | Current / best public | Actual code finding | Disposition |
| --- | --- | --- | --- |
| amanatar/optimized-biohub-max-score | absent /0.953 | All code exactly matches the previously inspected V6 source | No new0.965+ evidence; C039 already tested isolated fusion and repaired division negatively. Do not rerun unchanged arms. |
| mtoshidesu/testbiohub-lf-dctta020-sectta1 |0.947 /0.947 | Relative to the reviewed evgendvorkin0.947 source, after removing comments/docstrings (including parseable embedded Python), only DeepCenter division threshold0.25->0.20 and secondary edge-feature TTA weight0.75->1.0 remain | Existing pipeline/knob changes; no new mechanism or compelling improvement evidence. Source and normalized diff retained; no experiment promoted from this copy. |
| sarveshchhetri/robust-3d-cell-tracking |0.429 /0.432 | Classical detector normalization, stricter child-count handling, track-length changes and local validation additions | Changes are real but baseline is weak and these mechanisms overlap existing inference/constraints; no component evidence to prioritize. |
| denpugovkin/biohub-exp003-two-embryos |0.885 /0.885 | Single-seed detector/ILP, strong/top-k edge shortlist, image-centroid gap fill, temporal-boundary exemption in short-track pruning | Reviewed as a distinct source; mechanisms already catalogued in prior auditD. No new training or independent improvement evidence. No blanket inference from author/title. |

Evidence: state/notebook_radar/pulled/review_20260927_continuation/{before,after,changes,pull_results,source_comparisons}.json and source/hash/diffs in per-author folders. Scores are radar readbacks; best scores must not be attributed to latest unscored versions. Source comments and titles are claims.

Proceed with user-authorized C040 late600+C038 actual22 replay while preserving anchors. Future public refreshes should be bounded and spaced (at least6h unless user supplies a specific new lead), inspect changed sources only, and promote a distinct plausible mechanism through the existing controls/harness. Do not blindly repeat parameter families or poll the LB.
