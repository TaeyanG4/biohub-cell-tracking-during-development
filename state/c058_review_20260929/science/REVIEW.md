# Independent scientific review of completed C058

Recommendation: close this fixed recipe without a 97-movie extension, portable integration, T4 run, or submission. The pre-registered gate fails on both biological embryos, and degradation is not confined to one anomalous movie. This result rejects this fitted coordinate regressor; it does not establish that all additional spatial image information is unhelpful.

This review reads the immutable recorded outputs and statically inspects the study/model implementation. `recount.py` only aggregates saved labels, coordinates, transitions and official results; it does not fit, infer, search parameters, or implement a new metric. Root review owns independent input/output hashing and final state updates.

## Official result

The original CSV writer and actual vendored official evaluator report 0.9431126093954155 for frozen C023 and 0.9369254828667917 for C058, a **−0.006187126528623854** local change. The number of nodes is exactly 517,966 in both arms. Edge TP is 15,940→15,893, FP 648→712, and FN 611→658. Division TP/FP/FN remain 4/5/20. Thus the failure is an edge/coordinate identity effect, without any demonstrated division gain or node-count benefit.

| Evaluation group | Total/adjusted-edge change | Edge TP change | Movie wins/losses/ties |
|---|---:|---:|---:|
| all22 | −0.006187127 | −47 | 1 / 12 / 9 |
| 44b6, trained on 6bba | −0.026944712 | −22 | 0 / 3 / 3 |
| 6bba, trained on 44b6 | −0.003931318 | −25 | 1 / 9 / 6 |
| heldout12 | −0.003963005 | −18 | 1 / 6 / 5 |
| confirm10 | −0.008160992 | −29 | 0 / 6 / 4 |

The sole movie gain is 6bba_09961292 (+0.010737329). The largest losses are 44b6_12dfb391 (−0.043645774) and 6bba_3db54e20 (−0.041665116). Selectively retaining the winning movie or assigning deployment behavior by embryo would be evaluation selection, not a supported deployment rule.

## Training evidence and coverage

| Source embryo → evaluation embryo | Source movies | Eligible known training examples | Training residuals >3.5µm | Steps / presented samples |
|---|---:|---:|---:|---:|
| 44b6 → 6bba | 6 | 1,052 | 87 | 1,200 / 38,400 |
| 6bba → 44b6 | 16 | 5,965 | 486 | 1,200 / 38,400 |

These are crop/point rows, not independent biological events. Source crop overlap within an embryo and temporal correlation remain. Each model is evaluated only on the opposite embryo, and only the final registered checkpoint is used. The source checkpoint metadata records matching source stems, source embryo, final step count and sample totals. Recorded fitting time is approximately19 seconds per model; the rest of the queue performs image capture, extraction, inference, writer evaluation and validation. The final logged loss is only one sampled source batch, not a held-out fit-quality estimate or proof of generalization.

Across22 movies, 16,931 baseline predicted nodes have official matches. 7,017 of these supply real expanded training crops; 573 are in the >3.5µm tail. Inference support is available for 243,992/517,966 output nodes (47.11%); 270,192 nodes lack full spatial support and 3,782 are synthetic. 239,751 eligible nodes change integer output coordinates. Among known matched nodes, 7,827 are inference-eligible; the difference from7,017 training-eligible rows is the intentionally larger real-support margin needed for jittered training crops.

The target eligibility and known-only labels are consistent with the recipe. Unknown cells are not used as background negatives. Ten training-eligible labels have more than one known GT point within7µm; all assigned labels are nearest known GT, and the alternate notebook matcher agrees with the actual official matcher on the recorded labels. That does not prove biological identity: unannotated competitors cannot be ruled out.

## Retained identities and fixed-pair localization

All residuals below keep the original baseline node→GT identity, including nodes that later become unmatched or remapped. They are not recalculated against whichever GT becomes closest after correction.

| Original residual group, inference-eligible only | Count | Mean before→after (µm) | Improved/worsened/unchanged | Original identities lost/remapped |
|---|---:|---:|---:|---:|
| ≤2.5µm | 6,462 | 1.336000→1.771697 | 1,900 / 4,196 / 366 | 9 / 0 |
| >2.5 and ≤3.5µm | 732 | 3.144881→3.050871 | 361 / 336 / 35 | 15 / 0 |
| >3.5µm | 633 | 4.642254→4.711066 | 273 / 348 / 12 | 95 / 1 |
| All eligible known matches | 7,827 | 1.772561→2.129046 | 2,534 / 4,880 / 413 | 119 / 1 |

The target tail fails on each opposite embryo: 44b6's93 eligible tail pairs go4.772527→5.055348µm; 6bba's540 go4.619818→4.651773µm. Tail median also increases on each embryo. Across both,49 previously within7µm fixed pairs move beyond7µm. Already accurately localized points deteriorate substantially, so the outcome is not merely failure to recover rare difficult examples. The modest improvement in the middle residual band cannot rescue the registered hypothesis.

Across all16,931 original matched predicted nodes,16,802 retain the same match,128 become unmatched and1 is remapped. The after-matching graph includes114 formerly unmatched predicted nodes with a match:22 in44b6 and92 in6bba. **Those114 are not necessarily new GT recoveries**: a moved previously unmatched prediction can take an already represented GT from another prediction. Match-transition files alone do not identify which GT IDs are genuinely newly recovered. The net matched-prediction count is−14. Nine excluded, unchanged-coordinate predictions also lose their original matches, demonstrating that global one-to-one matching competition can alter identity even for a node that did not move.

## Interpretation and limits

Static review of the coordinate order, physical-unit targets, integer jitter target translation, zero-model initialization, matching recipe and opposite-embryo load paths did not reveal an obvious sign/axis/fold error. Training uses float16 stored crops converted to float32; inference reads normalized float32 images, as the registered README explicitly records. That small input-precision difference is not proven to explain degradation and does not justify post-result recipe tuning.

A 407,659-parameter model with1,052 or5,965 correlated point examples and sparse known labels can have substantial generalization risk. Official matches within7µm are operational labels rather than verified biological identities, and unmatched >7µm cells supply no coordinate targets. Thus the earlier error association between poor localization and lost links did not establish that this learner could recover those errors. Training loss alone cannot distinguish overfitting, target ambiguity and embryo-domain mismatch here. No particular causal mechanism is proven by the present outputs.

The public frozen detector/head already saw both embryos. This is opposite-embryo validation of the added component, not fully independent pipeline validation and not a leaderboard prediction. No submission-slot usage is justified by these data. Keep the C023/C024 anchors and this negative result intact, and require a distinct supported hypothesis for future work rather than fitting longer or sweeping this fixed arm.
