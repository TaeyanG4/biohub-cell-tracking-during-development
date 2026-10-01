# C058 model and target-distribution diagnosis

The saved model fits its source training labels substantially better than it transfers to the opposite embryo. Opposite-embryo outputs contain large systematic directional errors, especially in z. Integer rounding and float16 crop storage do not explain the effect. This is evidence for a generalization/target-distribution mismatch, not proof that a particular cause—overfitting, annotation convention, image domain or incorrect identities—is solely responsible.

Scope: saved original labels, final C023 float/integer coordinates and C058 opposite-embryo predictions; a deterministic source-training-crop diagnostic using the existing final checkpoints. No retraining, checkpoint selection, new scorer, graph replay, threshold search or correction-scale search. Original artifacts remain untouched. Reproduction and full statistics are in `diagnose.py` and `diagnosis.json`.

## Target and correction bias

All vectors below are z/y/x in micrometres. These are actual inference-eligible known matches (1,102 points from 44b6 and 6,725 from 6bba), not all predicted nodes.

| Evaluation embryo | Mean actual target correction | Mean predicted correction from opposite model | Target/prediction correlation by axis |
|---|---|---|---|
| 44b6 | (+0.016, +0.034, −0.013) | (+0.992, −0.011, +0.491) | (0.127, 0.346, 0.312) |
| 6bba | (+0.508, +0.193, +0.405) | (+0.150, −0.027, −0.037) | (0.153, 0.144, 0.208) |

The tail is particularly diagnostic. For 93 points from 44b6 with original residual >3.5µm, the target mean z shift is −1.031µm, but the model trained on 6bba outputs +1.054µm. Its z correlation is −0.024. For 540 tail points from 6bba, the target mean z shift is +1.441µm, but the model trained on 44b6 outputs −0.197µm; z correlation is 0.172. Thus large-error points are not merely receiving too little of the correct mean displacement; the mean z directions are opposite in both tail cohorts. This is an observation, not a deployment gate or a proposed GT-conditioned offset.

The source-label distributions themselves differ:

| Source embryo | Point-weighted target mean | Uniform-movie target mean (matches registered sampling) | Eligible source labels |
|---|---|---|---:|
| 44b6 | (+0.028, +0.027, −0.005) | (+0.302, −0.139, +0.067) | 1,052 |
| 6bba | (+0.501, +0.184, +0.423) | (+0.540, +0.208, +0.428) | 5,965 |

Uniform-movie sampling changes the44b6 target distribution appreciably because some movies contain few labels. Still, the44b6 evaluation output z mean+0.992 exceeds the6bba source sampler mean+0.540, so a simple transfer of the training mean is not a complete explanation. The model's image-dependent behavior matters. Centered residual variation remains substantial: total bias-squared is about1.208/6.000µm² of44b6 residual MSE and0.372/5.966µm² of6bba residual MSE. Removing a constant mean cannot be assumed to restore localization. No such correction was tested or selected.

## Source fitting versus transfer

A fixed source-only diagnostic selected at most128 evenly spaced stored crop rows per movie, independently of labels. The registered model for that source embryo was run without jitter on those stored crops. The same rows were compared with the previously saved actual opposite-model predictions. Both are final1200-step checkpoints, with no fitting or checkpoint choice.

| Source crops | Points | Original mean error | Own-source model error | Opposite-embryo model error |
|---|---:|---:|---:|---:|
| 44b6 | 581 | 1.712µm | 0.726µm | 1.986µm |
| 6bba | 1,997 | 1.955µm | 1.302µm | 2.229µm |

Source tail subsets show the same gap:44b6's47 sampled tail labels go4.767→1.479µm with the own-source model but4.747µm with the opposite model;6bba's211 go4.545→2.863µm own-source but4.652µm opposite. Own-source axis correlations are approximately0.90/0.92/0.90 for44b6 and0.71/0.63/0.62 for6bba, compared with weak correlations on the opposite domain.

These are **training examples**, often temporally/spatially correlated. The improvement demonstrates that the architecture and pipeline can represent substantial source-label information; it does not prove that it learned the correct biological cell center, nor establish same-embryo generalization. Strong fitting can reflect memorization of biased or misassigned targets. The much smaller source domain has the stronger fit, which is compatible with overfitting but does not establish it as the only cause.

## Rounding and stored-crop precision

On all eligible known points, compare the frozen C023 float center with its actual integer emitted center, using the **same original matched GT identity**:

| Embryo | Original float-center error | Integer-center error | C058 error before output rounding | C058 error after output rounding |
|---|---:|---:|---:|---:|
| 44b6 | 1.578µm | 1.608µm | 2.034µm | 2.061µm |
| 6bba | 1.763µm | 1.800µm | 2.103µm | 2.140µm |

The float→integer mean displacement magnitude is0.449/0.464µm, mainly z quantization, but it changes mean target error by only0.030/0.037µm. The mean signed rounding displacement is approximately zero on each axis. The model already worsens error by0.426/0.303µm **before** final output rounding. Therefore rounding is a small additional effect, not the main failure mechanism. These residual comparisons are not official unrounded-output scores; the actual submission writer requires integer coordinates.

To isolate the stored-float16 versus raw-float32 input difference, the opposite model was also run on the exact same sampled stored source crops. Against its saved fullprecision raw-input predictions, mean prediction-vector difference is0.000358µm for44b6 and0.000441µm for6bba; maximum absolute axis difference is0.001285/0.002052µm. Mean error changes only−0.000005/+0.000020µm. This checks the specific sampled inputs and excludes crop quantization as a plausible cause of the large measured transfer gap; it is not a universal precision-equivalence proof.

## Causal limitation of the fixed-graph pilot

C058 freezes final edges and node IDs. If coordinate correction leaves the GT matching unchanged, lowering residual alone cannot improve edge TP; official edge-score gains require recovered coverage or changed one-to-one matching. The earlier75.6% association between bad localization and missed links concerned upstream association decisions. Even a successful standalone localizer would need a separately controlled integration **before association** to test whether better coordinates can produce better links. C058 still fails the simpler necessary signal—opposite-embryo fixed-pair localization—so integrating this model upstream is not justified.

The next discriminating evidence should establish whether the annotated target corresponds to the image-localized cell center and whether a geometry-preserving localizer transfers that evidence, especially along z. This diagnostic does not justify embryo-specific deployment offsets, GT-derived runtime gates, or another sweep of the same regression model.
