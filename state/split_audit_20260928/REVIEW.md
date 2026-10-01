# Split and historical-result audit — 2026-09-28

The user requested rechecking the199/24 split and repeating work whose conclusions depend on an invalid split.

## Confirmed issue

Movie filenames do not identify independent biological acquisitions.199 movies belong to two embryo groups:44b6 has71,6bba has128. The original24 were a compute-limited training subset outside97 cached evaluation movie IDs, not a defensible train/validation/test ratio. The remaining78 were not an independent test set.

Raw evidence is saved in node_overlap.json, pixel_overlap.json and prior24_pixel_overlap.json. Three original24 training/evaluation pairs show byte-identical aligned image blocks at two times each (4608 voxels per block,100% equality,zero maximum error):

- 44b6_7e557709 vs44b6_12dfb391 (time offset15; spatial34,-227,152)
- 44b6_3a861e03 vs44b6_c8e2a523 (time offset36; spatial-16,-177,-177)
- 44b6_e57ff5c6 vs44b6_5f15d135 (time offset65; spatial-50,47,-67)

These establish source-image overlap, not merely coincident GT identifiers.6bba IDs appear to be reused locally; shared IDs alone do NOT establish pixel overlap there. No need to assume they do: complete embryo separation prevents within-embryo crop overlap for new fitted components in either group. The pixel audit is a targeted proof of a counterexample, not an exhaustive reconstruction of all crop origins.

## What remains valid and what changes

| Work | Retained evidence | Corrected interpretation / next step |
|---|---|---|
| C023/C024 and scored submissions | Actual Kaggle public scores, exact source versions and execution checks | Preserve anchors. Scores0.954 are real, but repeated public selection is not private-test proof. |
| All exact replay/T4 controls | Executed graphs, CSV identity, source/model hashes, fallback checks | Execution equivalence is valid regardless of split. It never established generalization. |
| C012–C016 and later coordinate-head CV | Raw fit/error tables and actual scored submissions | Existing v1284_head_train and division_scorer_train interleave movie IDs WITHIN embryos; these are not independent source-held-out CV. Do not rank heads by those CV claims. Reevaluate a materially promising unsubmitted learned mechanism with embryo folds before reviving it; do not repeat failed LB settings. |
| C031 division CNN | Raw training/diagnostic evidence | Audit its split recipe and effective division identities before a new embryo-held-out attempt. Do not infer all division learning impossible from this diagnostic. |
| C034 | Insufficient-known-label finding; no fitted model | Remains valid. |
| C035 | Actual cross-embryo diagnostics and unknown-label handling | Evaluation uses the OPPOSITE training embryo, so newly learned appearance weights do not cross the problematic same-embryo crop split. Keep results; limitation is small/far-negative training and only two biological groups. |
| C036 | Registration coverage/fixes, no learned stage | Raw diagnostic preserved; not an independent final-test result. |
| C037, C038 and C040 diagnostic-fold combinations | Opposite-embryo training/checkpoint routing during local research | Newly fitted components are embryo-disjoint. Keep signed metrics and failures. Routing is an evaluation device only, never a deployment router. Frozen base detectors/heads saw both embryos, so whole-pipeline independence is still unavailable. |
| C041–C046 fixed pooled models | Actual97/22 graph effects and portable/T4 equivalence | The evaluated fixed model includes weights trained on the evaluated embryo;24-vs97 ID exclusion does not prevent source overlap. Local gains are retrospective fit-domain results, not independent evidence. C042/C043 already scored0.954; keep facts without attributing gain. Hold unsubmitted C046 for scientific review. |
| C047 original |29 completed data preparation jobs; no trained model | Stopped manager53168 and its verified tree before training. Preserve original source/plan/status snapshot and crop evidence; never restart this mixed-model experiment as clean validation. |
| C032/C033/C039 and fixed postprocessing negatives | Exact graph deltas and actual scored negatives where available | Repeated local tuning and pretrained detector exposure were already limitations. Split discovery does not justify rerunning unchanged closed arms. Reopen only a distinct supported mechanism. |

## Corrected protocol and immediate redo

Use two whole-embryo outer folds. Fit on44b6 only, evaluate6bba only; fit on6bba only,evaluate44b6 only. Never blend both folds' models when reporting outer-fold validation. Prove train_movies and evaluated embryo disjoint from saved checkpoints before every replay. Neither frame-level nor movie-ID random splitting is acceptable for claims of source independence.

C048 replaces C047. Audit all199 for real GT supervision;190 have eligible appearance triplets (62/128),9 do not. Use all eligible training movies belonging to the training embryo, fixed192 maximum triplets/movie with up to96 close6–12um examples and fill from the remaining known triplets. Two fixed1200-step C035 weak_aug fits; no checkpoint,learning-rate or threshold sweep. Original24-vs-new-data cross-embryo graph comparison uses C038 preserved results, not pooled C046 as the validity reference. Execute existing cached97 movies (27/70) with the opposite-embryo model and same-base C023 off controls. This separates new-model fitting from evaluation while reusing existing inference/scorer. The full71/128 are the partition, not a claim every movie has cached full-pipeline evaluation.

After evidence supports a recipe, a separate production model may use all available eligible training movies. Its local replay/T4 scores are technical checks, not held-out validation. A final independent whole-pipeline local test is impossible with these two embryos and pretrained public detectors; do not invent a third test group. Report both outer folds and biological sample count2, not199 independent replicates.

Division training is no longer categorically data-blocked by the old outside97 rule. The census shows26/125 division annotations by embryo; a future opposite-embryo fit could use the training embryo's annotations inside the old97 without using its evaluation embryo. First audit duplicate biological events and class/feature coverage. This is a materially changed split, not permission to fit and score the same embryo or relabel unknown cells as negatives.

User authorization to pursue0.955 and submit worthwhile verified candidates continues. Preserve alarm schedule, one local GPU queue, original evidence, shared quota and no LB polling. C046 can finish remote technical checks; do not automatically submit it based on invalidated local generalization claims. Prior C045 was accepted56622952; no repeat.
