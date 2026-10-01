# Independent C060 prelaunch review

Status at this review: **model/data implementation is suitable for the bounded benchmark and a fixed pilot after the remaining gates below are satisfied.** No localization gain is established. Root owns the actual full-movie benchmark, original-writer verification, immutable plan registration and launch.

## Scope and distinct scientific question

This is distinct from closed C058: 22,432 parameters instead of 407,659, a valid spatial location map instead of a flattened vector regressor, actual three-node predicted-track images, stable known source-lineage supervision, exact reflection-group averaging, orientation agreement, and strict all-prediction spatial ownership. It tests the recipe jointly; a positive result cannot uniquely attribute the gain to one change.

The available source supervision is 1,021 / 6,255 points from the two embryos, including 58 / 371 residuals >3.5µm. Those counts retain a real localization tail. They are correlated observations, not independent biological events. Requiring both known GT context edges excludes demonstrated identity contradictions from training, but creates a domain difference from runtime predicted triples. This initially tests persistent localization on stable paths, not direct supervision for repairing the known identity-switch cases.

## Model and geometry review

Two valid 3×7×7 convolutions map 13×49×49 crops to 9×37×37 logits. Per-voxel channel LayerNorm preserves common-valid-region translation behavior of the learned features. There is no flattening, padding, absolute-coordinate input, or embryo-specific parameter routing at runtime. The 5,381 supported lattice sites cover the discrete physical 7µm sphere.

The fixed symmetric prior sigma 7/3µm gives a unique zero mode when the final head is initialized to zero. This is a conservative geometry choice, not source- or target-calibrated confidence. It can suppress useful large corrections and must remain unchanged after target results. The Gaussian point-target loss is conditional on a supplied trajectory, not a detector cell/background loss. Unknown cells receive no false-cell/background labels, although incorrect operational identities can still produce harmful conditional competition.

The odd grid permits exact target sign inversion under z/y/x reflection. Each temporal image receives the same reflection and real-support center jitter; targets subtract the physical jitter. Every axis has its original physical spacing. No z↔x/y rotations are claimed. The eight inverse-reflected **logit** maps are averaged. A spatial mode is used only if all eight have the same unique mode; ties/disagreement return exact zero. Neither posterior entropy nor mode probability is treated as calibrated biological confidence.

**Qualification:** the fixed center prior and finite radius deliberately depend on the crop anchor. The complete posterior is not claimed translation-equivariant under changing that anchor. Only learned valid convolution features satisfy the tested common-support property. Reflection equivariance of the full group-averaged posterior is tested up to floating-point arithmetic.

## Executed independent controls

`c060_independent_controls.py/json` exercised the actual modules without fitting or saving a model:

- Two exact zero initial proposals and exact zero symmetric posterior means.
- Finite loss and nonzero initial head gradient.
- All eight transformed Gaussian targets and nontrivial random-weight group-averaged maps; maximum supported-logit reflection difference 9.54e−7.
- Valid-feature common-support translation; maximum difference 2.98e−7.
- 32 joint real-slice jitter/reflection examples with exact pixel/target transforms.
- Flat-map, tied/disagreeing orientation abstention.
- Synthetic graph topology: missing context, synthetic context and a branching predecessor are excluded.
- **Actual production `infer_one` path**, with mocked images/proposals and a synthetic/ineligible same-frame competitor: crossing its bisector is rejected, an exact tie is rejected, and a point strictly inside its own region is accepted. All excluded coordinates remain exact. Thus the runtime tree includes synthetic and ineligible competitors, as required; it is not a nearest-neighbor-radius approximation.

The model agent independently used two real predicted triples and original image reads for the controls in `state/c060_preflight_20260929/model_controls.json`, including feature gradients after one disposable update and the rejected-jitter path. The data agent's `context/module_controls.json` matches all 22 inference/expanded masks and source label selections against the retained audit, leaves graphs unchanged, and checks 12 saved channels against fresh real crops. `context/gt_io_controls.json` verifies all 22 GT edge arrays (16,551 edges) against the original loader. The final target-key contract (`targets`) is checked by root's actual benchmark after a preflight-only API rename.

Static review confirms that `context_indices` reads only predicted graph topology, provenance and image bounds. The source-only `source_labels` function reads GT, and runtime `infer_one` does not call it or load GT labels. Target-embryo inference loads the opposite source checkpoint and checks source metadata. The provisional frozen C023 graph supplies track context; this is not yet a hidden-movie deployment recipe.

## One driver gate issue and remaining launch evidence

The inspected `analyse()` initially defines the good-point identity group as **eligible** nodes with original residual ≤2.5µm. Extend its lost/remapped-identity gate to **all originally good matched nodes**, retaining the eligible subgroup for reporting. C058 demonstrated that nine excluded nodes could lose matches without moving, due to one-to-one competition. The correction may pass mean-residual checks while indirectly harming an excluded identity. Root has been notified; this should be corrected before plan pinning. All-node mean and eligible-only mean have the same sign because excluded coordinates stay fixed, but identity transitions do not.

Remaining root evidence is the two complete zero-model movies, actual writer/official parity, measured runtime, final source/recipe hashes, and the final gate adjustment. The registered queue must execute controls on all 22 movies and validate output arrays, original matching, crop provenance, folds and hashes before scientific promotion. The finite benchmark does not replace these queue checks.

## Decision rule

For each opposite embryo, require improvement in **all eligible original-pair mean residual, eligible >3.5µm tail mean, and persistent-tail mean**, including abstained cases. Require no increase in original good-point mean error and no lost/remapped identities among **all** original ≤2.5µm matches. Require nonnegative actual official adjusted-edge and total changes in each embryo. Report changed/abstained/ownership-rejected counts and all-node identity effects.

Do not demand an arbitrary fixed score increment, minimum recovered-edge count, or frozen-edge TP improvement. When final edges and GT matching remain fixed, a smaller residual cannot change edge TP; transferable coordinate evidence may instead justify a separately controlled pre-association experiment. A tie caused by no accepted corrections is not positive localization evidence. A failed fixed recipe does not justify a threshold, prior-width or training-length sweep.

Passing supports manual review of an unchanged extension or an exact-control upstream integration test, not automatic T4 promotion or submission. Existing C023/C024 anchors remain untouched.
