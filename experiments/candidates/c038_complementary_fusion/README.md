# C038: small-effect appearance + image-motion combination

Registered after explicit user instruction to examine combinations before discarding small individual effects (2026-09-27). Historical C035/C036 raw results remain intact. This is a separate, transparent follow-up rather than retuning their original gates.

## Diagnostic already available

Recompute the unchanged C035 weak-augmentation and C036 proposal rules. Their only44b6 recovery is the same source/target; simple union adds no new recovery and retains C035's6bba harm. Requiring a trusted C036 motion estimate to name the same target as a C035 proposal leaves one diagnostic fix and zero harms, all in44b6. `overlap_diagnostic.json` records this honestly. **One fix is small but is not an automatic reason to stop:** execute a GT-free graph replay to learn whether it survives the full pipeline and what it does on unannotated predictions.

## Actual graph experiment

- `src/c038_complementary_stage.py` reuses C034's candidate recorder with an overridden GT-free begin method, C035 crop/CNN implementation and C036 local registration. Capture all predicted source candidates, never a GT-selected source whitelist. Only production gates determine eligible sources.
- After the entire C023 filter (including ILP edge restore), compare original current edges against the candidate appearance signal. Preserve nodes/coordinates and protect division/gap incident nodes. Use the same cosine thresholds as C035; choose between `appearance` alone and `agreement` with trusted local-image nearest-target agreement. The existing CNN and registration code are reused without fitting; no label enters decisions.
- Apply changes only to unused targets or reciprocal two-source swaps. Preserve source degrees and edge count, forbid target collisions/duplicates and unilateral edge stealing. Record proposed versus actually changed counts. If C023's later stages already repair the diagnostic mistake, report that suppression instead of treating a per-source win as a graph gain.
- Cross-embryo CNN models for local evaluation; no hidden-embryo routing. CPU FP32 embeddings are numerically checked against the existing GPU diagnostic. A deployment model/ensemble and portable notebook still need separate actual replay/T4 validation.
- Local replay notebooks are C023 copies with explicit local adapter imports. **They are NOT Kaggle submission notebooks and must not be pushed.** Execute with the existing `eval_pp_variants_local.py`, variants off/appearance/agreement, matching C023 caches, integer rounding and official aggregation. All22 off-controls must reproduce stored C023 metrics/counts.

## Scheduling and judgment

`python src/c038_complementary_study.py prepare` builds local replay notebooks, saves overlap diagnostic and CPU parity. `run` refuses while C037 status is running, then uses the existing queue for official12+10 replay; eight-hour bound. This prevents GPU contention with the user's4070 Ti SUPER. The scheduled follow-up starts this prepared experiment after C037 completes, without asking again.

Do not impose the old five-net-fix threshold on this combination. Review actual signed official deltas, changed-edge/degree counts, losses, and embryo consistency. Small supported gains can advance to extension75 and fixed combinations with the C037 correction. Lack of extra recovery in a simple union is not proof every combination fails. Conversely, adding many weak variants does not itself establish benefit; avoid arbitrary weight sweeps on the same22 movies. No submission based on diagnostic rankings alone.

State/logs/results in this folder, source lock/hash, long compute in background, one compact heartbeat check while running. C023/C024/final picks unchanged. No direct Kaggle operation by these scripts. Any eventual candidate must have real deployment replay, matching T4 visible4, recorded version/hash and quota/duplicate checks.
