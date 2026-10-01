# Final local review: C032/C033 — no submission

Queue completed 2026-09-27 02:06:43 KST, all 29 jobs successful. Reviewed 2026-09-27 03:06 KST. No new Kaggle push, T4 job or submission is warranted by this batch.

## Official aggregate results

Existing C023 aggregation functions and eval_pp_variants_local.summarize were reused on cached per-movie metric rows. All nine original split summaries were reproduced within 1e-12 before combining the 97 unique movies. Group scores were not averaged arithmetically. See FINAL_REVIEW.json for input hashes and diagnostics.

| Arm | All97 score | Delta | Edge delta | Extension75 delta | Division TP/FP/FN all97 |
|---|---:|---:|---:|---:|---|
| control_fp32 | 0.940490013 | +0.000000000 | +0.000000000 | +0.000000000 | 27/46/124 |
| mean_det | 0.940095180 | -0.000394833 | -0.000830663 | -0.001140978 | 28/47/123 |
| future_det | 0.939671800 | -0.000818213 | -0.000905732 | -0.001609870 | 28/52/123 |

| Arm / group | Total delta | Adjusted-edge delta |
|---|---:|---:|
| mean_det / 44b6 | -0.004954752 | -0.003790673 |
| mean_det / 6bba | +0.000322570 | -0.000493280 |
| mean_det / extension44b6 | -0.006485234 | -0.005203183 |
| mean_det / extension6bba | -0.000282069 | -0.001246623 |
| future_det / 44b6 | -0.007206988 | -0.006412887 |
| future_det / 6bba | +0.000025701 | -0.000278787 |
| future_det / extension44b6 | -0.010845290 | -0.009968097 |
| future_det / extension6bba | -0.000380241 | -0.000948455 |

## Decision and limits

Both extended temporal candidates are rejected for this batch. Their positive 22-movie pilots do not survive extension75. On all97, both lose adjusted-edge score in each embryo group. Small positive 6bba total deltas are division-driven and do not support improved association. All local embryos were training embryos for public detectors, so these results do not prove a hidden-test decrease, but provide insufficient grounds to spend a submission slot. No post-hoc embryo selector, threshold sweep or weakened acceptance rule is introduced.

Mean_det: all97 edge wins/ties/losses 42/1/54; extension75 30/1/44. All97 +5786 final nodes, +1 edge TP, +57 edge FP; the increase in detections does not translate into a better adjusted-edge score. Worst movie adjusted-edge delta -0.047254 (44b6_c50204e0).

Future_det: all97 edge wins/ties/losses 47/0/50; extension75 33/0/42. All97 -1440 final nodes, -26 edge TP, +47 edge FP. Worst movie adjusted-edge delta -0.061629 (44b6_0b24845f).

Mean_det_head was already rejected at the pilot stage (+0.002310 heldout12, -0.000759 confirm10); both C033 fixed structured modes were negative on both pilot sets (-0.001057 / -0.000313). These three arms were not extended.

The user authorized judgment, pushes, T4 verification and submissions, including all useful remaining daily slots. No candidate here qualifies, so no quota/network operation is needed. C023/C024 scored artifacts and final-pick settings remain unchanged. The batch follow-up automation is being disabled; no continuing wait/poll loop is needed.

The local queue remains historical evidence. Its generated RESULTS.md language about user-run T4 checks describes the original queue scope; later user authorization is recorded separately in AUTO_SUBMISSION_HANDOFF.md and auto_submission_state.json.
