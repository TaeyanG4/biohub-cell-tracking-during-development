# C035 final review: no graph integration or submission

Reviewed 2026-09-27 08:14 KST. Execution finished at 07:42:02 KST, about 9.5 minutes after launch. All 52 phases succeeded: one real-data smoke, 24 training-crop extractions, 22 evaluation-crop extractions, four model fits and one cross-embryo scoring phase. Source hashes and all 123 recorded artifacts verified unchanged; lock released and stderr empty. All 22 unchanged C023 controls and exact C034 candidate comparisons passed.

Training used 24 movies disjoint from the official97 movie IDs: 2,163 triplets / 3,342 patches from 44b6 and 2,304 triplets / 4,506 patches from 6bba. Unknown candidate detections were never used as training negatives. Each of the four models completed its fixed 1,200 steps, with no target-embryo checkpoint or threshold selection.

| Method | Test embryo | Retrieved matched GT child | C023 correct | Conservative fixes / harms | Net |
|---|---|---:|---:|---:|---:|
| Basic CNN | 44b6 | 406 / 459 | 439 | 1 / 0 | +1 |
| Augmented CNN | 44b6 | 414 / 459 | 439 | 1 / 0 | +1 |
| Unaligned patch NCC | 44b6 | 432 / 459 | 439 | 1 / 0 | +1 |
| Basic CNN | 6bba | 1,613 / 1,843 | 1,737 | 1 / 8 | -7 |
| Augmented CNN | 6bba | 1,625 / 1,843 | 1,737 | 5 / 8 | -3 |
| Unaligned patch NCC | 6bba | 1,691 / 1,843 | 1,737 | 2 / 7 | -5 |

These counts are conditional, per-source retrieval diagnostics, not edited graph edges or official metric deltas. A harm means losing retrieval of the matched GT child; unknown detections do not thereby become verified biological negatives. Assignment competition and later postprocessing have not been executed with these proposed changes.

Augmentation improved raw retrieval by eight and twelve groups relative to the basic CNN and reduced conservative losses in 6bba, but both learned arms failed the fixed compute gate in both directions. For the augmented arm, ranking net versus C023 cost argmin was -26 / -112 and conservative net was +1 / -3; the gate required both to reach +5 in each direction, with positive conservative net in at least two movies per direction. The 44b6 gain was confined to one movie. No post-hoc threshold sweep is warranted by this fixed study.

Training negatives are mostly separated nuclei: median child-to-negative separation 23.75 / 26.68 um, with only 150 / 69 triplets at <=12 um. Thus acquiring more defensible labels solved the C034 label-count blocker but did not guarantee hard-negative supervision representative of ambiguous detector candidates. This and the lack of a third embryo limit interpretation; the experiment does not disprove all appearance learning or augmentation. Single fixed-seed comparisons do not establish a robust augmentation effect.

**Decision: close C035 as unsupported for graph integration.** No modified-graph official score, extension75 appearance replay, notebook, T4 verification, push or submission was produced. Unaligned NCC is also unsupported under this rule; this is not a test of motion-aware image registration. Preserve crops/models for a separately justified future study. C034's insufficient-label conclusion, closed C032/C033 decisions, C023/C024 anchors and user final picks remain unchanged. No shared submission slots consumed. Disable the C035 follow-up after this review.
