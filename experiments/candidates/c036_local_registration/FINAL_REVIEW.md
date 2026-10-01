# C036 final review: fixed local-registration study closed

Reviewed 2026-09-27 after the user's status request. All22 movies completed without error at **08:26:46 KST**, about4.5 minutes after launch. Translation/photometric/subvoxel smoke passed. Source hashes and all66 recorded per-movie artifacts still match, the execution lock is released and stderr is empty. The run verified inherited C035 original-coordinate caches and their exact C023/C034 control provenance, and recorded newly read image-chunk hashes.

| Test embryo | Annotated reachable source groups | Trusted image estimates | Proposals | Fixes / harms | Net | Positive movies |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 | 459 | 328 | 1 | 1 / 0 | +1 | 1 |
| 6bba | 1,843 | 630 | 0 | 0 / 0 | 0 | 0 |

Both embryo directions fail the predeclared compute gate (net>=5 and at least2 positive movies per embryo, plus precision/ranking requirements). The one proposal retrieves the matched GT child in `44b6_12dfb391`; it is not yet a graph edit or an official-score gain. Local ranking on trusted groups has net0/+1 against C023 cost argmin. These small effects do not justify integration or a submission.

Coverage is a material limitation. Image-boundary checks reject **1,218/2,302 groups (52.9%)**: 111/459 in44b6 and1,107/1,843 in6bba. The fixed contextual patch and search region must both lie within the recorded volume. Overall958 groups (41.6%) are trusted; further rejections include80 ambiguous peaks. No padding or retrospective confidence/size tuning was introduced to increase coverage.

The local estimator did execute: trusted residual motion beyond the coarse phase shift has median0.88/0.98um and95th percentile2.01/2.87um. Local-versus-coarse correctness changes occur in2/10 groups and cancel in aggregate; identical aggregate retrieval counts do not mean the local step was skipped. This remains a translation-only, fixed-window experiment and does not disprove all image registration or optical-flow methods.

Unknown targets retain their original unknown labels. A retrieval miss is not a newly verified biological negative. This conditional diagnostic only covers sources with an annotated reachable successor and ignores global assignment competition and later graph stages; no modified-graph official metric was computed.

**Decision: close C036 as unsupported for graph integration.** No extension75 registration replay, submission notebook, T4 job, Kaggle push or competition submission; no shared slots consumed. Retain raw diagnostics/caches, preserve C023/C024 and user final picks, and disable this study's finite follow-up. Earlier C032-C035 decisions remain unchanged.
