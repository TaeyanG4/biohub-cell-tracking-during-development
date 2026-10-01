# C051 completed: real boundary support increased, no added recovery

All 25 jobs completed at 2026-09-28 13:01:11 KST, about six minutes after launch. Independent review verified 1,161 input hashes, 92 output hashes and 996 actual image chunks. All 2,302 original C036 reason/trust/predicted-point controls reproduced across 22 movies; coarse distances and all 1,084 originally nonboundary groups' pair distances remained exact. Numerical translation/photometric, interior parity, flat-patch and insufficient-support checks passed.

The fixed recentering method actually increased trusted support, using only real full-size templates and search regions. Of 2,572 support attempts, 1,271 met the registered real-support conditions; confidence, multiscale and cycle checks further reduced accepted groups. No padding, template resizing, fitting or threshold selection occurred.

| Embryo | Originally rejected boundary groups | Newly trusted groups | New proposals | Recovered annotated matches | Lost annotated matches | Unknown proposed targets |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 | 111 | 71 | 0 | 0 | 0 | 0 |
| 6bba | 1,107 | 413 | 1 | 0 | 1 | 1 |

Trusted coverage increased from 958/2,302 (41.6%) to 1,442/2,302 (62.6%). The 484 newly trusted groups produced no additional annotated recovery. The one new proposal replaces an annotated-correct C023 target with an unknown target in 6bba_05db0fb1, source 40948. This loses the known-match diagnostic; it does NOT establish the unknown target as a biological negative. Preserve the original unknown label and do not train on it as a negative. Existing nonboundary recovery remains the same single C036 recovery; it is not a new C051 gain.

**Decision: close this fixed boundary-recentering method without graph integration, extension, T4 or submission.** The decision follows zero incremental recoveries and the lost annotated match, not the old C036 >=5 gate. This is a source-ranking diagnostic, with no graph edits or official pipeline-score delta. Increased registration coverage alone does not establish a useful tracker change.

No retrospective support/confidence/threshold sweep is warranted by these results. Preserve sources, caches, C023/C024 and all technical proofs. This closes the registered coverage mechanism, not every possible registration or optical-flow algorithm.
