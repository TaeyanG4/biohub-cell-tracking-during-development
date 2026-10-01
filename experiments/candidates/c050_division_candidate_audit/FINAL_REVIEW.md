# C050 corrected whole-embryo candidate-scorer review

The two existing C016 learner fits completed successfully. Analysis stopped because checkpoint train_movies contained NumPy unicode scalars unsupported by PyTorch's default weights-only allowlist. The separately recorded recovery allowed only the required scalar/dtype classes, retained weights_only=True, and reran analysis and verification without editing pinned sources, fitting again, or changing checkpoint/prediction bytes. Original failure records remain in state/c050_analysis_recovery_20260928.

All 12 input hashes and 9 output hashes verified. Reloaded checkpoints exactly reproduced saved probabilities; normalization statistics matched only training-embryo features, and actual checkpoint train_movies were disjoint from the opposite test embryo. The registered recipe and threshold 0.9 were unchanged.

| Training embryo → test embryo | Test movies | Strict positive parents | TP / FP / FN | Precision | Restricted recall | Candidate AP |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 → 6bba | 70 | 76 | 6 / 154 / 70 | 0.037500 | 0.078947 | 0.011918 |
| 6bba → 44b6 | 27 | 18 | 1 / 16 / 17 | 0.058824 | 0.055556 | 0.028201 |

Known-label rows total 221,133, including 94 strict positive candidate/parent cases. This is restricted candidate recall, not recall over all 151 division annotations: missing candidates and omitted early/grand positives are outside this diagnostic. Unknown labels were excluded, never treated as negatives. Historical features come from C012-derived candidates; no C023 graph integration or official pipeline improvement has been measured. Frozen public detectors saw both embryos, and biological sample count remains two.

**Decision: close this fixed geometric/intensity/lineage candidate-scorer recipe without graph integration or submission.** Both transfer directions produce substantially more false proposals than true proposals. This corrects the earlier movie-CV interpretation without declaring all division learning impossible. No retrospective threshold or checkpoint search; preserve C023/C024 and C046 scientific hold.

Next evidence-backed question is the known C036 boundary-coverage gap: 1,218 of 2,302 groups were rejected. A single candidate-independent recentered-context method can test that limitation using real image support and unchanged registration confidence rules; no padding or new scorer is needed.
