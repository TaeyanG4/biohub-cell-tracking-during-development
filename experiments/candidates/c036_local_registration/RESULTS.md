# C036 local image registration diagnostic

No graph edits or official-score delta were produced.

| Embryo | Groups | Trusted | Proposals | Fixes / harms | Net | Positive movies | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 44b6 | 459 | 328 | 1 | 1 / 0 | 1 | 1 | False |
| 6bba | 1843 | 630 | 0 | 0 / 0 | 0 | 0 | False |

Advance to official-replay integration review: False

- No graph changes or official modified-graph score. Per-source proposals ignore global assignment conflicts and later C023 stages.
- Ground truth defines this annotated reachable-source diagnostic only; it never enters the image estimator or proposal rule.
- Unknown targets remain unknown; retrieval misses are not confirmed biological negatives.
- The public detectors saw both embryos. These movie IDs are local validation, not an independent hidden-test distribution.
- Global phase correlation can be wrong; local NCC models translation only and rejects boundaries, ambiguous peaks and inconsistent motion.
- Failure closes this fixed two-scale algorithm, not every possible optical-flow or registration model.
