# C037 official end-to-end pilot

| Arm | Group | Score | Delta | Edge delta | Edge wins / losses |
|---|---|---:|---:|---:|---:|
| early150 | all22 | 0.946343 | +0.000357 | +0.000357 | 5 / 17 |
| early150 | heldout12 | 0.958110 | +0.002949 | +0.002949 | 4 / 8 |
| early150 | confirm10 | 0.935353 | -0.001955 | -0.001955 | 1 / 9 |
| early150 | 44b6 | 0.948687 | +0.011674 | +0.011674 | 2 / 4 |
| early150 | 6bba | 0.945781 | -0.000850 | -0.000850 | 3 / 13 |
| late600 | all22 | 0.946134 | +0.000148 | -0.000426 | 12 / 10 |
| late600 | heldout12 | 0.956433 | +0.001271 | +0.001271 | 7 / 5 |
| late600 | confirm10 | 0.936646 | -0.000661 | -0.001943 | 5 / 5 |
| late600 | 44b6 | 0.945197 | +0.008183 | +0.008183 | 2 / 4 |
| late600 | 6bba | 0.945951 | -0.000679 | -0.001346 | 10 / 6 |
| late600_blend25 | all22 | 0.945710 | -0.000276 | -0.000276 | 10 / 11 |
| late600_blend25 | heldout12 | 0.955199 | +0.000037 | +0.000037 | 3 / 8 |
| late600_blend25 | confirm10 | 0.936752 | -0.000555 | -0.000555 | 7 / 3 |
| late600_blend25 | 44b6 | 0.938589 | +0.001576 | +0.001576 | 1 / 5 |
| late600_blend25 | 6bba | 0.946156 | -0.000475 | -0.000475 | 9 / 6 |

Review small consistent effects and combinations before closure. No Kaggle operation performed.
- Both base detectors were pretrained on these embryos.
- Cross-embryo fine-tunes are diagnostic folds; hidden inference cannot route by embryo ID.
- A pooled or fixed ensemble deployment model needs its own actual replay and T4 verification.
