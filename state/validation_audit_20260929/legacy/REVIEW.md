# Legacy comparison corrections — 2026-09-29

Completed14:41:55 KST; independent2,821inputs/317outputs,313uniquegraph rows and
27aggregate rows passed. Models, original graphs and old experimental records
were unchanged. No retraining or competition action.

| Actual organizer result | Corrected effect and decision |
|---|---|
| C037 late600 parent97 | +0.001057126 vsC023. |
| C052 incremental effect | +0.000750676 over C037late60097; +0.000562530 on extension75. Both embryos improve over parent. Old replica reported only+0.000243356 on97 and−0.000034822 on75. This strengthens the division-supervision direction. |
| C055 readmission-off97 | +0.000414898 vsC023 overall, but44b6−0.002696666; only+0.000046534 on75. Previous mixed-domain limitation remains. |
| C055 v12 guarded97 | +0.000243910 overall,44b6−0.002842109,75−0.000067545;−0.000170988 versus readmission-off. Its own component is negative; no promotion. |
| C05722 | +0.000764147 vsC023 but−0.000632461 vsC052;44b6−0.012249943 andheldout12−0.004999444 vsC052. Closure stands. |

The current user deadline/priority reset justifies one distinct C065 direct
pooled production refit of the supported C052 recipe. Previous C053 and C054
combined models by parameter or output averaging. No claim that refitting will
improve the hidden score. Full97 actual metric and deployment controls remain
necessary. Existing scored negatives/ties do not become new submission slots
because their local replica was corrected.

Source rows: actual_summary.csv and actual_per_movie.csv. Independent proof:
independent_verification.json and verify.py. Actual metric provenance and
limitations: ../REVIEW.md. The original historical files remain unchanged.
