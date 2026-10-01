# C063 disposition — 2026-09-29

The fixed C063 recipe is closed. Both opposite-embryo overall and good-point
localization gates fail. No graph integration, T4 run or submission is justified.

The original study completed 24 compute jobs, then its final analysis failed at
13:45:28 KST. CSV reload promoted FP32 displacements to FP64, causing a maximum
1.96e-7 um algebra discrepancy against the saved residuals and a 1e-9 assertion.
The separate analysis_recovery queue restores only the original displacement
dtype, preserving the assertion tolerance, all gates, original sources, final
1200-step models and predictions. It completed at 13:51:33 KST; the native
Windows completion toast was submitted at 13:51:34. The original failed record
remains unchanged. No retraining or reinference occurred.

Independent science review verified all 2,302 original inputs, all 22 feature
and label chains, both final 1200-step checkpoints and exactly 38,400 source-only
sample draws per fit, including independently reconstructed RNG sequences and
counters. Both prediction files retain 16,931 original pairs, with 9,113 eligible
and 7,818 exact unchanged exclusions. Restored FP32 displacement arithmetic
matches every saved residual within 8.88e-16 um. Separate recovery manifest
verification is recorded under state/c063_review_20260929/verification.json.

| Opposite evaluation embryo | Mean 3D error (um) | Good-point mean 3D error (um) |
|---|---|---|
| 44b6 | 1.660920 → 1.988720 | 1.222710 → 1.630660 |
| 6bba | 1.793354 → 1.958010 | 1.371646 → 1.582025 |

Pooled opposite mean 3D error worsens 1.780917 → 1.960894 um, and mean absolute
z error worsens 1.082821 → 1.194944 um. Large 3D tails improve in both embryos,
but 44b6 axial-tail 3D error worsens, and the larger good-point population is
harmed in both. Four movie means improve and 18 worsen. Nine proposals acquire
new ownership conflicts. Source-fit overall and good-point errors also worsen.
The evidence does not isolate a unique causal explanation for the failure.

Do not tune strength, thresholds, decoder, epochs or windows on these outcomes.
No official graph or leaderboard improvement was measured. C023/C024 remain the
0.954 anchors. Continue only a distinct, justified finite investigation after
checking HANDOFF; state/c064_next_review contains the next-method review.

Evidence: component/analysis/decision.json; state/c063_review_20260929/science/
REVIEW.md, evidence.json and recount tables; state/c063_review_20260929/dtype/
REVIEW.md; analysis_recovery/{plan,status,output_hashes}.json.
