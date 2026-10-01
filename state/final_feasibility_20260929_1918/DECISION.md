# Remaining submission feasibility — 2026-09-29 19:19 KST

The user asks whether a submission with plausible +0.002 to +0.003 improvement
can finish before midnight. No remaining candidate currently has a verified
whole-evaluation result supporting that gain, and no local gain proves a Kaggle
gain. Historical user-reported C0530.954 and C0540.953 despite positive local
results are direct evidence that the mapping is unreliable.

The completed C06644b6 block was scored with the actual organizer metric,
without modifying the live experiment. All27 finished graph hashes and the
actual row counts are saved in review.json and partial44b6.csv. C066 vs C023
is +0.002476510 on THIS EMBRYO ONLY, but vs the existing C052 opposite model
it is -0.001843673 (edgeTP-6/FP+9, divisions unchanged). This is not an all97
gain and not the eventual pooled deployment. Continue the registered other
embryo and full review; do not promote from the partial result.

At19:18,43/70 of the remaining inference movies completed. Comparing their
measured inference+ILP time to exactly the same movies from C065 gives a
0.9028 runtime ratio. Unfinished-movie historical runtimes project31.3min
inference, followed by about21.2min replay and5min metric/hash margin.
Projected transfer completion is20:16, later than the original19:54 ETA.
These are estimates, not a promise; preserve the launch record.

- C066 then needs145min production +60min remote/submission +20min margin:
  projected00:01, just outside cutoff. Latest start under this budget is20:15.
  Recalculate actual time after completion; do not launch if it cannot fit.
- C067 known diagnostic97 improvement+0.001547598 remains the strongest
  measured remaining uniform-model result. Same-device recovery55 + local30
  + remote60 +20margin projects23:01 if started20:16. It is the practical
  alternative if C066 transfer is weak or late. Exact original gates remain.
- C068 research is ongoing. If it needs the same125+110min local recovery,
  a20:14 finish already projects00:09 before extra collection/review time.
  Do not treat its slots or speculative improvement as guaranteed feasible.

Priority stays evidence-based: finish C066; compare both embryo results with
C052 and the measured C067 alternative, then allocate the local GPU. C066
production receives priority only with worthwhile evidence AND sufficient
time. No fifth candidate, threshold/epoch tuning, concurrent local GPU jobs,
weakened exactness gates, or post-midnight submissions. C065 accepted56671187
is complete; never poll its score or submit it again. Remaining quota maximum3
must be freshly checked before each action. Keep the earlier19:44 heartbeat
check because the production go/no-go cutoff is tight.
