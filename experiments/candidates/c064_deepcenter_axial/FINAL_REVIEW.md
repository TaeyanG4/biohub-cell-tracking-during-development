# C064 fixed axial probe — closed 2026-09-29

The frozen DeepCenter axial proposal fails the fixed gate in both embryo
groups. Do not integrate it into a graph, run T4, submit it or tune the window,
threshold, blend or decoder from these outcomes.

Timing completed14:08:07KST (native toast14:08:12). Two exact production
FP32/8TTA fields took0.4229124seconds. All912inputs and6outputs verified.
Signal completed14:10:43KST (native toast14:10:44), two jobs,921inputs and469
outputs verified. Both model-state digests equal the benchmark and original
state. Two benchmark fields were reused;229remaining fields were calculated.
No training, graph change, new matching or Kaggle action occurred.

Independent saved-field recount reproduced all256 original IDs/GT targets,
all231profiles, every displacement, all before/after3D and abs-z errors,
unchangedXY, exclusions and original-all-node ownership. Its gate exactly
agrees with the registered analysis. The128points per embryo are tail-enriched;
44b6 was DeepCenter training data and6bba was checkpoint-selection validation,
so these numbers are not independent population or graph-score estimates.

| Embryo | Group | Mean3D error (um) | Mean abs-z error (um) |
|---|---|---|---|
|44b6|All128|2.736136 → 3.412213|1.878906 → 2.678711|
|44b6|Good66|1.151290 → 1.696275|0.590909 → 1.255682|
|44b6|3D-tail58|4.520567 → 5.297947|3.306034 → 4.202586|
|44b6|Axial-tail12|5.897540 → 7.648043|5.552083 → 7.177083|
|6bba|All128|2.972298 → 3.834245|2.170898 → 3.173828|
|6bba|Good57|1.427765 → 2.442354|0.798246 → 2.052632|
|6bba|3D-tail64|4.341497 → 5.143847|3.427734 → 4.265625|
|6bba|Axial-tail18|5.387925 → 5.001871|5.055556 → 4.423611|

Both groups have full real support.77/95points move, with19/6ownership
conflicts respectively. One group's axial-tail improvement does not outweigh
the overall,3D-tail and good-point failures. No official score was measured.

Evidence: analysis/decision.json,summary.csv,per_movie.csv; timing/signal
plan,status,output_hashes; state/c064_preflight/{timing_verification,
signal_verification,independent_recount}.json. Sources and results remain
immutable. C023/C0240.954 anchors are preserved.
