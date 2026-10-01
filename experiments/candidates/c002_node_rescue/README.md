# C002 - Selective Node Rescue

Status: GPU PEAK CACHE COMPLETE

Purpose: test whether a stronger 3D point detector can recover B0 endpoint-missing nodes without replacing the full B0 detector.

Completed stage: cached low-threshold StrongUNet peaks on the four visible diagnostic movies using CUDA with CPU threads limited to 1. Total cached peaks: 196,991. No graph edits and no Kaggle submission were performed.

Next gate: when CPU pressure is acceptable, compare cached peaks against endpoint-missing GT cases and B0 nodes to estimate rescue ceiling and false-positive burden.

GPU cache output: `gpu_peaks/`
