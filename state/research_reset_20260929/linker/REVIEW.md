# Hengck v12 frozen-candidate linker probe, 2026-09-29

The published linker can score C023's frozen detection candidates directly. The
actual checkpoint loaded strictly and executed on the existing 193 production
packets in 32.1 seconds on the local RTX 4070 Ti SUPER. It provides different
association evidence, but replacing the primary linker's rankings is unsupported:
16 known target links improve and 37 previously correct links regress.

## Source and model identity

Local radar SQLite was queried first. The cached notebook is
`hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp`, last run 2026-09-15,
no verified public/LB score. No notebook refresh or own submission polling was
performed. One dataset file listing and only the necessary three selected files
were downloaded from `hengck23/hengck23-cell-point-detector-demo`:

- `model_v12.py`: 21,346 bytes, SHA256
  `586ac08e4429fa0ca0db0c06afdf6d7ef77d459c0260bbf286bf47169d484082`.
- `00000008.pth`: 43,076,221 bytes, SHA256
  `ecd8869de9cf405c1a93a563b4810faad5fd2378f57341e18c4e6b5a87201552`.
- `loss_and_metric_v12.py`: 126 bytes, only a simple DotDict class needed by the
  checkpoint. Safe weights-only load allowed this inspected class; unrestricted
  pickle execution was not used. Published training loss implementation is absent.

This is distinct from C056's `model_v5.py` / `00000030.pth` point detector. The
10,740,518-parameter epoch-8 model uses a residual 3D U-Net with (64,128,256)
channels, sampled decoder/encoder features [d0,d1,e2] totaling 448 dimensions,
four 256-dimensional self/cross attention layers, 128-dimensional metric
embeddings, and a pairwise MLP using appearances, displacement, distance, cosine.
The primary teacher's existing packet features are 64-dimensional and cannot
substitute for this model's features.

The cached notebook states training used FOCUS-3D pseudo-centroids and simulated
movement, no Kaggle tracking annotations, on t=0,5,10,... source frames. This is
different supervision, not independent image validation: the source embryos may
overlap evaluation. Training loss and augmentation details are not available in
the distributed loss file. Published inference picks one outgoing edge and
resolves target conflicts, so it does not create true forks by itself.

## Executed adapter and controls

`probe.py` is a bounded diagnostic, not a graph scorer or replacement pipeline.
It reads the existing C052 production packets and C057 conditional parent labels,
verifies every input hash, runs the original primary pretrained teacher, and
compares Hengck scores on the same frozen coordinate candidates.

The published `unet.make_feature`, `sample_pyr_feature_at_zyx`, and `linker` APIs
work without using its point detections or learned coordinate refinements.
Image preprocessing exactly follows the cached notebook: original z, every
fourth y/x voxel, metadata quantiles .001/.999, lower clipping only. Candidate
coordinates divide by (1,4,4); coordinate roundtrip error is zero. Sampling at
the model's own refined coordinates reproduces all three original node feature
tensors exactly. Inference is FP32, TF32 off, math SDPA, batch 2 frames and one
CPU thread. All logits were finite, strict checkpoint loading passed, and all
teacher packet probes remained below the existing 2e-5 tolerance.

The 12-packet initial API smoke used three equally spaced packets per embryo and
ordinary/division pool, then the diagnostic used the exact pre-existing 193
packet audit (145 division and 48 ordinary). The 193 selection was fixed before
this model was examined. All 1,834 target-level teacher correctness values
exactly match the prior saved audit. C057's existing 7um ambiguity masks are
used only for evaluating known links; unknown cells are not labelled negatives.

| Embryo/pool | Known links | Teacher correct | Hengck correct | Rescued | Harmed |
|---|---:|---:|---:|---:|---:|
| 44b6 division packets | 89 | 82 | 81 | 2 | 3 |
| 44b6 ordinary packets | 101 | 100 | 101 | 1 | 0 |
| 6bba division packets | 1,451 | 1,411 | 1,389 | 12 | 34 |
| 6bba ordinary packets | 193 | 189 | 190 | 1 | 0 |

Among actual daughter links, 13 are rescued and 30 are harmed (44b6 2/2;
6bba 11/28). These are link counts, not recovered forks. Among all links,
16/52 primary mistakes are rescued while 37 primary-correct links are harmed.
The probe is dominated by correlated 6bba examples and is not a test-set score.

## Next finite experiment justified by these results

Do not replace the production linker, bulk-add Hengck nodes, or tune postprocess
thresholds from these results. The relevant signal is complementary ranking,
not higher standalone accuracy. A bounded next diagnostic can retain both score
matrices on these same 193 packets and test ONE predeclared equal-weight mean
of per-target parent probabilities, using the existing conditional masks for
reporting only. Since raw logit scales differ, averaging raw logits would need a
separate calibration justification. Include teacher-duplicate exact control,
both embryos, daughter versus ordinary counts, and gains/losses; no sweep.

If that fixed combination improves known links in both embryos without daughter
loss, use the existing C037 runtime hook and C052 Queue/inference/replay for a
22-movie frozen-detector technical pilot. Preserve C023 coordinates, fusion, ILP
and postprocessing; a full-movie teacher-duplicate off control must exactly
reproduce caches and graphs. Only actual graph scoring can establish whether
new pair rankings survive dual-direction/dual-seed fusion and ILP. The measured
packet cost suggests added inference is tractable; full-movie runtime and memory
are not yet measured. No new training, graph integration, submission candidate,
or demonstrated LB gain was produced by this diagnostic.

Artifacts: `probe_result_193.json` contains input receipts and runtime evidence;
`probe_rows_193.csv` contains every known-link result. Initial 12-packet evidence
is retained separately in `probe_result.json` and `probe_rows.csv`.
