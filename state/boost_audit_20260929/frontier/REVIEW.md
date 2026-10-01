# Public-source audit for +0.002 to +0.003, 2026-09-29

## Conclusion

No cached public source establishes a measured post-patch improvement beyond our 0.954. The existing SQLite radar contains 272 notebooks. Its last broad crawl at 2026-09-28 01:35:45 UTC checked 232 successfully with zero new notebooks and zero score improvements. Highest cached public source score is 0.953: x138 and copies, plus Amanatar's **older** best version. This is not a claim that no new public work appeared since then.

One bounded read of nine relevant third-party notebooks reused the existing radar `enrich_public_scores` helper (four workers). The initial sandboxed attempt failed with socket-policy WinError 10013. One authorized escalation retry of the same nine refs at 2026-09-29 07:57 KST succeeded for six, failed for three with SSL INVALID_SESSION_ID, and was not retried again. Fresh results: Harmonic Fusion V3 0.953/best 0.953; Anvith original 0.953/best 0.953; Amanatar 0.901/best 0.953; Kunal current 0.904/best 0.953; Hengck end2end linker and noisyislands Transformer both no score. No observed best-score improvement. Hengck point detector, andnyu synthetic conditional, and zhincez morphology failed SSL and remain stale. The helper reads scores only; it does not establish current source/version equality. No own submission was read, no notebook source was refetched, no weights were downloaded, no shared DB or source changed. Global Python was accessible after escalation, so its initial failure was sandbox visibility rather than a proven missing installation.

## Reliability of advertised frontier claims

- `amanatar/optimized-biohub-max-score`: cached current V6 public 0.901; best 0.953 belongs to V4. Claimed 0.965+ was never verified; actual V6 T4 visible4 0.9250646856464496 used fallback after missing horizon globals on every movie. C039 already repaired/isolation-tested components and closed them. No revival.
- `flexonafft/biohub-lineage-forge-precision-tracking`: targeted source review 2026-09-29 already pinned V12 / 353597016. Best 0.947 is V8; V12 own score unavailable. Its guarded readmit was C055; both that and the StrongUNet C056 adaptation now closed. Its selected local score lacks a readmit-off control and is not independent validation.
- Cached Harmonic Fusion V3 / Kunal / Anvith original sources: copies of the known x138 stack. The fresh Kunal current score fell to 0.904; its new/current source was not fetched and is not established identical. Its best 0.953 does not prove current-version quality.

## Concrete remaining source leads

### 1. Temporal nucleus half-max size: a genuinely untested feature, not proven score gain

Source: `state/notebook_radar/pulled/zhincez__a-dividing-nucleus-gets-smaller-not-dimmer/a-dividing-nucleus-gets-smaller-not-dimmer.ipynb`, code cell 8 `probe`/`curve`, source SHA256 `1dd8e153fdb16061f61d107a1357fca0ab1839eac0e51f00a774ad6465d6d9ab`.

`probe` takes the original raw 11x11x11 voxel box, background p10, peak=max, and volume=count(voxel >= background+0.5*(peak-background)). `curve` follows the annotated parent and both daughters over t-6..t+6, averages the daughters at each future step, normalizes each curve by t-6. Author's reported median shrink is largest at daughter t+3 (~27%); brightness stays almost flat. Source outputs were cleared, so this is a published EDA observation, not independently reproduced classification evidence. Boxes are highly anisotropic in physical space (about 17.9x4.5x4.5 um), can be truncated at boundaries, and may include nearby nuclei. Published controls use GT trajectories and are easier than real fork candidates.

Why not duplicate C049/C050: C049 used 3-frame (3,8,32,32) image CNN; C050 inherits peak/mean intensity with no explicit half-max size trajectory. HANDOFF explicitly retains temporal size as an untested lower-priority feature. Explicit future t+3..+5 size is absent from those recipes. A new fixed feature diagnostic on actual production fork candidates, with track-derived centres and whole-embryo opposite folds, is distinct. GT may only label/evaluate; it cannot provide inference trajectories. Unknown targets remain unknown. Deduplicate raw overlap/effective events. Reuse existing extraction, candidate enumeration, classifier, and evaluator; no new graph scorer or threshold sweep.

Cost estimate: 1-3 h bounded feature/label and candidate-survival pilot if the existing global scientific runtime is available; compute itself is small relative to model fitting. Do not interpret that estimate as a measured benchmark. Go/no-go signal is per-parent precision about 0.3 at recall about 0.3 in both opposite-embryo directions, then actual graph replay. One true recovered fork is about +0.0005 on the current 97-movie division arithmetic; 4-6 genuinely new forks can reach the user's numerical target locally before FP/edge costs. There is no guarantee it transfers to LB. This is the most economical genuinely new source signal.

### 2. Independent Hengck end-to-end linker: untested model with different supervision

Source: `state/notebook_radar/pulled/hengck23__end2end-cell-linker-raw-edge-ja-0-9-no-ilp/end2end-cell-linker-raw-edge-ja-0-9-no-ilp.ipynb`. Public dataset `hengck23/hengck23-cell-point-detector-demo`, `model_v12.py`, checkpoint `00000008.pth` (~43 MB). This is **not** the retained StrongUNet model_v5/00000030 detector used in C002/C056. A model-v12 edge complementarity diagnostic has no located implementation or closed result in src/HANDOFF.

It uses FOCUS-3D pseudo-centroids and simulated motion without Kaggle GT labels, then a StrongUNet pyramid plus transformer to predict edges. Source runs own detections, forward/reverse softmax product, top1 conflict resolution, and no forks. Claimed raw edge J about 0.9 on 11 training movies is not an official total/LB score. Although labels differ, it saw the same source images, so calling it an honest independent validation would be wrong. Existing 09-24 audit language asserting that should not be repeated.

Useful bounded first step: obtain only code/checkpoint metadata, inspect whether transformer can score C023's frozen detection candidates, then evaluate ranking complementarity on existing ambiguous groups with no graph modification. Never bulk-add its detection set (C056 burden evidence). Unknown model architecture and own-detection feature conditioning make integration less predictable than the size diagnostic. Planning cost ~4-8 h to inspect/adapter/small inference, no demonstrated runtime yet. Stop if it needs a full new pipeline or gives no independent ranking signal. No measured +0.002 potential currently.

### 3. Synthetic budget-neutral detector swap: lower priority

`andnyu/biohub-synthetic-conditional-third-model`, weights `bhpepper/biohub-synthetic-5fold-ensemble-v1/synthetic_5fold_swa.pth`. Replaces up to 0.5% weakest base peaks by independent synthetic-model peaks with >=4-frame tracklet support, base probability near threshold, and probability gain >=0.005. Public 0.946 vs author's other 0.945 is confounded by removal of a negative rank patch. Existing synthetic edge tiebreak already scored =base. C055/C056 show weak complementary peak yield, while swaps can destroy valid base nodes. No reason to prioritize this over the first two; do not call count neutrality proof of node/edge neutrality.

## Recommended decision

Among these public-source leads, an explicit temporal-size candidate diagnostic is the cheapest genuinely untested mechanism. It has no actual candidate evidence yet and should remain secondary to any stronger internally measured mechanism; author-claimed EDA alone does not justify integration. Keep Hengck v12 as a second independent-association lead if runtime/data access and time allow; no model download was requested or performed here. Neither is a validated improvement; do not submit solely from public marketing or median EDA. Retain C023/C024 anchors. Do not rerun C055/C056, readmit-off, old knobs or already-closed transformer recipes.
