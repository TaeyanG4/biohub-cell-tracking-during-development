# Frozen DeepCenter coordinate probe: provenance blocker

2026-09-29. Bounded source-provenance review only. No model inference,
training, adapter, candidate change, background process, queue, or ETA.

## Verdict

The reviewed package does **not unambiguously establish the exact historical
checkpoint's label-coordinate convention**. Do not run the proposed coordinate
probe or choose a conversion by comparing GT scores. This is a limitation of
provenance for a new coordinate use; it does not invalidate C023's existing
DeepCenter acceptance-score path or establish a scored pipeline bug.

The package manifest pins the shipped training source and best checkpoint as
individual files. Its best checkpoint summary is epoch 2; the bundle is named
as an epoch-500 snapshot. The published coordinate contract maps pooled XY
indices to original pixels by `4 * pooled + 1.5`. In contrast, the shipped
training `make_heatmap` places targets at `[z, y/4, x/4]`, whose literal label
inverse is `[z, 4*y, 4*x]`.

`../deepcenter_contract.py` AST-extracted that exact manifest-pinned function
without modifying it. Synthetic annotation `(16,32,48)` gives a unique target
mode `(16,8,12)`. The two inverses return `(16,32,48)` and `(16,33.5,49.5)`.
The difference is 0.609375 um on each XY axis (0.861786 um in Euclidean norm),
with no z difference. No real-data predictions or scores were used.

The pack builder copies existing weights/config/history and separately calls
`copy_source_scripts(output)` on then-current repository scripts. Thus its
manifest binds the shipped files to the package, but does not bind epoch-2
training execution to that source revision. The shipped `save_checkpoint`
function stores config, model state, optimizer state, epoch, best score, and
history; it does not record a source revision or coordinate-convention tag.
The snapshot manifest likewise records copied files/config rather than an
epoch-2 training-source digest. The actual tensor checkpoint's metadata was
not additionally loaded in this bounded review; no stronger absence claim
about its contents is made.

## Distinctness from previous studies

The proposed signal is the location of a **learned full-frame centre heatmap**
peak for an existing node. That differs mechanistically from raw-intensity
snapping, C055/C056 node readmission, and C036/C051 temporal registration.
C023's current `deepcenter_score_point` takes a local score maximum and
discards its location. Reviewed evidence did not identify an exact completed
duplicate of this fixed 256-point information test. That limited distinctness
does not resolve the provenance blocker or establish effectiveness; no
repository-wide absence claim or new launch is justified.

## Consequence

Retain the frozen probe as an unlaunched, blocked proposal. A source-supervised
centre learner with a deliberately specified and tested coordinate contract
is a scientifically cleaner future option than inferring the old checkpoint's
coordinate convention from validation scores. No runtime estimate is attached,
because no inference adapter or queue is being prepared.

Decisive source locations: packaged training `make_heatmap` line 276;
`save_checkpoint` lines 431-455; pack-builder `copy_source_scripts` line 140
and `build_artifact` line 248 (copy call at 263); artifact manifest
`coordinate_contract` and `contents.source_scripts`.
