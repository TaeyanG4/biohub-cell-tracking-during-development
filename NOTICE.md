# Notice: third-party work and data

Our own code and documents in this repository are released under the Apache License 2.0 (see `LICENSE`).

## Public Kaggle work this project builds on

The candidate notebooks under `experiments/candidates/` are modified copies of public Kaggle notebooks. Kaggle publishes notebook code under the Apache License 2.0; the original authors keep their copyright. We thank them:

| Work | Author (Kaggle) | Used for |
|---|---|---|
| Temporal 3D U-Net detector, DeepCenter, node Transformer, support pack | pilkwang | detection and edge scoring in every candidate (weights not included here) |
| Harmonic Fusion post-processing, Lineage Forge | flexonafft | association and post-processing chain |
| `biohub-cell-tracking-0-947-lb` | reyhanksatria | our baseline B0 (C001-C009) |
| `biohub-frontier947-readmit-v1` and related series | thtennant | readmission chassis; identified what x138 changed |
| `biohub-x138` and the public V1284 head `biohub-v1284-head-s075` | anvithpothula | base of C011-C070; head used in C023 and the C024 ensemble |
| V1057 ILP-edge restore | josephadamski | restore stage in C021/C022 and later |
| Weak-leaf pruning, V6 components | amanatar | tested in C009 and C039 |
| Lineage Forge V12 guarded readmission (same author) | flexonafft | tested in C055/C056 |
| Appearance descriptors | arnav170 | tested in C034 |
| Structured trajectory model | indarkarhana | tested in C033 |

The organisers' scoring code ([royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)) is used by `src/evaluate_local.py` but is not copied into this repository.

## Not included

- Competition data (images and annotations): available to competition participants under the competition rules; download it from Kaggle. The one image in `docs/figures/sample_frame_tracks.png` is a low-resolution projection of one training frame, shown for illustration.
- Model weights (public or ours), Kaggle kernel outputs, run caches and logs.
- Third-party notebook sources that we only read for review.
