# Public notebook audit and lineage inventory

Heuristic audit only. RED_FLAG means the source contains an explicit exploit/sentinel pattern; REVIEW means manual physical-validity audit is still required.

| Notebook | Author | DET | Secondary edge | Bidir | Audit | Inputs |
|---|---|---:|---:|---:|---|---|
| 🧬 Biohub LB : 941 | analyticaobscura | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| Biohub 0.942 LB, one knob past the public line | busyaprime | 0.96 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>busyaprime/biohub-knob-provenance<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| Biohub 0.941 LB, every knob and who set it | busyaprime | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>busyaprime/biohub-knob-provenance<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| Biohub Harmonic Fusion | flexonafft | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| Biohub Cell Tracking | kunaldesale2408 | 0.960 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1<br>pilkwang/pilkwang-public-dataset-for-notebooks-figures |
| Biohub Lineage Forge — Precision Tracking | flexonafft | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| 0.940 | nusrati | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1<br>pilkwang/pilkwang-public-dataset-for-notebooks-figures |
| UNet baseline - inference & submission | thibautgoldsborough | - | - | - | PASS_HEURISTIC | thibautgoldsborough/cellmot-baseline-artifacts |
| Biohub Cell Tracking: Learned Graph w Gap Recovery | pilkwang | - | - | - | REVIEW | pilkwang/biohub-tracking-support-pack-50ep-v1<br>pilkwang/pilkwang-public-dataset-for-notebooks-figures |
| Biohub Cell Tracking: Two Seeds Logit Blend | pilkwang | 0.96875 | 0.15 | - | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1<br>pilkwang/pilkwang-public-dataset-for-notebooks-figures |
| Biohub Harmonic Bidirectional Association V1 | raykkretzschmar | 0.96875 | 0.15 | 0.20 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| biohub-942tta | redoctopusk | 0.965 | 0.15 | 0.15 | REVIEW | pilkwang/biohub-deepcenter-unet3d-center-prior-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1 |
| Biohub Cell Tracking: 0.946 LB | reyhanksatria | 0.965 | 0.15 | 0.15 | REVIEW | <br><br> |
| Biohub Cell Tracking V4 UNet ILP Reproduction | yaroslavkholmirzayev | - | - | - | REVIEW | pilkwang/biohub-tracking-support-pack-50ep-v1 |
| 🧬Clean Approach + Lightweight Local CV | No Hack | yusuketogashi | 0.96875 | - | - | RED_FLAG | pilkwang/biohub-tracking-support-pack-50ep-v1<br>thibautgoldsborough/cellmot-baseline-artifacts |
| No-Hack | Biohub Cell Another Approch 3rd | yusuketogashi | 0.96875 | 0.15 | 0.20 | REVIEW | pilkwang/biohub-local-association-ranker-unet300-v1<br>pilkwang/biohub-temporal-unet3d-seed314159-v1<br>pilkwang/biohub-tracking-support-pack-50ep-v1<br>pilkwang/pilkwang-public-dataset-for-notebooks-figures |
