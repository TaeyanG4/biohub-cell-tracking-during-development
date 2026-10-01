Kaggle code competition "Biohub - Cell Tracking During Development" (3D zebrafish nuclei tracking). Repo: H:\dev\kaggle-data\biohub-cell-tracking-during-development. Deadline 2026-09-29 23:59 UTC (about 3 days). Goal: improve the hidden-test score (+0.001 is useful, about +0.01 would be great).

## Current state
- Best public LB 0.954, rank 197 / 3,920 (silver cut-off = rank 196 at 0.954). The public LB is 29 % of a hidden test made of embryos that are not in train.
- Best submissions: C023 = public x138 notebook (pilkwang UNet3D detector + node transformer, dual seed, ILP, Hungarian motion relink, safe-division rule, x138's public V1284 coordinate head) + our every-pair jump-stabilized relink + ILP-edge restore. C024 = the same with a 2-head ensemble (ours + x138's).
- Metric: size-weighted adjusted edge Jaccard + 0.1 x division Jaccard.
- Details: HANDOFF.md (section 23 = current state, sections 20-24 = recent work), AGENTS.md (existing tools), experiments/submission_log.csv.

## Already tried (do not repeat)
- LB gains: jump-stabilized relink + ILP-edge restore (+0.001), x138's public head instead of ours (+0.001).
- No gain or worse: tuning x138's knobs (relink gates, readmit / gap-close / linefit, division geometry, detection threshold and weights, edge fusion weights, ILP costs, TTA variants); learned division scorer; appearance-based division CNN; node-count budget and track pruning; ILP fork re-injection; probability-level TTA pooling; AdaBN; our own coordinate heads (v1-v4) and a 3-head ensemble; weak-edge filter; frozen-frame prior; ILP-seeded relink flow.
- Residual errors of the best pipeline: ~45 % undetected or unmatched endpoints, ~35 % wrong associations, ~20 % half breaks, ~0 % pure gaps. Divisions: 30 TP / 46 FP / 121 FN on 97 local movies.
- Pitfall: all local movies are training movies for the public detectors, and our heads were trained on the same two embryos, so local gains often do not transfer. Settings that gained +0.002 to +0.005 on 97 local movies lost 0.001-0.004 on the LB.

Do not push kernels or submit to Kaggle without the user's explicit OK (the 5 daily submissions are shared with teammates).
