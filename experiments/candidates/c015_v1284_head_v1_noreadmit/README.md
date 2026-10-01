# Candidate C015: C012 with readmit off

- **Parent**: C011 via `src/build_c012_candidate.py --candidate c015`; identical to C012 (x138 pipeline + our head v1, sha256 `9d3484f794b48c379b657714878ff3d7bee6042dd932ef257fced992b34edda6`, now served from `taeyangg4/biohub-v1284-heads/v1284_head_v1.pt`) except `BIOHUB_READMIT_RADIUS_UM` 4 -> 0.
- **Why**: single-change test of x138's readmit stage with the head. Local: 12 held-out movies 6bba equal to C012 (0.9508 vs 0.9510), 44b6 adj-edge +0.0048; 10 unseen 6bba movies +0.0022 (edge better on 7/10, one division FP fewer). thtennant's head-less LB pair also favoured readmit off (0.946 -> 0.947). C014 (0.949) changed the head and readmit together, so this isolates readmit.
- **Predicted LB**: 0.952-0.953 (6bba proxy).
- **Verification**: `python src/verify_c012.py --candidate c015 --head experiments/candidates/c014_v1284_head_noreadmit/dataset/v1284_head_v1.pt --dataset-slug biohub-v1284-heads --env BIOHUB_READMIT_RADIUS_UM=0` - ALL PASS.
- **Kernel**: `taeyangg4/biohub-c015-v1284-head-v1-noreadmit` v1, pushed 2026-09-24 01:30 KST. Submission pre-authorised by the user for the last 2026-09-23 (UTC) slot.

## Kaggle run and submission

- Kernel COMPLETE 2026-09-24 01:44 KST; pinned head loaded, readmit inactive, 234,554-row submission.
- T4 vs local head v1 run: refined coordinates |diff| median 2.4e-4 um; visible-4 official 0.9229 (= local prediction).
- Submitted 2026-09-23 16:47 UTC, ref 56499136 (last slot of the day; pending at time of writing).
