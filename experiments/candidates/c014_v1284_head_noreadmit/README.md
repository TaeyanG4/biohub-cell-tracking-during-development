# Candidate C014: x138 pipeline + our V1284 head v2, readmit off

- **Parent**: C011 (x138 minus its private head), built with `src/build_c012_candidate.py --candidate c014`.
- **Changes vs C011**: V1284 'candidate' mode with head v2 from the private dataset `taeyangg4/biohub-v1284-heads` (`v1284_head_v2.pt`, sha256 `78053e749f3475a18d6f85a7564375a1dd007a626ca13f6773d50ac6a132042e`, pinned); `BIOHUB_READMIT_RADIUS_UM` 4 → 0. Every other setting equals x138.
- **Kernel**: `taeyangg4/biohub-c014-v1284-head-noreadmit` (private, T4). Submission authorised in advance by the user ("C014도 완료되면 제출해").

## Why head v2 and readmit off

End-to-end on the 12 held-out movies (never captured for head training), local inference + ILP with the Kaggle code, x138 post-processing with readmit disabled, official formula on rounded coordinates:

| head | total | adj edge all | adj edge 44b6 | adj edge 6bba | division tp/fp/fn |
|---|---:|---:|---:|---:|---|
| none | 0.9404 | 0.9246 | 0.9107 | 0.9281 | 3/4/12 |
| v1 (30 movies) | 0.9478 | 0.9311 | 0.9096 | 0.9366 | 3/3/12 |
| **v2 (71 movies, 44b6/6bba pairs balanced)** | **0.9485** | **0.9335** | **0.9221** | 0.9364 | 3/5/12 |
| v3 (187 movies, 85 % 6bba pairs) | 0.9417 | 0.9317 | 0.9177 | 0.9352 | 2/5/13 |

With readmit on the same heads score 0.9470 (v1), 0.9434 (v2), 0.9353 (v3); head-less 0.9367 → 0.9404 when readmit is removed. thtennant's LB agrees on readmit (gapfill 0.947 → readmit 0.946).

v2 is best on both the total and the more stable edge component, and the only head that improves both embryo types clearly. The division component rests on 15 GT divisions and swings by ~0.005 per event. Selecting among five variants on the same 12 movies carries optimism; C014 is the configuration with the strongest and most balanced local evidence, not a guaranteed LB gain.

Head v2 CV (movie-grouped, 71 movies): distance to GT 1.619 → 1.301 µm (−19.6 %; 44b6 −13.9 %, 6bba −24.7 %).

## Verification

`python src/verify_c012.py --candidate c014 --head experiments/candidates/c014_v1284_head_noreadmit/dataset/v1284_head_v2.pt --dataset-slug biohub-v1284-heads --env BIOHUB_READMIT_RADIUS_UM=0` — ALL PASS (only labels, the V1284-mode block and the declared readmit setting differ from C011; head loads with `weights_only=True`; bounded shift; metadata).

## Kaggle run and submission

- Kernel COMPLETE 2026-09-23 16:45 KST: pinned head loaded, readmit inactive, gap filler active, 235,927-row submission.
- T4 vs local head v2 run: refined coordinates |diff| median 2.5e-4 um; edge-probability |diff| median 3.3e-5; visible-4 official 0.9159 (= local prediction).
- Submitted 2026-09-23 07:47 UTC, ref 56487249 (pending at time of writing).
