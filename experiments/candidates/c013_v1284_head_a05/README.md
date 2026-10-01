# Candidate C013: x138 pipeline + our V1284-compatible head at half output scale

Same as C012 (see `../c012_v1284_head/README.md`) except the head file: head v1 with its output layer
(weight and bias) multiplied by 0.5, which lowers the mean shift from ~0.83 um to ~0.55 um.

End-to-end on the 12 held-out movies: 0.9425 vs 0.9367 for zero mode (+0.0058); adjusted edge Jaccard
improves on both embryo types (44b6 +0.0013, 6bba +0.0070), whereas C012 (full head) scores higher overall
(0.9470) but loses 0.0038 adjusted edge Jaccard on 44b6. C013 is the conservative variant.

- dataset file: `taeyangg4/biohub-c012-v1284-head/v1284_head_a05.pt` (sha256 `209bc591180f8b6968aba4b8ee67861d3a26d5bdd52a5259dd46d399a5806178`)
- kernel: `taeyangg4/biohub-c013-v1284-head-a05` (private, T4)
- build: `python src/build_c012_candidate.py --candidate c013 --head experiments/candidates/c012_v1284_head/dataset/v1284_head_a05.pt --dataset-file v1284_head_a05.pt`
- verify: `python src/verify_c012.py --candidate c013 --head experiments/candidates/c012_v1284_head/dataset/v1284_head_a05.pt`
- submission: only on the user's explicit instruction.
