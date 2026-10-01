# 2026-09-14 execution snapshot

Current source of truth: `../../HANDOFF.md`.

Key changes since the 2026-09-13 research-plan snapshot:

- Submission 56193365 completed at public LB 0.933. no-linefit branch closed.
- Submission 56200431 completed at public LB 0.936. dual-HOCT+no-linefit branch closed.
- Standard HOCT submission 56186731 is 0.944; edge-TTA DET0.96 submission 56171162 is 0.945.
- Clean public-0.947-settings anchor kernel `taeyangg4/biohub-dctta-lite-public0947-anchor` v1 completed and validated. Visible patched metric = 0.8936. Authenticated Kaggle history confirms submission ref `56209758` is PENDING. The submit command produced no useful stdout, so silent output must not be treated as failure.
- R3 normalized-HOCT feature extraction completed on the four visible diagnostic movies: 1,056,190 rows x 288 normalized features.
- BCE probe improves embryo-holdout AUC but not parent top-1 ranking; do not deploy based on AUC/AP.
- Pairwise target-parent ranker changes only three winners on held-out 6bba: 2 fixes, 1 break; no changes on held-out 44b6. Not submission-ready.
- Next R3 gate is graph-level conservative edge replacement plus official metric and broader embryo validation.

See `../../HANDOFF.md` for implementation traps and promotion gates.

