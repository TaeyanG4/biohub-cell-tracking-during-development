# C041 fixed global models review

Finished2026-09-27 20:58:57KST,10 jobs.1586 input hashes and950 output hashes verified; exact two-movie original single-CNN final graphs/metrics and22 C023 off controls passed. No model training/evaluation overlap for24 correction-training movies versus97; public base models saw all199, and repeatedly used local97 is not independent test data.

| Fixed arm | Official22 delta | Heldout12 | Confirm10 | 44b6 | 6bba |
|---|---:|---:|---:|---:|---:|
| Appearance mean cosine | +0.000329361 | +0.000355371 | +0.000305242 | +0.001727786 | +0.000179189 |
| Transformer parameter mean600 | +0.001201508 | +0.001941422 | +0.000541706 | +0.008891505 | +0.000379878 |
| Both | +0.001701601 | +0.002297632 | +0.001169970 | +0.010638341 | +0.000747423 |

All deltas are adjusted-edge gains; division TP5/FP6/FN19 unchanged. Transformer8 wins/14 losses and combined10/12: weighted improvement, not uniform per-movie success. Appearance graph audit299 changed edges, combined332; these counts do not establish correctness. Transformer fixed mean is a new model hypothesis, not equal to averaging model outputs. CNN cosine averaging uses concatenated normalized embeddings, not averaged CNN parameters. Every movie uses identical models; no embryo router or evaluation-selected whitelist.

Decision: construct two portable C023-derived candidates, C042 Transformer and C043 combined. C040 provided positive97 evidence for diagnostic cross-embryo late600; these fixed deployment models have22-movie evidence only. Current authorization permits exploratory submissions for such evidence, without pretending they are hidden-test improvements. Appearance-only retained without consuming a third candidate slot.

Require embedded/runtime/source parity, actual as-configured12 notebook replay and visible4 writer parity; actual Kaggle T4 official visible4 with repair_fallback=0 before exact-version submission. Current C041 notebooks are local-only and must never be pushed. Budget policy: fresh shared-quota readback, at most2 additional exploratory while reserving2, no quota/uncertain-success retries or LB polling. C023/C024 final picks remain unchanged.
