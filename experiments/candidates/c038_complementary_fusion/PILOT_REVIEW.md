# C038 official pilot review, 2026-09-27 12:25 KST

Finished11:46:14 KST. All22 off-controls reproduce C023 counts and adjusted-edge scores. All7 pinned sources and9 supplemental input hashes match. No inference, scoring or source-drift error was found.

| Mode | All22 delta | Heldout12 | Confirm10 | Test44b6 | Test6bba |
|---|---:|---:|---:|---:|---:|
| appearance | +0.000328641 | +0.000355371 | +0.000303872 | +0.001727786 | +0.000178392 |
| agreement | +0.000108079 | +0.000114920 | +0.000101994 | +0.000554087 | +0.000059878 |

All deltas are local official-formula adjusted-edge gains; division counts are unchanged. Both modes are positive in both splits and both embryos. These small gains warrant extension under the user's explicit instruction, but do not establish hidden-test improvement.

- All predictions, including unannotated ones, enter the GT-free eligibility process:111126 source groups. No GT source whitelist.
- Appearance:1108 proposals,300 changed edges. Official per-movie edge scores improve on4 movies, decline on2, tie on16. Count deltas sum to TP+1/FP-5/FN-1. Harmful movies are6bba_3abfe10a and6bba_474be664.
- Agreement:129 proposals,45 changed edges. Official scores improve on2 movies and tie on20, with no pilot score losses. It removes one counted FP each on44b6_12dfb391 and6bba_3db54e20, with no change in TP/FN. **This is not a measured recovery of the earlier diagnostic's lone missed true link.**
- Most proposals are suppressed by degree/collision protections. Changes are applied after C023 restore, so there is no later C023 stage to erase accepted changes. Exact source/edge overlap and whether restore already fixed the old diagnostic proposal need the separately registered passive graph-capture follow-up.

## Decision

Advance both unchanged modes to matched extension75 and official97 aggregation. Also test a small fixed combination with C037 early150, which had the strongest aggregate adjusted-edge pilot gain. This choice is retrospective and exploratory; no checkpoint/threshold sweep. Actual combo off/appearance/agreement replay must be scored against both C037-off and C023, not obtained by summing previous deltas. No other C037 arm is launched in this follow-up.

`followup_extension/README.md` records the finite queue. It first passively saves full pilot returned graphs and requires all66 official rows to match the completed pilot. Graph audit uses no GT in decisions; unannotated edges remain unknown. Source/models/raw pilot outputs stay unchanged. All notebooks remain local-only. Pooled/fixed deployment model, portable notebook, actual T4 validation and exact-version submission are still future requirements. C023/C024/final picks remain unchanged; no slots used.
