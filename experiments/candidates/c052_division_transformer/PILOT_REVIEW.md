# C052 pilot review — 2026-09-28

Completed17:45:18KST before ETA17:58.14/14 jobs passed;19144 input and22699
output hashes independently reverified at first scheduled17:48 check. Source
and original study artifacts preserved. No Kaggle operation or submission.

## Technical and supervision checks

- Real capture/off output and cached logits exact on both embryo smoke inputs.
- All87 division capture detection/ILP controls exact against prior C023.
- All22 off metric rows and historical final graphs exact.
- Both model folds train only the opposite embryo from their evaluation.
- Actual production supervision:44b6 has20division windows/39daughter targets/
  19sampling clusters;6bba125windows/246targets/125clusters. Original ordinary
  windows1102/1144. Both600-step logs contain300 real division and300 ordinary
  batches;all usable division sampling clusters were sampled.
- Exact image context audit found1confirmed duplicate pair. Sampling clusters
  are not independent biological events;two embryos remain the validation unit.

## Existing official aggregation

| Group | Score | Delta C023 | Edge delta C023 | Delta old600 |
|---|---:|---:|---:|---:|
| all22 | 0.947464882 | +0.001478710 | +0.000903997 | +0.001330409 |
| heldout12 | 0.957948634 | +0.002786778 | +0.002786778 | +0.001515482 |
| confirm10 | 0.937810766 | +0.000503257 | -0.000778795 | +0.001164326 |
| 44b6 | 0.948709464 | +0.011695807 | +0.011695807 | +0.003512687 |
| 6bba | 0.947050409 | +0.000419799 | -0.000246868 | +0.001098960 |

Aggregate22 graph edge TP+7/FP-11/FN-7 versus C023. Division TP5 unchanged,
FP6->5,FN19 unchanged. No claim that more true divisions were recovered.
44b6 improvement is concentrated in two movies(+6/+4 edgeTP);6bba edgeTP net-3.
Confirm10 and6bba adjusted-edge deltas remain negative vsC023 despite positive
total scores. Both splits/embryos improve versus the prior C037late600 recipe.
Some per-movie score changes come from output node-count changes only. Do not
call every adjusted-edge win a corrected annotation.

## Decision

Proceed to ONE registered unchanged-model75 extension with existing tools,
no training,knob/checkpoint sweep or immediate deployment. The additional
supervision changed real graphs and improved versus old600 in both directions;
the remaining6bba/confirm damage must be resolved on75/all97 and division-link
survival diagnostics. Small improvements need evidence review,no arbitrary>=5
gate. Original C052 stage packets/graphs support read-only diagnostics.

After97,only worthwhile evidence may advance fixed deployment/portable/T4
checks and user-authorized autonomous submission with fresh quota/dedup.
Frozen public detector/head saw both embryos;no hidden-gain guarantee and no
embryo-prefix deployment routing. C023/C0240.954 anchors/final picks unchanged.
