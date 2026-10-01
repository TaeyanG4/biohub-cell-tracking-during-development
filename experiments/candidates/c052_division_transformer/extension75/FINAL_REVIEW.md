# C052 fixed75 extension final review — 2026-09-28

Completed19:52:02KST(all11jobs) beforeETA20:13. Independent first scheduled
review reverified42569inputs/10739outputs. Two actual full-movie cache/ILP
reproductions,75off official metric rows and historical final graphs passed.
Original14-job pilot and all model/source artifacts remain unchanged.

## Existing official score results

| Set | Embryo | Score | Delta C023 | Edge delta C023 | Delta oldlate600 |
|---|---|---:|---:|---:|---:|
| extension75 | all | 0.940506683 | +0.001326714 | +0.001326714 | -0.000034822 |
| extension75 | 44b6 | 0.944184484 | +0.000517083 | +0.002898035 | +0.000090040 |
| extension75 | 6bba | 0.939915193 | +0.001810283 | +0.001145236 | +0.000252137 |
| all97 | all | 0.941787202 | +0.001297189 | +0.001227263 | +0.000243356 |
| all97 | 44b6 | 0.944636164 | +0.002775966 | +0.004849136 | +0.000628004 |
| all97 | 6bba | 0.941289179 | +0.001457319 | +0.000816293 | +0.000452109 |

- All97:edgeTP+80,FP-28,FN-80 vsC023.44b6+22TP/-21FP/-22FN;
  6bba+58TP/-7FP/-58FN. Not every per-movie adjusted-edge change is a
  corrected annotated link;the metric also contains output node counts.
- Official division totals27TP/45FP/124FN:netTP0,FP-1,FN0 vsC023.
  44b6 loses1truefork and1FP;6bba gains1truefork,FP unchanged.
- Positive aggregate edge and total in BOTH embryos and75 vsC023.44b6
  benefits remain concentrated:8wins/19losses on97.6bba45wins/22losses.
- Versus oldC037late600,the pooled75 total is slightlyNEGATIVE(-0.000034822),
  even though75edge+0.000482002. All97 total+0.000243356/edge+0.000681044.
  Do not hide this limitation or infer a leaderboard gain of these sizes.

## Read-only daughter survival

All151actual learned production packets pass saved-logit parity;teacher remains
unchanged and frozen detection coordinates are exactly equal.302daughter
annotation rows are not302independent biological events.44b6 admission32->34
ends in final28->28(1gain/1loss).6bba admission175->176 ends in final136->137
(5gains/4losses). Missing fused probabilities mean<=.02,not zero. Final matched
daughter counts are a diagnostic,not official fork counts or a GT edit whitelist.
This measures limited survival of the extra supervision through the pipeline.

## Decision: one fixed deployment candidate C053

The unchanged75 results resolve the pilot's negative6bba aggregate edge against
C023 and preserve positive whole-embryo transfer evidence. Advance ONE fixed
equal-parameter mean of the two existing C052step600 models,using original C041
averaging and C042 portable patch machinery. No further training,selection of
weights,threshold/checkpoint sweep,appearance arm or biological-prefix router.

The fixed average includes same-embryo fitted parameters. Its97 local scores
areFIT-DOMAIN technical evidence,NOT independent validation. C052 opposite
folds supply component transfer evidence;frozen public detector/head already
saw both embryos. Require actual fixed97 execution,portable12/writer4 parity,
source/model/CSV hashes and actualT4 visible4/fallback0 before autonomous
worthwhile submission with fresh quota/dedup. No C053 upload/push/submission
has yet occurred. C023/C0240.954 anchors and final picks remain unchanged.
