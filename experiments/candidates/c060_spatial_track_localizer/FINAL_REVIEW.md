# C060 completed: fixed localization recipe not promoted

Additional completed diagnosis11:36KST: all22 accepted changes include only5
z-coordinate changes; none of109 originally matched moved points changes z.
Two immutable models on256 fixed source-crop inputs reproduce saved production
guards exactly. Orientation disagreement is122/128 and114/128 on opposite
samples. Even unguarded modes worsen originally good points in BOTH embryos;
source-domain large-offset fit is also weak. The head is trained/nonzero, but
tail mean z proposals remain near zero with wrong signs. This supplies no basis
for relaxing guards or sweeping the prior. Details in
`state/c060_review_20260929/model/REVIEW.md` and `science/REVIEW.md`.

Completed2026-09-29 **11:27:26KST**, all10 jobs succeeded. Native Windows
completion notification was submitted at11:27:31. The experiment ran correctly
but did not establish useful transferable localization. **No97 extension,
pre-association integration, portable/T4 run or submission of this recipe.**

| Official local group | C023 | C060 | Difference |
|---|---:|---:|---:|
| All22 |0.9431126093954155|0.9431126093954155|0|
|44b6|0.9170136575466079|0.9170136575466079|0|
|6bba|0.9472972763229539|0.9472972763229539|0|

All22 movie scores tie. Edge TP/FP/FN and division TP/FP/FN are unchanged.
The two CSV files differ because some coordinates moved; identical score does
not mean inference was skipped. Of517,966 nodes,254,579 have eligible real
three-frame context;9,267 pass exact8-view mode agreement and4,085 move.
No mode-agreed proposal is rejected by the subsequent strict ownership guard.

On8,594 eligible original known pairs, mean residual is1.778759→1.778784um.
699 originally>3.5um pairs go4.694986→4.692913um, with only9 changed points.
44b6 overall improves1.602764→1.602308um, but6bba slightly worsens
1.804916→1.805012um. Originally good<=2.5um mean worsens in BOTH embryos:
1.174591→1.175500 and1.364662→1.364727um. All14,041 original good identities
remain matched to the same GT; one original tail match is lost and one formerly
unmatched prediction becomes matched. This is not evidence of a new recovered
biological cell. The predeclared component gate fails.

## Verification

Independent `state/c060_review_20260929/verification.json` checks all3,824
input hashes and2,657 recorded output paths. No experimental source, data,
model or result drift was found. One observer-file change is explicitly recorded:
`watcher.stdout.log` was empty when outputs were hashed, then the independent
Windows watcher appended the exact successful notification receipt after the
terminal status. Its CP949 JSON matches the separate UTF8 receipt, including
ownerPID and post-completion timestamp. The original output manifest is intact.

All22 frozen baselines equal C055/C023; IDs, counts, times and edges are exact,
excluded coordinates unchanged, runtime masks reproduced. Every moved point
passes strict nearest-original-node ownership against ALL original same-frame
nodes. Both final1200step checkpoints have correct opposite-embryo source
provenance and38,400 source samples. Original writer and actual vendored
official CSV checks pass on both22-movie outputs;fallback0/degradation0.

## Disposition

Close this fixed recipe. Do not relax mode-agreement/prior/thresholds, select
another epoch, or infer a score gain from tiny residual movements. The user's
localization priority remains active as a research direction; this experiment
does not close that direction. Bounded saved-model diagnostics distinguish
centered predictions, orientation disagreement and generalization failure
without training or selecting another deployment variant.

Only two biological embryos and temporally correlated crops are available;
the frozen public detector/head saw both. Stable-known source-lineage filtering
excluded most ambiguous identity-switch cases from supervision. These limits
must carry forward. C023/C0240.954 anchors remain unchanged, C046 is already
submitted56654681. No submission slot was consumed. The completed heartbeat
was deleted; no new experiment is queued.
