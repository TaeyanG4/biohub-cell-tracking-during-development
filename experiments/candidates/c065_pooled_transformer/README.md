# C065 — one directly fitted production Transformer

Registered direction 2026-09-29 under the user's renewed request to revisit
validation and prioritize improvement before **2026-09-30 00:00 KST**, including
submission validation. Five submission slots user-reported; recheck shared quota
only before an actual submission. Preserve C023/C024 scored anchors.

## Why this is first

Actual organizer rescoring confirms C052 opposite-embryo component gain
+0.001807802 on97, positive in both embryos. The fixed parameter mean C053
retains only+0.001108508, and output mean C054+0.001272997 locally. Their reported
public scores do not demonstrate the desired improvement. The deployment
conversion remains a plausible bottleneck; this comparison does not prove
averaging alone caused the loss.

C065 directly fits ONE production model on the union of the preserved C052
supervision, instead of combining independently optimized model weights or
outputs. This is standard production refitting of a supported component, not
a new independent validation claim. HANDOFF and C037/C041/C052/C053/C054 sources
checked: the C052 production versions averaged models; this pooled C052 fit
has not previously run. C057's changed objective is not reused.

## Fixed first phase

Reuse the exact C052 train function, FixedSampler and C037 learner with
`group='pooled'`, which the original learner already supports. Original base
Transformer initialization; fixed600steps/seed3701/AdamW1e-5/weightdecay1e-4/
cosine600/teacherKL0.1/divisionweight2/dropoutoff/FP32 mathSDPA. Exactly300
division-cluster and300 original24 ordinary batches. All source records from
both embryos are eligible. No new labels, crop extraction, checkpoint or
hyperparameter choice. Unknown nodes remain attention context, not negatives.

The combined division sampling pool is the union of both source pools. This
changes exposure compared with equal fold-model averaging and must be reported.
Do not claim full199 training or balanced independent biological events.

Pin every packet/label/model/source and copied runtime. Dry-reconstruct the600
sampler choices before training, require all existing division clusters visited.
After the only final fit, verify actual trace, all learned tensors, original
teacher digest, source membership, exact checkpoint reload on real packets and
the actual runtime apply hook. Training loss is not a promotion criterion.

## Separate deployment phase

Following fit integrity checks, use a fixed GT-free checkpoint on EVERY movie,
with unchanged C023 detector/head/feature indexing/Transformer fusion/ILP and
postprocessing. Reuse existing portable C053 machinery with truthful C065
provenance; no embryo-name router. Independently execute and compare original
and portable inference on two full movies; all97 replay uses actual organizer
metric on integer output graphs, regardless of small mixed22 results.

Report total/embryos/movie effects, edge and division TP/FP/FN and node cost
against C023, C052 diagnostics and C053/C054 deployment comparators. Do not
require every localization surrogate stratum to improve. This pooled model's
local evidence remains fit-domain technical evidence; C052 supplies the
component-transfer evidence. Any worthwhile submission still requires portable
writer equality, actual T4 exact version/hashes/fallback0 and fresh quota/dedup.
No submission is authorized by this README alone; existing user authorization
applies after those real checks. No Kaggle action in the fit phase.

Every hidden phase records real PID/start/ETA and immediately attaches the
existing native Windows completion/failure notifier. Observers stay outside
hashed experiment directories. Same biohub-t4 follows actual phase ETA-10minutes,
then20minutes (short phase first launch+30seconds). Complete all work by midnight;
reserve the final hours for T4 and submission rather than starting long fits.
