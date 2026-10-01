# Independent C061 completed science review

**Close this fixed recipe. Both its synthetic-transfer gate and its real-anchor
gate fail.** Do not extend to97 movies, integrate before association, run T4 or
submit this component. The completed experiment provides new information about
axial learnability, but no evidence of a production localization improvement.

The queue reports completion at2026-09-29 12:17:08 KST. This review reads saved
CSV data and checkpoint metadata only: no forward inference, fit, graph edit,
rematching, guard change or official scoring. Root separately audits hashes
and notifications. Machine-readable independently reconstructed summaries are
in `recount.json`, `synthetic_recount.csv`, `real_recount.csv` and
`movie_recount.csv`; `recount.py` reproduces them.

## Completion, source separation and label accounting

Both checkpoints and their metadata record the fixed1200 steps,43,200 sample
presentations per fit and exactly4,800 examples per each of nine axial classes.
The44b6 fit used its71 movies/2,268 selected known GEFF points;6bba used its128
movies/4,096 points. Training-stem lists, class/sample counts, final recipe and
checkpoint metadata agree. Final logged batch CE was1.049310/1.359265 versus
initial log9=2.197225. These losses are training observations, not independent
performance estimates. The6,364 source records and their shifted views are
correlated observations across only two biological embryos.

Each checkpoint has57,276 synthetic evaluation rows: all nine shifts of every
registered point, on both its source and the opposite embryo. The114,552 rows
contain exactly the expected stem/sample/shift combinations without duplicate
keys. Shift signs, targets and saved signed/absolute errors recalculate exactly.

Each real evaluation contains9,294 eligible original C023 known pairs:
1,174 on44b6 and8,120 on6bba. The full diagnostic adds7,637 pairs held unchanged
(416/7,221), for16,931 originals (1,590/15,341). Independent joins confirm exact
node ID, GT ID, graph row, time and original offsets against both the C061
manifest and immutable C058 labels. Reconstructed exclusions and all saved
paired diagnostic identities agree. Neither corrected coordinates nor the
successful subset determines the reference identities.

This is **fixed-identity residual evaluation**, not proof that identities would
remain matched after graph modification. No corrected graph was constructed
or re-matched, and no actual competition-score change was measured.

## Synthetic task: source learning, incomplete transfer

The no-correction mean is3.611111um when all nine balanced shifts are pooled.
This artificial distribution contains many large shifts unlike real C023
residuals, so its pooled gain cannot establish deployment benefit.

| Trained on | Evaluated on | Domain | Views | Mean absolute z after, um | Exact class |
|---|---|---|---:|---:|---:|
|44b6|44b6|Own-source fit|20,412|1.009535|56.27%|
|6bba|6bba|Own-source fit|36,864|1.236515|49.52%|
|6bba|44b6|Opposite embryo|20,412|1.760496|35.50%|
|44b6|6bba|Opposite embryo|36,864|1.999556|35.07%|

All nonzero shifts improve on the source embryo, and the source zero-shift
error stays below one native plane. The opposite models improve the larger
shifts, but **both +/-1-plane shifts and zero fail in each opposite direction**:

| Evaluated embryo (opposite model) | Shift -1 before→after, um | Shift0 after, um | Shift +1 before→after, um |
|---|---:|---:|---:|
|44b6 (trained6bba)|1.625→1.873622|1.895117|1.625→1.951003|
|6bba (trained44b6)|1.625→2.339905|2.235962|1.625→2.076477|

The registered zero-shift limit is1.625um. Both opposite zero-shift outputs
also have directional bias: mean signed error+1.257440um on44b6 and
-1.120361um on6bba, versus+0.064484/+0.003571um on their own-source zero-shift
tests. Exact zero prediction falls from56.48%/49.49% own-source to29.37%/29.93%
opposite. There are no tied logits in the retained synthetic outputs.

Thus averaging across all displacements would conceal the failures closest
to the original detector's common small-error regime. The complete per-shift
gate correctly rejects both transfer directions.

## Real anchors: material damage on both embryos

The following uses the model trained on the **other** embryo. `Tail` is original
3D residual>3.5um; `axial tail` is original |dz|>3.5um; `good` is original3D
residual<=2.5um. Values include zero/tied outputs, and no ownership filter is
applied after seeing the results.

| Evaluated embryo | Group | n | Mean3D before→after, um | Mean absolute z before→after, um |
|---|---|---:|---:|---:|
|44b6|All eligible|1,174|1.609211→2.370271|0.992440→1.919825|
|44b6|Tail|99|4.807781→5.479628|3.446970→4.284091|
|44b6|Axial tail|29|5.847772→5.860486|5.323276→5.155172|
|44b6|Good|1,000|1.183390→1.974529|0.674375→1.634750|
|6bba|All eligible|8,120|1.805917→2.840888|1.131696→2.337438|
|6bba|Tail|668|4.666698→5.288417|3.247567→3.974925|
|6bba|Axial tail|179|5.515390→5.289440|5.092877→4.775140|
|6bba|Good|6,665|1.360100→2.494443|0.770686→2.090435|

Only6bba's axial-tail subgroup improves both3D and absolute z means. On44b6
the axial mean improves slightly but3D mean worsens, so it fails the declared
joint requirement. All eligible, ordinary tail and good-point means worsen in
both directions. Good-point damage is substantial: mean3D rises0.791139um and
1.134343um. Seven of the eight registered real subgroup gates fail.

Across9,294 eligible opposite-embryo pairs, mean3D error rises
**1.781070→2.781440um (+1.000371um)** and absolute z1.114106→2.284686um.
1,305 points improve,4,099 worsen and3,890 tie in3D error.5,694 proposals are
nonzero; a nonzero proposal can tie in absolute error by moving across its
fixed target. There are no decoder ties. Two of22 movies improve mean3D
(44b6_0b24845f and44b6_341df25f);20 worsen. This is not an effect restricted to
one failed movie.

Holding all7,637 excluded pairs unchanged does not change the conclusion:
all16,931 original mean3D1.780917→2.330054um, absolute z1.082821→1.725393um.
Persistent tails also worsen:44b6 n65,4.591035→5.511388um;6bba n502,
4.543071→5.204352um. Broken-link tails worsen4.973454→5.498750um on6bba
and5.222148→5.418910um on44b6. A few tail points lack either continuity flag;
they remain in ordinary-tail results rather than being dropped.

There are189 opposite-embryo proposals flagged outside their original
prediction's strict spatial ownership (81/108), and0 outside image bounds.
These are diagnostics only, as preregistered; the errors above include them.
The study gives no basis for a post-hoc guard or confidence/strength sweep.

Own-source real-anchor behavior also matters. Despite fitting the synthetic
task, the44b6 model worsens its real mean3D1.609211→1.845511um and good
1.183390→1.412745um. The6bba model worsens its overall1.805917→1.939255um
and good1.360100→1.586261um, though its own tail improves4.666698→4.389709um
and axial tail5.515390→4.151440um. Synthetic training versus real detector
anchoring is therefore another observable gap, alongside embryo transfer.

## What changes after C058/C060

1. **Axial image information is not completely absent.** Direct-known-point
   supervision with balanced shifts and a whole-crop head learns nontrivial
   source axial localization, including large shifts. This contrasts with
   C060's very weak large-offset fitting. It does not establish biological
   correctness or a useful graph correction.
2. **Removing prediction-to-GT residual labels did not remove transfer bias.**
   This experiment trains directly from known GEFF points, yet has pronounced
   opposite-embryo bias even at exact annotated x/y and zero artificial shift.
   Incorrect prediction assignment is therefore not required for this new
   failure to occur. This is not a controlled ablation identifying one sole
   cause; architecture, sampling and task also differ from C058/C060.
3. **Large artificial-offset recovery is an inadequate acceptance metric.**
   Opposite pooled synthetic error improves, while near-center synthetic
   errors and real already-good points are damaged. Source task success and
   a favorable average across shifts must not be presented as real localization
   or leaderboard improvement.

This specific direct-axial recipe is closed. The evidence supports neither
more epochs, decoder/strength changes, guard relaxation, nor a deployment
embryo router. It also does not logically exhaust every localization method.
Any further method must add a distinct justified information source or task
and address both source-to-target and synthetic-to-real gaps before graph
integration. C023/C024 scored anchors and accepted submissions remain untouched.
