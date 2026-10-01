# Remaining association and uncertainty methods

2026-09-29. This review does not reopen C033/C036/C048/C051/C057/C060, run a
model, change a graph, or train anything. Localization remains the first
priority. C059 temporal division is secondary and still deferred. The local
public radar was not queried because the question is a distinct mechanism in
the already inspected infrastructure; no notebook refresh or score polling
was needed.

## Recommendation

There is one reasonably bounded association follow-up: **compare identities
using short trajectory appearance prototypes rather than individual snapshots**.
First test its ranking signal using the frozen opposite-embryo C048 encoders,
with an exact same-encoder single-frame control. Do not start with another
global solver or a newly trained sequence model. A positive, consistent result
would justify reviewing integration; current evidence does not show that it
will improve the official score.

This could be explored after the next grounded detector/localization method,
or as a short independent diagnostic while that method computes. It is not a
reason to resume the closed coordinate recipes or weaken C060's guards.

## Why simple smoothing or another assignment is insufficient

The two-movie identity audit found that missed matched-GT links often have a
smooth existing predicted continuation. In tail-touching missed links,75 of92
directed endpoint views retain a unique predicted continuation;74 of75 of
those continuations have unknown GT identity, and52 already lie inside7um of
the expected GT point. An alternative predicted point is closer to the GT, so
the official matcher assigns the annotation to that other trajectory.

Among large-residual edges that remain correct, residual directions are very
persistent (median cosine0.972/0.954); missed edges have negative medians
(−0.520/−0.204). Thus a smooth path is not sufficient evidence that it is the
metric-correct identity. A biologically reasonable smoother could reinforce
the existing wrong-scored path. The official matcher is fixed: replacing it
locally with uncertainty-aware matching would change the question rather than
improve the submitted metric.

The broader97-movie audit similarly found2,153 false-edge cases whose expected
GT partner already matches another prediction;1,712 of those alternate
predictions already have an incident edge. This supports considering competing
trajectories, but provides no deployable rule for selecting the right one.
Unknown continuations remain unknown, never biological negatives.

## Concrete bounded trajectory-prototype diagnostic

1. Reuse C023's real relink candidate groups, original image centers and
   actual final predicted trajectories through the existing C038
   `AllNodeRecorder`, `appearance_vectors` and candidate protections. Protect
   forks, gaps, synthetic nodes, absent original-image centers and ambiguous
   predecessor/successor paths. These are all inference-available conditions.
2. At a proposed boundary `t→t+1`, use the actual three-node source history
   `t−2,t−1,t` and each candidate child's actual three-node future
   `t+1,t+2,t+3`. Extract the existing32-dimensional normalized C048 appearance
   embedding at each original predicted center using the model trained on the
   opposite embryo. Normalize the mean of each three-vector tracklet, then
   compare the two prototypes by cosine similarity. This is one fixed
   temporal aggregation formula, not a window/weight sweep.
3. Compare against the same checkpoint's original single-frame source/child
   cosine on precisely the same eligible groups. A repeated-identical-vector
   control must reduce exactly to the single-frame score. Keep existing C048
   geometry/proposal rules unchanged in this diagnostic; do not select a new
   margin from the evaluation set.
4. Begin with the two previously declared full movies
   `44b6_12dfb391` and `6bba_05db0fb1`, all eligible predicted groups rather than
   an annotated-source repair whitelist. Report all-candidate coverage and
   unknown-choice burden, then labelled ranking rescues/harms relative to the
   paired control. Strict known-GT groups are for diagnosis only. No graph edit
   occurs in this first probe.
5. Continue only if the additional temporal information yields a useful
   rescue/harm balance on both embryos without merely excluding hard groups.
   Otherwise close the fixed aggregation. Actual graph integration would
   reuse C038's collision-safe reciprocal/unused-target edits and the original
   evaluator, retain final node count/coordinates, and require22 exact off
   controls and actual official edge/division counts. Long biological tracks
   or better cosine margins are not substitutes for graph evidence.

This changes the evidence presented to a decision from one observation to
multiple independent time positions along each *predicted* trajectory. It does
not simply refit the old pair encoder, change its strength, or choose different
public assignment weights. It also differs from C060's three-channel absolute
coordinate posterior: no annotated absolute-center offset is predicted here.
Nevertheless the embeddings may be too similar across adjacent cells, and
existing errors can contaminate the prototypes. Those are explicit failure
modes of the proposed probe.

## Available data and measured feasibility

The immutable C048 study already has two opposite-embryo encoders trained from
8,667/24,292 known GT triplets across62/128 movies. Its actual97 effect was small
and mixed:total+0.000038835,6bba−0.000061212. That is a usable frozen feature
source and exact control, not evidence for repeating the same pairwise arm.
Known distinct annotated nuclei supply labelled negatives; unknown cells must
not be converted into negatives for a future learner.

A fresh **metadata/graph-only** inventory, reusing C060 context rules and
13×49×49 real support, completed in6.3seconds across all22 movies. It measures
symmetric node contexts, not the exact two-tracklet candidate group count:

| Context |44b6 supported centers|6bba supported centers|Known centers|Tail centers|All-known consistent labels / tails|
|---|---:|---:|---:|---:|---:|
|3frames|84,610|169,969|8,594|699|8,037 /465|
|5frames|74,746|150,777|7,917|639|7,053 /356|
|7frames|65,860|133,436|7,280|586|6,207 /293|

Five-frame support retains88.6% of three-frame support; seven frames retain
78.3%. Longer context is available, but every extra frame reduces the number
of known consistent tail examples. Five-frame contexts still include150
known-tail centers touching wrong links; seven-frame contexts include139.
Longer histories cannot be assumed to remove ambiguity. Counts include
overlapping time windows and overlapping crops of just two embryos; they are
not independent biological examples.

Relevant cost evidence: C048's entire32-job extraction/training/97 replay took
52.1minutes. C060's two full-movie8-view inference benchmarks took76.95 and
130.97seconds, and30 disposable training steps took2.43seconds. These are
different computations, **not an ETA for the new prototype probe**. The first
two-movie adapter must record its actual number of unique requested nodes,
frame reads and encoding time before any unattended larger phase. Prototype
aggregation itself is cheap and can use each encoded node once; image encoding
and any later graph replay dominate. No new evaluator, crawler, or visualizer
is needed.

## Other ideas and their present disposition

| Method | Distinct mechanism | What is still missing | Priority |
|---|---|---|---|
|Uncertainty-aware association|Integrate a calibrated spatial likelihood over candidate positions instead of comparing point centers|A trustworthy location likelihood. C060's orientation disagreement is89–95%, own-source tail fit is weak, and no uncertainty calibration passed. Do not reinterpret its failed posterior as calibrated uncertainty.|Defer until a spatial model shows calibration/transfer.|
|Latent trajectory supervision under partial labels|Marginalize over multiple possible detections consistent with a known GT path instead of fixing each frame's nearest prediction as truth|Inference-available emissions and collision-constrained training, plus proof that latent biological paths improve the fixed official matcher. This is considerably more integration than a loss tweak.|Research option, not the next deadline experiment.|
|Learned sequence/tracklet ranker|Learn relative identity across several frames using known lineage positives and known distinct negatives|The fixed prototype probe must first show information beyond the existing pair encoder. Only two embryos and decreasing stable-tail coverage make a larger model easy to overfit.|Conditional follow-up, no fit now.|
|Joint multi-frame optimizer|min-cost path/ILP chooses several time boundaries together on fixed nodes|Better discriminative costs; jointly optimizing the same weak distances or logits can make smooth wrong paths more confident.|Do not build a solver before the signal test.|
|C059 temporal division morphology|Separate predicted parent/daughter size/brightness trajectories, original48 vs temporal56 features|Only4 independent annotation windows in the two-movie feasibility probe; no classifier fit or graph evidence. ~98% candidate triples are unknown.|Secondary, remains deferred while localization is prioritized.|

Robust partial-label handling is necessary throughout, but merely masking more
ambiguous examples is not a standalone score-improvement mechanism. C060's
strict known-neighbor rule already reduces expanded tail supervision640→429,
removing many difficult identity-switch examples. A next learner should report
this coverage honestly rather than interpreting a cleaner training loss as
recovery of the missing cases.

## Explicit differences from closed studies

- **C033:** public18-weight linear score, one-step past/future displacement
  features and independent per-frame Hungarian reassignment. Both raw and
  stabilized arms lost. Longer appearance prototypes are not that one-step
  geometric score; nevertheless ordinary velocity/acceleration retuning would
  duplicate its evidence and is not proposed.
- **C036:** two-scale9×33×33 /13×49×49 pairwise translation NCC with forward/
  backward cycle checks, one annotated recovery on44b6 and none on6bba. Longer
  appearance prototypes use multiple time positions rather than redoing the
  same pairwise translation estimate.
- **C051:** moved the same NCC template/search windows to obtain real boundary
  support;484 newly trusted groups yielded zero additional annotated recovery.
  Increasing that method's coverage again is not the new mechanism.
- **C035/C038/C048:** single-frame appearance encoding, followed by restricted
  proposals/reciprocal swaps. Fixed temporal pooling is additional evidence,
  while retraining the same pair encoder or sweeping its thresholds would
  repeat the closed arm.
- **C057:** conditional parent CE enlarged competition outside a7um identity
  gate while keeping the same hard matched parent and pairwise packet. A
  latent trajectory objective would change the assignment variable itself;
  it is not a justification for another C057 loss/weight sweep.
- **C059:** division-specific morphology draft, unregistered/deferred. Do not
  describe it as completed, failed, or the primary localization experiment.

Artifacts: `track_context_inventory.py/csv/json` and this review. No new
candidate, queue, fit, graph output, score, or submission was created.
