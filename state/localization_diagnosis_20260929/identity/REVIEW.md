# Spatial identity audit after C058

2026-09-29. Read-only, two predeclared movies: `44b6_12dfb391` and
`6bba_05db0fb1`. No model fit, candidate change, score computation, or graph
modification. The actual official matcher reproduced all **782 / 1,215** saved
C058 baseline label identities exactly. Raw probes used the earlier
`image_blob_check.py` Gaussian/local-maximum/segment algorithm, with a fixed
SHA256-first48 tail and48 control sample per movie (192 observations).

## Main finding

Large residuals are a real warning signal, but they combine at least two
different patterns: persistent offsets along a successfully linked trajectory,
and abrupt changes in which predicted trajectory is nearest the GT annotation.
The audit cannot determine the true biological identity from sparse point
annotations. It does show why an unconstrained single-frame offset target can
be misleading, and why the original75.6% cannot be read as a75.6% recoverable
coordinate-error ceiling.

| Paired known GT edges touching a >3.5um endpoint | 44b6 | 6bba |
|---|---:|---:|
| Correctly linked / missed |53 /15|86 /31|
| Residual-direction cosine, correctly linked median |0.972|0.954|
| Residual-direction cosine, missed median |−0.520|−0.204|
| GT movement, missed median |1.68um|2.03um|
| Required movement between matched predictions, missed median |6.91um|8.33um|
| Change of residual vector, missed median |5.82um|6.13um|

The analogous missed-edge residual cosine is also negative among endpoints
with smaller errors, so direction reversal itself is not unique to the tail.
The tail causes much larger spatial jumps. Across these movies,46/185
tail-touching matched-endpoint edges are absent, compared with19/1,746 edges
without a tail endpoint. This is a descriptive association, not causal proof.

## Existing predicted paths usually continue

We inspected forward and backward continuations at each missed known edge.
The two directions of an edge are correlated, not independent events.

| Tail-touching missed edges, directed endpoint views |44b6|6bba|
|---|---:|---:|
| Directed views |30|62|
| Exactly one existing predicted continuation |23|52|
| Continuation has no matched known GT identity |23|51|
| Continuation already lies within7um of expected GT |19|33|
| Continuation movement median |0.57um|1.68um|
| Movement to the other prediction assigned by the matcher median |9.20um|8.45um|
| Existing continuation → expected GT median |5.89um|6.52um|
| Matcher-selected alternative → expected GT median |3.83um|4.15um|

Thus the frozen tracker often follows a spatially smooth trajectory, while
the nearest annotation match switches to a different prediction. It remains
unknown whether the tracker or annotation match has the correct biological
identity. The unmatched continuation is **not a false-cell label**. This
pattern supports testing temporal identity as part of coordinate estimation;
it does not justify adding the long matched-endpoint edge or moving nodes
toward GT through a diagnostic whitelist.

## Raw intensity does not identify the target center

| Fixed image sample |44b6 control / tail|6bba control / tail|
|---|---:|---:|
| Samples |48 /48|48 /48|
| Same nearest smoothed bright peak |47/48 /38/48|46/48 /32/48|
| GT distance to peak, median |1.72 /4.99um|2.35 /5.85um|
| Predicted distance to peak, median |1.72 /2.33um|2.26 /3.17um|
| GT / predicted intensity, median |1.001 /0.899|1.000 /0.856|
| Segment has >15% intensity valley |0 /0|0 /0|

A simple shift toward the nearest bright peak is unsupported: tail GT points
are typically farther from the peak than the current prediction. Moreover,
nonzero image background and broad overlapping signal make a raw-intensity
valley weak evidence. The earlier half-height audit often reached its15um
measurement limit, especially in6bba. Neither measurement establishes that
the two points share a nucleus or that the intensity maximum is the annotated
center. No new visualizer was created and no manual identity labels were
assigned.

## Consequences for the next localization hypothesis

1. Preserve the distinction between **candidate-prediction ambiguity** and
   **known-GT ambiguity**. Around50% of tail GT points here have at least two
   predictions inside7um (44b6:27/54;6bba:46/94), while C058's known-GT
   ambiguity was near zero because annotations are sparse. The latter does
   not establish unambiguous training identity.
2. A useful distinct method should retain spatial alternatives and temporal
   identity information. It should not merely strengthen the C058 correction
   or chase an intensity maximum. One falsifiable direction is a spatial
   target distribution conditioned on an existing short trajectory; it must
   still use known-only supervision and opposite-embryo validation.
3. The next diagnostic should separately track persistent residuals along
   correct trajectories and abrupt switches near missed edges, plus damage
   to correct nodes. These are evaluation strata only. Runtime behavior must
   never depend on GT residual groups, a GT whitelist, or embryo identity.
4. A coordinate proposal needs original-writer/actual-official graph evidence.
   Lower paired MSE or smooth trajectories alone are insufficient. This audit
   does not prove that the proposed temporal information can recover either
   class, and does not reopen C058.

Artifacts: `evidence.json`, `nodes.csv` (1,997 rows), `edges.csv` (1,931 rows),
`image_samples.csv` (192 rows), `continuations.csv` (130 correlated directed
views), and `continuations.json`. Inputs are SHA256 recorded. The first JSON
summary emission failed on a NumPy boolean key after all raw work and CSVs
were saved; `summarize.py` recovered from those CSVs without repeating probes,
and the writer was corrected. All diagnostic work is complete; no queue remains.
