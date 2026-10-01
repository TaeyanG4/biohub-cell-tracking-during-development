# Localization priority: what the 75.6% finding supports

2026-09-29. The user requested that localization remain the main research
priority after C058 failed. C059 temporal-division work is deferred.

The original count is **1,047 of 1,385 missed GT links whose endpoint detections
both exist**. At least one officially matched endpoint has a residual above
3.5 micrometres. This is not the fraction of all detections that are wrong,
and is not a measured recoverable-score ceiling.

Three completed diagnostics now constrain the next method:

* Saved C058 checkpoints fit their source training labels but shift the other
  embryo in incorrect directions. Both embryos' large-error z corrections have
  the wrong mean sign. Rounding and crop precision do not explain the loss.
* Among 1,276 large-residual nodes in all22 movies, 936 have every known adjacent
  link correct, 320 touch a missed link, and20 lack a matched known neighbour.
  Persistent offset on an intact path and a changing matched identity are
  different phenomena; a single displacement-regression target conflates them.
* Two-movie raw-image probes give no support for snapping to an intensity peak.
  Existing smooth paths often continue near a missed annotation, while the
  official per-frame match switches to another predicted path. Sparse GT cannot
  establish which predicted path is biologically correct.

Next bounded design: condition a spatial coordinate distribution on raw images
from an actual short predicted track. Preserve physical geometry and test
reflection/translation conventions directly. Any protection against crossing
to another predicted identity must use predicted geometry only. Missing track
context, image boundaries and unknown cells cannot be filled with fabricated
labels. Whole-embryo component validation remains mandatory.

This is a new hypothesis, not an established improvement. Before training,
measure real triple-context/tail coverage and the effect of identity-preserving
geometry on reachable targets, then validate actual crop transforms, gradients,
zero-output parity and runtime cost. No recipe, model or queue is registered yet.

Fixed final topology is useful for checking damage and rematching, but cannot
recover an absent edge if GT matches remain identical. Strong transferable
coordinate evidence could justify a separately controlled pre-association test
even if frozen-edge scores tie. Negative paired localization cannot justify it.

Evidence: `model/REVIEW.md`, `identity/REVIEW.md`, `continuity.json`,
`tail_continuity_splits.json`. GT-derived diagnostic strata are evaluation only,
never runtime masks. C023/C024 scored anchors remain unchanged.
