# C064 independent prelaunch review

Root completion review2026-09-29 14:07KST: all four initial blocking findings
below are resolved in probe SHA256
`0f18b5f430da17acdae34e72abedede374b1d41687da781da4b379d74021234c`.
The recorded CPU controls bind that exact source and current protocol/sample/
checkpoint/recipe. Driver prepare rejects stale bindings. Benchmark now uses
exactly the first requested frame per embryo and writes measured_frames=2.
The actual loaded model is eval/FP32 and its tensor-state digest is compared
before/after benchmark and probe. Analysis binds every saved profile back to
its hashed heatmap and raw frame, original centre and target, then recounts
3D/abs-z/signed biases, XY no-op and proposal/tie/exclusion exactly. Original
signed biases and stratum ownership/tie/no-op counts are also recorded.
No GPU result was inspected during these changes. Scientific proposal/gate
unchanged. Driver and launcher parse checks passed. The final review was
completed locally after the earlier reviewers became unavailable; initial
independent findings are retained below rather than rewritten as a second
independent completion review. Safe to register the bounded timing phase.

**Updated review: the four initial blockers below are resolved in probe
`0f18b5f430da17acdae34e72abedede374b1d41687da781da4b379d74021234c` and driver
`e48cb495b3db14eddec55066d587062663e219d935f03d6a59df6c0e605d21a1`.**
Independent CPU control passed (`independent_controls.py/json`):273 exact
original-score/profile comparisons, all9 signed axial displacements,
tie/boundary/nonfinite/no-field no-ops, exact256 original identities and all
256 real-support checks. The deterministic benchmark frames are
44b6_0113de3b:t42 and6bba_05b6850b:t9. No GPU/model construction/inference or
training ran in this independent review.

No remaining prelaunch blocker was found. Actual CUDA initialization, frozen
model-state and real original-score/profile parity are deliberately validated
by the two-frame timing phase. Do not count CPU controls as already proving
those runtime facts. Root should pin the exact current source/control hashes
before that phase and independently review its real result before signal.

Corrections verified: exact2-frame schema, controls/current-source/recipe
bindings in probe and driver;eval() and frozen-state before/after; analysis
re-reads every saved heatmap and provenance, reconstructs the profile, verifies
original targets and unchanged XY, and recounts3D/abs-z/signed errors. Signed
before biases and per-stratum ownership/tie/no-op counts now appear directly.
Initial findings are retained below as review history only.

Initial source review2026-09-29. Reviewed root queue driver/PowerShell launcher,
probe source and source-selected protocol. No reviewed source was edited and
no GPU/model inference ran in this review. Findings below refer to initial
probe SHA256`b352f0c1f33c9c4bebac46d8d9ce394186a1af029552b5b75b4bafda9eebde76`;
authors should resolve them before pinning.

## Blocking interface/control findings

1. Probe `RECIPE.benchmark_frames=8` and `allframes[:8]` still implement the
   superseded benchmark and would sample44b6 only. The fixed protocol and
   driver require exactly two frames, lexicographically first per embryo.
   `benchmark.json` also lacks `measured_frames`, which signal preparation
   directly reads and asserts equals2. Correct these before any real benchmark.
2. `controls.json` currently binds sample/protocol/checkpoint, but not its
   executed probe-source hash. Driver only checks`status=passed`; it could
   consume stale controls after the still-unregistered probe changes. Record
   exact probe source+recipe in controls and compare source/sample/protocol/
   checkpoint/recipe to current inputs in driver prepare.
3. Protocol requires frozen-model proof, while probe only checks FP32 at
   construction. Record one state-dictionary digest immediately after load
   and after each benchmark/probe phase, require `eval()`/same state, and save
   the before/after proof. This is a control gap, not evidence of actual drift.
4. `analyse` rechecks IDs but does not independently compare saved target
   coordinates to original sample residuals, does not recount abs-z/bias/x-y
   no-op fields, and does not bind saved profiles back to each pinned heatmap.
   Add these checks or provide an independent post-probe verifier before
   accepting the gate. Profiles should match the exact heatmap slice and
   per-frame provenance; no recomputation of GPU inference is needed.

## Driver and launcher assessment

The finite driver reuses existing Queue, checks inputs before and after,
pins plan SHA in status, captures job return codes, hashes phase outputs/logs,
and refuses a phase with existing status. Signal preparation verifies the
prior timing plan/status/output hashes and reuses the two saved fields.
Runtime estimator conservatively uses measured frame time,1.5multiplier and
120seconds allowance; max_hours is explicitly a start-job bound, consistent
with the existing Queue behavior. Failure remains recorded for review.

The launcher runs owner and notifier hidden, records exact PID creation time,
separate native watcher logs/receipt, updates current monitor and avoids an
active agent wait. ETA-10/20-minute schedule metadata and short-phase fallback
are explicit. It relies on the existing state/background_notifications
directory, which currently exists. Root still owns the separate actual
heartbeat update; writing monitor metadata alone is not that scheduling call.

The driver output sets are coherent: timing hashes benchmark plus heatmaps;
signal hashes all heatmaps, saved profiles and analysis. No queue method
uploads or submits. The scientific proposal and fixed gates match protocol
apart from the benchmark disagreement listed above. Boundary/tie/no-field
no-ops, exact±6.5um integer-z selection, unchanged XY and ownership-as-diagnostic
are implemented directly. No GT mask selects proposals.

## Nonblocking reporting observations

Source/validation-selected interpretations are explicit in per-pair output
and decision. Add original signed biases beside after biases and per-stratum
ownership/tie counts if desired for direct readability; saved target/shift/
profile rows already contain the needed values. This must not change the gate.
The external prototype dimensions and strong shift assertions are appropriate;
rounded output displacement is exactly representable, avoiding C063's FP32
CSV assertion mismatch. `%17g` plus round_trip read supports exact recount.
