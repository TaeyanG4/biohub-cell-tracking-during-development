# C062 driver/launcher provenance review

Read-only static review, 2026-09-29. No experiment execution, queue launch,
or shared-source edits. Reviewed driver SHA256
`5e6cdc207e0354658f6dacb2e04e166b7c8917faebe7c5f33305b66fa2618138`
and launcher SHA256
`f94856f270b430166a08023d51176770a22246726ff3bea153ad24ba0122e43b`.
The probe was still being finalized; observed SHA256
`f339e67cdc00e9957f722716bc83fc74ed6a15ac6d91a0f7dcaa506b12a5d719`.

## One required phase-boundary correction

`prepare(..., phase='signal')` verifies the preflight **input** hashes but
does not verify `preflight/output_hashes.json` against the files before
hashing the current `out/probe` tree into the signal plan. A preflight output
changed after the independent review could therefore be newly blessed by
the signal plan. Add `verify(read(out/'preflight/output_hashes.json'))`
before collecting/pinning those files; that manifest already uses ROOT-relative
paths compatible with `verify`. No new hash implementation is needed.

The signal stage should also check the preflight plan SHA against
`preflight/status.json['plan_sha256']`, so the accepted completed plan is the
one being promoted. This is a provenance gate, not a numerical failure found
in the current files.

## Pinning details to close before registration

The scientific data footprint is otherwise appropriately broad: both complete
source Zarr and GEFF trees, actual cached ILP graph trees, low-detection caches,
C055 zero reference graphs/table, C058 baseline/known-label pairs, opposite
C048 checkpoint+JSON+plan/fold proof, and vendored official metric/DeepCenter
checkpoint/manifest paths are selected. No encoder fit is performed.

Add the following already-read or executed provenance files to the registration
set:

- `experiments/candidates/c058_raw_localizer/plan.json`: `dependencies()` reads
  it to derive inherited metric/DeepCenter files, but currently does not pin it.
- `state/c062_preflight/launch.ps1` and
  `tools/notify_background_completion.ps1`: the lifecycle implementation is
  currently outside the input hashes.
- For source-import completeness, `src/local_registration_probe.py` and
  `src/c055_guarded_readmit_stage.py` are imported through current dependencies
  were missing from the first inspected snapshot. A final reread confirms the
  probe agent has now added both plus Queue to its explicit dependency list;
  this item is resolved. The probe does not call their motion/guarded-readmission
  mechanisms.

The in-memory plan is used for execution, while its SHA is saved in status.
For direct after-run drift rejection, compare `sha(folder/'plan.json')` with
that original saved value again before `q.close`. Without this, independent
review will still detect plan mismatch, but the queue itself can say complete.

## Lifecycle findings

- Existing status blocks duplicate phase starts; Queue uses exclusive lock
  and owns subprocess execution; preflight and signal use separate folders.
- Job and input hashes are checked before and after all planned jobs.
  Signal adds new outputs without intentionally rewriting pinned preflight
  probe files. Output hashes cover the probe tree and per-phase job logs.
- Both owner and watcher launch hidden. Owned PID and exact start time are
  written before watcher start. Terminal/unexpected-exit notifications use the
  existing helper, and observer logs are outside the scientific hash tree.
- Current `state/background_notifications`, monitor, science review and
  duplicate-check paths exist. No shell quoting issue appears for these fixed
  workspace-relative arguments.
- The launcher only writes monitor `automation_status='ACTIVE'`; it does not
  create/update the app heartbeat. Root must schedule the actual same-thread
  heartbeat separately and verify that operation, as in prior launches.
- The monitor write follows process creation. If that write fails, the finite
  compute process and native watcher may still be running; inspect their owned
  PID/start and launch receipt before retrying. Do not blindly relaunch.

No other launch blocker was found in this bounded review. This report does
not approve the separate probe's proposed-choice science or establish runtime
success; another reviewer is handling its same-ID and proposal controls.
