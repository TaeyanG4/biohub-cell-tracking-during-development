# C036: training-free local 3D image registration

User explicitly requested continuing to the next approach on 2026-09-27 after C035 finished. C035's weak augmentation improved its basic CNN but failed the fixed diagnostic gate; it is closed. This study tests actual image displacement, not another classifier or the unaligned patch-NCC control.

## Fixed method

Run `python src/local_registration_probe.py run` with global Python. CPU FFT/template matching, four CPU threads, hidden background process, four-hour maximum, exclusive lock, source and artifact hashes. Low GPU use is expected. No new dependency, model training, inference queue, graph scorer or Kaggle operation.

Reuse the image reader and global phase correlation from `src/frame_motion_audit.py`. For each annotated reachable C023 source in the inherited C034/C035 diagnostic, calculate global image translation (full z, stride2 in XY). Crop the next-frame search region around source + global translation, independent of the candidate locations, chosen target and all GT labels.

Locally register a 9x33x33 voxel nuclear patch and a 13x49x49 voxel contextual patch with `skimage.feature.match_template`, normalized valid-mode 3D cross-correlation. Search radius +/-3z and +/-12xy voxels (4.875 um per axis); refine interior peaks with a three-point quadratic. Preserve the original detector point's offset from the integer crop center. Crop sizes and displacement use the exact (1.625,0.40625,0.40625) um voxel scale. No padding at image boundaries.

Trust only agreement across both patch sizes <=1.5 um, forward/backward cycle error <=1.0 um, all NCC >=0.65, and correlation peak margin >=0.03 against peaks at least2 um away. Reject flat crops and boundary maxima. This estimates one destination before looking at candidates; it does not independently maximize similarity at every candidate.

Propose switching an existing C023 selection only when the nearest existing gated candidate is within2.5 um of the estimate, beats the second by>=0.75 um, improves the selected candidate's residual by>=1.0 um, and increases C023's original cost by<=1.0. No missing-node recovery or gate expansion.

## Evidence and continuation

- Reuse C035's original pre-stabilization coordinates (including readmitted nodes). Verify all22 inherited cache hashes/metadata and their passed official C023/exact C034 controls. Record hashes of every newly read raw image chunk. No repeat detector or unchanged graph replay is needed for this read-only image diagnostic.
- Before real-image work, synthetic checks cover signed integer translations, wrong coarse priors, gain/background changes, subvoxel translation, forward/backward transport, flat crops and image boundaries. These are numerical checks, not score evidence.
- Evaluate the existing22 movies, keeping C034 labels unchanged. Ground truth selects the conditional diagnostic population and judges retrieval only; it never enters registration or the proposal rule. Unknown detections never become training negatives. Report per-movie and per-embryo fixes/harms, trusted coverage/rejection reasons, raw global/local retrieval and trusted local ranking against C023 cost.
- **Fixed compute gate in each embryo:** at least5 net conservative recoveries vs C023, positive net in at least2 movies, fixes at least twice harms, and nonnegative trusted rank net vs cost argmin. No post-hoc parameter sweep to rescue this gate. This is a compute allocation rule, not proof of hidden-test improvement.
- On pass, review restricted degree-consistent graph integration in the existing `eval_pp_variants_local.py` harness, preserving nodes/divisions and resolving target competition. Require actual official12+10 and matched extension75 gains. A built notebook must execute the same code and pass local replay and real T4 visible4 evaluation before any authorized exact-version submission.
- On failure, honestly close this fixed local-registration study, retain evidence and stop its heartbeat. Failure does not disprove every image-motion approach. All detectors saw the two embryos, and source ranking is not an official graph score.

## Follow-up

Read compact `status.json`, not closed C032/C033/C034/C035 queues. One check while normally running, then end without waiting or unchanged notification. Do not edit hashed sources or duplicate the process. Fix genuine execution bugs with archived provenance; never silently reuse stale artifacts. Completion evidence: `analysis.json`, `RESULTS.md`, `movie_diagnostics.csv`, per-source `groups/*.csv` and `smoke.json`. Update HANDOFF, decision, ledger and followup state; disable the finite automation after closure/submission.

Standing authorization covers worthwhile pushes, actual T4 checks and exact-version submissions without renewed approval. Check current submissions/quota, avoid duplicates and blind quota400 retries, no LB polling. Preserve C023/C024 and final picks. Long compute stays in background; up to six hourly checks on the existing `biohub-t4` heartbeat.
