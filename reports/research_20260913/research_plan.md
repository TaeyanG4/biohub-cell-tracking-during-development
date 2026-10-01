# Biohub: evidence-driven first-place research plan

> **CURRENT STATUS WARNING (2026-09-14 KST):** This is a historical research snapshot. For current Kaggle scores, closed branches, R3 results, and next actions, use HANDOFF.md at the project root. Do not infer current submission state from the PENDING/planned entries below.


Research date: 2026-09-13 (Korea). No new competition submission in this research pass.

## Executive decision

Prioritize association learning and jointly selected node/link hypotheses over
further coordinate-snap sweeps on the four visible examples. This is a research
priority, not a claim that first place or a particular leaderboard gain is likely.
The measured competition baseline remains Public 0.945. Submission 56186731 was
PENDING with no errorDescription at the authenticated API check during this pass.
The current leaderboard leader was not re-verified during this pass.

## Evidence levels

- Local measurement: executed on this workspace and saved with a receipt.
- Source finding: supported by the paper, author documentation, or repository.
- Proposed experiment: not yet an observed improvement.
- Deployment gate: licensing, provenance, validation, or runtime still unresolved.

## 1. Correct earlier conclusions before spending more compute

### GT-assisted candidate selection is not a deployable feature

`src/analyze_miss_feature_ranking.py` loops over GT points, queries BOTH the
baseline and StrongUNet nearest neighbors at the GT position, and only then
computes their disagreement. Its AUC 0.9737 and the 8/8 observations above 12 um
are oracle diagnostics. They do not support a test-time 12 um replacement rule.
The script now carries an explicit warning. Candidate generation must be blind
to GT; GT may label fixed candidates only inside training/evaluation partitions.

### Sparse annotation is not exhaustive ground truth

Unmatched predicted detections cannot all be called false positives. Likewise,
the nearest baseline node to a missed GT cell may be a different real cell, not
a misplaced instance of the same cell. A larger StrongUNet node count indicates
a count-penalty risk, not a measured false-detection count. Earlier
`precision_like` values should not be interpreted as precision.

### Four visible videos are diagnostics, not OOF

The four public examples are training copies. They contain 2,193 annotated nodes,
2,127 annotated edges and three GT division events across two embryo prefixes.
Repeated tuning has consumed their usefulness as an untouched validation set.
Whole-pipeline training exposure, including pretrained detectors, must be audited
before calling a new split embryo-disjoint OOF. Training a new head on a held-out
embryo does not make a detector previously trained on it out of fold.

## 2. Fresh official-metric error budget

Executed `src/audit_official_error_budget.py` using metric commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`. It uses the official node matches,
reconstructs recovered GT edges, and asserts that its TP and FN decomposition
equals the official counts. No prediction is edited using GT.

| Fixed prediction | Diagnostic score | Edge TP | Edge FP | Edge FN | FN: missing endpoint | FN: both endpoints present |
|---|---:|---:|---:|---:|---:|---:|
| Public 0.946 reference | 0.8936490609 | 2023 | 141 | 104 | 24 | 80 |
| DCTTA DET0.965 | 0.8931916331 | 2022 | 141 | 105 | 26 | 79 |
| DCTTA + HOCT DET0.965 | 0.8951190616 | 2021 | 135 | 106 | 26 | 80 |
| Public + TOQL, anchor4um/cap4 | 0.8945071127 | 2026 | 142 | 101 | 21 | 80 |

All four rows have division TP/FP/FN = 0/2/3. This is not a leaderboard table.
For the reference graph, 80/104 missing edges have both endpoints detected.
The dense `6bba_05db0fb1` example contributes 118 of 141 edge FPs and 82 of 104
edge FNs, so pooled improvements can conceal a single-video effect.
HOCT reduces FPs while losing some TPs; it has not demonstrated division recovery
on this diagnostic set. The TOQL bridge addition leaves the 80 existing-endpoint
association misses unchanged and recovers three TPs elsewhere.

Artifacts: `official_error_budget.json`, `official_error_budget_per_video.csv`,
and `diagnostic_gt_missing_edges.csv` in this directory. The last is explicitly
GT-derived diagnostic data and must never enter test-time candidate selection.

## 3. External work selected for implementation relevance

### HOCT: edge-centric representation and small-head adaptation

Source: Bragantini, Theodoro and Royer, Higher-Order Cell Tracking Transformer,
arXiv:2607.11754 (2026), and `royerlab/hoct`.
The model represents candidate links and uses geometric relations between links.
The public code includes frozen-backbone linear-probe adaptation from sparse
edge corrections. The paper's human-in-the-loop benchmark is not Kaggle evidence.

New official `general_v1` weights were released on 2026-08-24. We downloaded
25,496,698 bytes and checked SHA256:
`5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33`.
The checkpoint has 6,252,593 parameters. CPU load and CPU/GPU synthetic forward
tests passed on the local RTX 4070 Ti SUPER. Outputs include 288-dimensional
edge features. This is compatibility evidence, not a real-video accuracy result.

Code audited at `2ccc5040823bc944ab67790abd1f56eea7cd4f05`:
`src/hoct/_models.py`, `_api.py`, `correction.py`, and the exported JIT forward.
The model's input contract is 19 features; existing UNet 32-channel features
cannot simply be substituted. The exposed `create_graph_from_points()` currently
ends in `pass`. Use a properly prepared graph or the supported segmentation path,
not that incomplete helper. Preserve checkpoint feature ordering and standardization.

Additional executable finding: the exported edge features precede `head_norm`.
Applying copied original head weights directly to raw features is NOT identity:
our synthetic input produced maximum logit difference 226.5970; applying
`head_norm` before the same head reproduced original logits exactly. This does
not invalidate a separately fitted probe; it requires an identity regression test
when adapting the existing head. Never confuse raw features and normalized features.

Proposed experiment: same nodes, candidate edges, morphology, window and solver;
compare v0/v1, then a regularized train-only head. Start with fixed candidates to
isolate association effects. Supervise competing parents only where a real parent
is annotated; unknown cells are not automatic negatives. Benchmark tiled attention
because fewer candidate edges do not eliminate quadratic edge-attention costs.

### Ultrack: choose detection hypotheses and tracks together

Source: Nature Methods (2025), DOI 10.1038/s41592-025-02778-0; `royerlab/ultrack`.
Ultrack retains multiple segmentation hypotheses and selects compatible segments
and associations jointly with temporal evidence and an ILP.

Proposed adaptation: restrict alternative baseline/StrongUNet detections to
ambiguous neighborhoods; make overlapping hypotheses mutually exclusive and
score complete local tracks. Do not force a replacement before checking its
incoming and outgoing links, and do not union every proposal into the output.
This is an adaptation of a principle, not a reproduction of the full Ultrack system.
Measure proposal coverage first: no solver can recover a candidate never proposed.
Use an available free solver; do not assume an academic Gurobi license is available.

### Linajea: backward parent displacement from sparse points

Source: Automated reconstruction of whole-embryo cell lineages by learning from
sparse annotations, Nature Biotechnology, DOI 10.1038/s41587-022-01427-7;
`funkelab/linajea`. The method learns cell indicators and backward movement vectors
from sparse annotations and performs graph optimization. The repository warns
that current code differs from the paper's tagged versions.

Proposed head: predict parent_position(t-1) - daughter_position(t), rather than a
single forward vector that becomes ambiguous at division. Combine offset residual,
appearance and local candidate competition. Preserve two daughters sharing one
parent while forbidding multiple parents per daughter. Sparse-label masks must
also apply to displacement loss.

### Trackastra and ByoTrack: independent association evidence

Trackastra (ECCV 2024; arXiv:2405.15700; `weigertlab/trackastra`) supplies
transformer association and greedy/ILP linking from images and segmentations.
Its SAM2-feature model is explicitly a 2D variant, not a ready-made 3D solution.
ByoTrack's official documentation provides 2D/3D Kalman/optical-flow tracking,
smoothing and stitching. Treat these as independent teachers or motion evidence,
not an assumption that an external benchmark win transfers to this competition.
Any pseudo labels should be restricted to training partitions, consensus-supported
tracklets and explicit confidence masks; avoid recycling our own errors as truth.

### FOCUS-3D: possible morphology teacher, conditional access

The author repository provides volumetric segmentation and finetuning; the
Hugging Face model card lists general, membrane and nuclei weights under Apache2.
The download is gated on accepting conditions and sharing contact information.
No such terms were accepted, no private dataset was uploaded, and these weights
were not downloaded in this pass. Resolve access, competition eligibility and
training provenance before budgeting it as a required dependency. The existing
Hengck checkpoint is a separate downloaded artifact, not proof FOCUS-3D is installed.

### External division data: useful labels, mismatched priors

The author of Kaggle discussion 732103 releases a CC0 synthetic collection and
reports 165,267 mitotic parent/fork events. This is not a count of daughter edges.
Both daughters inherit the parent's track_id; reconstruct lineages from edges,
not by grouping track_id as a unique trajectory. The author reports an inflated
division prior and incomplete texture/contrast matching. Verify counts locally,
pretrain a fork ranker, then calibrate/fine-tune on real training folds.

CTC provides public real/simulated 3D training data, with per-dataset usage terms
and separate unlabeled test sets. CE, CHO and SIM+ are candidates, not already
integrated training assets. Dataset-specific sparse-annotation and evaluation
rules differ from Kaggle and must be preserved during conversion.
Virtual Embryo Zoo lists published embryo trajectories; it is not automatically
synthetic or independent of the competition. Check source acquisitions, overlap,
license and coordinate/time units before use. The SSBD fish embryo link was found,
but its direct page could not be fully retrieved during this pass.

## 4. Low-cost local code hypothesis

The current notebook defines linefit smoothing with default weight 0.8 and window
2 on each side. Its backward traversal checks that the current node has a unique
predecessor but does not explicitly stop when that predecessor forks. This is not
proof of a bug: smoothing can help noisy trajectories, but may blur daughter
separation or mask association mistakes. Compare unchanged smoothing, no smoothing,
and division-safe smoothing on identical pre-smoothing graphs. Do not tune further
using the same four diagnostic videos; use held-out training partitions.

## 5. Experiment queue and gates

| ID | Scope | One controlled change | Status |
|---|---|---|---|
| R0 | Validation/provenance | Full-train embryo groups; confirm exposure of every checkpoint | Required, not completed |
| R1 | Cheap postprocess ablation | Original vs off vs division-safe smoothing | Planned |
| R2 | HOCT model ablation | v0 vs v1 on identical complete graph-feature contract | Weights/synthetic smoke passed; real-video experiment pending |
| R3 | Association learning | Frozen HOCT representation plus regularized head, trained without held-out labels | Planned after R0/R2 |
| R4 | Detection/link selection | Local mutually exclusive baseline/StrongUNet proposals, jointly scored | Planned |
| R5 | Division and motion | Backward-parent displacement and fork ranker; synthetic pretrain, real-fold calibration | Planned |

Fold1 TOQL consensus is a bounded secondary ablation, not the main first-place
strategy. Preserve HOCT veto evidence and check whether new bridges merely
reinsert previously rejected links. Stop promoting it if gains remain confined
to one diagnostic example or disappear under genuinely held-out evaluation.

Promotion requires complete dataset coverage, input-only inference, verified
coordinates/IDs, consecutive output edges, indegree <=1, outdegree <=2, no duplicate
edges, and preserved division evidence. Report official edge/division TP/FP/FN,
node-count adjustment and per-embryo results, not node recall alone.
Test candidate proposal recall separately from scoring and solver behavior.
Calibrate thresholds only within training partitions. With only two independent
embryos, do not claim strong uncertainty estimates from treating thousands of
frames or crops as independent embryos.

Use the code-competition budget: internet disabled, <=12h GPU/CPU run and
submission.csv. Pre-download versioned dependencies/models and profile complete
pipelines on representative sparse and dense movies; do not extrapolate only from
sample count or GPU neural-network time. Preserve the current submitted version.
No manual edits to hidden-test labels, no metric exploits and no access bypasses.

## 6. Sources and reproduction pointers

1. Official metric: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
2. HOCT paper: https://arxiv.org/abs/2607.11754
3. HOCT source: https://github.com/royerlab/hoct
4. HOCT weights: https://github.com/royerlab/hoct/releases/tag/weights-v1
5. HOCT probe implementation: https://github.com/royerlab/hoct/blob/main/src/hoct/correction.py
6. Ultrack paper: https://doi.org/10.1038/s41592-025-02778-0
7. Ultrack source: https://github.com/royerlab/ultrack
8. Ultrack supplementary: https://github.com/royerlab/ultrack_supplementary
9. Linajea paper: https://doi.org/10.1038/s41587-022-01427-7
10. Linajea source: https://github.com/funkelab/linajea
11. Trackastra paper: https://arxiv.org/abs/2405.15700
12. Trackastra source: https://github.com/weigertlab/trackastra
13. ByoTrack documentation: https://byotrack.readthedocs.io/en/latest/
14. FOCUS-3D source: https://github.com/yu-lab-vt/FOCUS-3D
15. FOCUS-3D weights/access terms: https://huggingface.co/Qinghua-thu/FOCUS-3D
16. Synthetic division author discussion: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103
17. CTC training data and terms: https://celltrackingchallenge.net/datasets/
18. CTC 3D index: https://celltrackingchallenge.net/3d-datasets/
19. Virtual Embryo Zoo: https://virtual-embryo-zoo.sf.czbiohub.org/
20. ELEPHANT sparse incremental learning paper: https://doi.org/10.7554/eLife.69380
21. 3DeeCellTracker paper: https://doi.org/10.7554/eLife.59187

The last two are additional literature candidates, not integrated experiments.
Actual new execution artifacts are under `artifacts/hoct_general_v1_research/` and
this report directory. No new classifier training, full-fold OOF, v1 real-video
scoring, or improved public leaderboard result was completed in this research pass.

