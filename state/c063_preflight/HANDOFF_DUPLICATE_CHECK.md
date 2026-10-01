# C063 prior-work check

Read HANDOFF current updates and sections12/20-24, V1284 training and C037
capture entries, AGENTS current C058/C060/C061 closures, plus
state/c062_preflight/geometry/REVIEW.md and science/REVIEW.md.

C012/13 trained an independent224d center/directional-difference MLP; C023
restores the already-trained public head. C037/52 captured indexed64d node
vectors for edge learning. C058 raw single-frame and C060 raw three-frame
CNNs failed real-anchor gates. C061 directly learned synthetic axial offsets
from raw pixels and failed opposite-embryo real-anchor gates. None of these
inspected executions fits a shared candidate-location scorer using a frozen
production32-channel dense spatial cube at real C023 anchors. C063 tests that
representation, preserving the C023 public head and its temporal/TTA context.

This is a bounded checked-record conclusion, not proof that no similar public
notebook exists. Existing public-notebook research is in
state/c061_notebook_review_20260929 and is not needlessly refreshed.
