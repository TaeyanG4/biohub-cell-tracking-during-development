import json
import sys
from pathlib import Path
from audit_fulltrain_safe_division_gt import simulate
from sweep_fulltrain_safe_division import aggregate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

gt_paths = sorted(Path("data/full_train_gt/train").glob("*.geff"))
print(f"Found {len(gt_paths)} full train GT graphs.")

# Let's test configurations:
# 1. B0 baseline (parent=9, sister=14, existing=10, sym=0.6, diverge=2.25)
# 2. C004 base (parent=9, sister=16, existing=12, sym=0.85, diverge=0.5)
# 3. div_envelope_wide (parent=11, sister=16, existing=12, sym=1.0, diverge=0.5)
# 4. div_zero_diverge (parent=9, sister=16, existing=12, sym=0.85, diverge=0.0)
# 5. div_base_strict (parent=9, sister=14, existing=10, sym=0.6, diverge=2.25)

configs = {
    "b0_exact": {
        "parent_max": 9.0,
        "sister_max": 14.0,
        "existing_max": 10.0,
        "symmetry_tau": 0.6,
        "diverge_um": 2.25,
    },
    "c004_base": {
        "parent_max": 9.0,
        "sister_max": 16.0,
        "existing_max": 12.0,
        "symmetry_tau": 0.85,
        "diverge_um": 0.5,
    },
    "div_envelope_wide": {
        "parent_max": 11.0,
        "sister_max": 16.0,
        "existing_max": 12.0,
        "symmetry_tau": 1.0,
        "diverge_um": 0.5,
    },
    "div_zero_diverge": {
        "parent_max": 9.0,
        "sister_max": 16.0,
        "existing_max": 12.0,
        "symmetry_tau": 0.85,
        "diverge_um": 0.0,
    },
    "div_conservative": {
        "parent_max": 10.0,
        "sister_max": 14.0,
        "existing_max": 10.0,
        "symmetry_tau": 0.8,
        "diverge_um": 1.0,
    },
}

# Run on all movies
print(f"Running simulation on {len(gt_paths)} GT movies...")

for name, cfg in configs.items():
    rows = []
    for p in gt_paths:
        for mode in ("nearest", "farther"):
            rows.append(simulate(p, existing_mode=mode, **cfg))
    agg = aggregate(rows)["robust"]
    print(f"[{name:18s}] mean_jaccard={agg['mean_jaccard']:.4f} | TP={agg['tp']:3d} | FP={agg['fp']:3d} | FN={agg['fn']:3d} | precision={agg['min_precision']:.3f} | recall={agg['min_recall']:.3f}")
