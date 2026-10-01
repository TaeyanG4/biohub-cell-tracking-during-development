from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path("/content/biohub-r3")
cmd = [
    sys.executable,
    str(root / "src" / "train_geometry_swap_ranker.py"),
    "--npz", str(root / "inputs" / "fulltrain_geometry_swap.npz"),
    "--out", str(root / "outputs" / "fulltrain_geometry_swap_report.json"),
    "--folds", "5",
    "--device", "cuda",
    "--seed", "20260914",
]
print("running", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
print("COLAB_GEOMETRY_SWAP_DONE", flush=True)
