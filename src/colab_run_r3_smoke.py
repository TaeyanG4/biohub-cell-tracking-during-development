from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path("/content/biohub-r3")
cmd = [
    sys.executable,
    str(root / "src" / "run_hoct_r3_compact.py"),
    "--submission", str(root / "inputs" / "submission.csv"),
    "--image-root", str(root / "inputs" / "smoke" / "data" / "visible_test" / "test"),
    "--gt-root", str(root / "inputs" / "smoke" / "data" / "visible_gt" / "train"),
    "--checkpoint", str(root / "models" / "general_v1.pt"),
    "--datasets", "6bba_05b6850b",
    "--out", str(root / "outputs" / "r3_compact_smoke.npz"),
    "--hard-k", "8",
    "--device", "cuda",
]
print("running", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
print("COLAB_R3_SMOKE_DONE")
