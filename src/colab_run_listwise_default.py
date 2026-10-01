from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path("/content/biohub-r3")
cmd = [
    sys.executable,
    str(root / "src" / "train_r3_listwise_ranker.py"),
    "--npz", str(root / "inputs" / "r3_visible4_compact_hard8.npz"),
    "--out", str(root / "outputs" / "r3_listwise_default.json"),
    "--epochs", "100",
    "--lr", "0.001",
    "--anchor", "0.05",
    "--batch-size", "512",
    "--seed", "314159",
    "--device", "cuda",
]
print("running", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
print("COLAB_LISTWISE_DEFAULT_DONE")
