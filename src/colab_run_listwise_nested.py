from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path("/content/biohub-r3")
out_dir = root / "outputs" / "r3_nested_visible4"
cmd = [
    sys.executable,
    str(root / "src" / "train_r3_listwise_nested.py"),
    "--npz", str(root / "inputs" / "r3_visible4_compact_hard8.npz"),
    "--head-contract", str(root / "inputs" / "r3_head_contract_v1.npz"),
    "--out-dir", str(out_dir),
    "--epochs", "10,20,40",
    "--lrs", "0.0001,0.0003,0.001",
    "--anchors", "0.02,0.05,0.1,0.2",
    "--inner-val-frac", "0.25",
    "--batch-targets", "2048",
    "--device", "cuda",
    "--seed", "20260914",
]
print("running", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
print("COLAB_LISTWISE_NESTED_DONE", flush=True)
