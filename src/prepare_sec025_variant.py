from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "exp_edge_tta_det096" / "biohub-edge-tta-det096.ipynb"
TARGET_DIR = ROOT / "experiments" / "exp_edge_tta_det096_sec025"
TARGET = TARGET_DIR / "biohub-edge-tta-det096.ipynb"
META = TARGET_DIR / "kernel-metadata.json"

old = b"os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'"
new = b"os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.25'"

payload = SOURCE.read_bytes()
count = payload.count(old)
if count != 1:
    raise RuntimeError(f"Expected exactly one secondary-edge assignment, found {count}")

TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET.write_bytes(payload.replace(old, new, 1))

# Validate notebook JSON without reserializing it, so all unrelated bytes remain identical.
json.loads(TARGET.read_text(encoding="utf-8"))

metadata = {
    "id": "taeyangg4/biohub-edge-tta-det096-sec025",
    "title": "Biohub Edge TTA DET096 SEC025",
    "code_file": TARGET.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "kernel_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

print(f"prepared: {TARGET}")
print(f"single controlled change: {old.decode()} -> {new.decode()}")
