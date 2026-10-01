from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments" / "exp_edge_tta_det096" / "biohub-edge-tta-det096.ipynb"
DONOR_DIR = ROOT / "kaggle_notebooks" / "harmonic_fusion"
TARGET_DIR = ROOT / "experiments" / "exp_edge_tta_det096_secondary_feature_tta075"
TARGET = TARGET_DIR / "biohub-edge-tta-det096-secondary-feature-tta075.ipynb"
META = TARGET_DIR / "kernel-metadata.json"


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


donor_path = next(DONOR_DIR.glob("*.ipynb"))
donor = json.loads(donor_path.read_text(encoding="utf-8-sig"))
start_marker = "_secondary_tta_source = _ps.read_text()"
end_marker = 'print("secondary edge-feature TTA patch installed and enabled", flush=True)'

secondary_block = None
for cell in donor["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = source_text(cell)
    if start_marker in text and end_marker in text:
        start = text.index(start_marker)
        end = text.index(end_marker, start) + len(end_marker)
        secondary_block = text[start:end] + "\n"
        break

if secondary_block is None:
    raise RuntimeError("Could not extract secondary edge-feature TTA block from donor notebook")

parent = json.loads(PARENT.read_text(encoding="utf-8"))
anchor = "print('Edge-feature TTA patch installed and enabled')\n"
matches = []
for idx, cell in enumerate(parent["cells"]):
    if cell.get("cell_type") != "code":
        continue
    text = source_text(cell)
    if anchor in text:
        matches.append((idx, text))

if len(matches) != 1:
    raise RuntimeError(f"Expected one primary edge-TTA anchor, found {len(matches)}")

idx, text = matches[0]
if start_marker in text:
    raise RuntimeError("Parent already contains secondary edge-feature TTA patch")

text = text.replace(anchor, anchor + "\n" + secondary_block + "\n", 1)
parent["cells"][idx]["source"] = text

TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(parent, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
json.loads(TARGET.read_text(encoding="utf-8"))

metadata = {
    "id": "taeyangg4/biohub-edge-tta-det096-secondary-feature-tta075",
    "title": "Biohub Edge TTA DET096 Secondary Feature TTA075",
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

print(TARGET)
print("inserted secondary feature TTA block with weight 0.75")
