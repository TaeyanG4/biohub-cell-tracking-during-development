from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_notebooks" / "sjlee_dctta" / "biohub-lf-dctta-v020.ipynb"
OUTDIR = ROOT / "experiments" / "exp_dctta_lite_det096"
OUT = OUTDIR / "biohub-dctta-lite-det096.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match for {old!r}, found {count}")
    return text.replace(old, new, 1)


nb = json.loads(SRC.read_text(encoding="utf-8"))

# Cell 0 contains the production environment. Preserve the DCTTA-specific
# additions while restoring the verified 0.946-lineage settings plus the
# controlled DET 0.96 and tight motion radius candidate.
cell0 = "".join(nb["cells"][0].get("source", ""))
repls = [
    ('os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"', 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"'),
    ('os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"', 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"'),
    ('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"', 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"'),
]
for old, new in repls:
    if old != new:
        cell0 = replace_once(cell0, old, new)

# Disable the notebook's internal GT-based post-process sweep so it cannot
# silently rewrite the test configuration. We want a controlled experiment.
insert = (
    '\nos.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\n'
    'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n'
)
anchor = 'os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"\n'
cell0 = replace_once(cell0, anchor, anchor + insert)
nb["cells"][0]["source"] = cell0

# Cell 1 is only a configuration guard; keep it aligned with DET=0.96.
cell1 = "".join(nb["cells"][1].get("source", ""))
cell1 = replace_once(cell1, '"BIOHUB_DET_THRESHOLD": 0.965,', '"BIOHUB_DET_THRESHOLD": 0.96,')
nb["cells"][1]["source"] = cell1

# Ensure the DCTTA-specific feature paths remain enabled.
full = json.dumps(nb, ensure_ascii=False)
required = [
    'BIOHUB_EDGE_FEATURE_TTA',
    'BIOHUB_SECONDARY_EDGE_FEATURE_TTA',
    'BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT',
    'BIOHUB_DEEPCENTER_TTA',
]
for token in required:
    if token not in full:
        raise RuntimeError(f"missing DCTTA token {token}")

OUTDIR.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

meta = {
    "id": "taeyangg4/biohub-dctta-lite-det096",
    "title": "Biohub DCTTA Lite DET096",
    "code_file": OUT.name,
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
(OUTDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(OUT)
print("prepared controlled DCTTA-lite: det=0.96, deepcenter_safe_div=0.25, motion_tight=5.5, validator disabled")
