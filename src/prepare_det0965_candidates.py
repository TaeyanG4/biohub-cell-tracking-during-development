from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def as_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match for {old!r}, found {count}")
    return text.replace(old, new, 1)


def common_edits(nb: dict) -> dict:
    cell0 = as_text(nb["cells"][0])
    # Keep the verified 0.946 detector threshold.  Only the DeepCenter safe-div
    # gate and motion-tight ablation are changed relative to the public source.
    if 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"' not in cell0:
        raise RuntimeError("source DET threshold is not 0.965")
    cell0 = replace_once(
        cell0,
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
    )
    anchor = 'os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"\n'
    insert = (
        'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\n'
        'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n'
    )
    cell0 = replace_once(cell0, anchor, anchor + insert)
    nb["cells"][0]["source"] = cell0

    cell1 = as_text(nb["cells"][1])
    if '"BIOHUB_DET_THRESHOLD": 0.965,' not in cell1:
        raise RuntimeError("configuration guard is not DET=0.965")
    nb["cells"][1]["source"] = cell1
    return nb


def write_kernel(
    source: Path,
    outdir: Path,
    notebook_name: str,
    kernel_id: str,
    title: str,
    datasets: list[str],
    hoct: bool,
) -> None:
    nb = json.loads(source.read_text(encoding="utf-8"))
    nb = common_edits(nb)

    if hoct:
        cell4 = as_text(nb["cells"][4])
        cell4 = replace_once(
            cell4,
            'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "1.0"',
            'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"',
        )
        nb["cells"][4]["source"] = cell4
        if 'BIOHUB_HOCT_VETO' not in json.dumps(nb, ensure_ascii=False):
            raise RuntimeError("HOCT source lost veto code")

    full = json.dumps(nb, ensure_ascii=False)
    for token in (
        "BIOHUB_EDGE_FEATURE_TTA",
        "BIOHUB_SECONDARY_EDGE_FEATURE_TTA",
        "BIOHUB_DEEPCENTER_TTA",
        "BIOHUB_MOTION_RELINK_TIGHT_UM",
    ):
        if token not in full:
            raise RuntimeError(f"missing required token {token}")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / notebook_name
    out.write_text(
        json.dumps(nb, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (outdir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(out)


base_datasets = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]

write_kernel(
    ROOT / "kaggle_notebooks" / "sjlee_dctta" / "biohub-lf-dctta-v020.ipynb",
    ROOT / "experiments" / "exp_dctta_lite_det0965",
    "biohub-dctta-lite-det0965.ipynb",
    "taeyangg4/biohub-dctta-lite-det0965",
    "Biohub DCTTA Lite DET0965",
    base_datasets,
    False,
)

write_kernel(
    ROOT / "kaggle_notebooks" / "sjlee_hoctveto" / "biohub-lf-hoctveto-div-b.ipynb",
    ROOT / "experiments" / "exp_dctta_hoct_det0965",
    "biohub-dctta-hoct-det0965.ipynb",
    "taeyangg4/biohub-dctta-hoct-det0965",
    "Biohub DCTTA HOCT DET0965",
    [
        *base_datasets,
        "musculer/biohub-hoct-general-v0-official",
        "sjlee101/biohub-hoct-020-wheels",
    ],
    True,
)

print("prepared DET=0.965 controlled candidates")
