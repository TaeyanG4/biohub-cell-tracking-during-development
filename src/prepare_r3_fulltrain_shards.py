from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "experiments" / "exp_dctta_lite_det0965_public0947"
BASE_NB = BASE_DIR / "biohub-dctta-lite-det0965-public0947.ipynb"
EXTRACTOR = ROOT / "src" / "extract_hoct_r3_compact.py"
OUT_ROOT = ROOT / "experiments" / "r3_fulltrain_featuregen"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def replace_all(nb: dict, old: str, new: str) -> int:
    hits = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if old in src:
            hits += src.count(old)
            cell["source"] = src.replace(old, new).splitlines(keepends=True)
    return hits


def build(prefix: str, shard_index: int, shard_count: int) -> Path:
    if not 0 <= shard_index < shard_count:
        raise ValueError((prefix, shard_index, shard_count))
    label = f"{prefix}-s{shard_index}of{shard_count}"
    nb = copy.deepcopy(json.loads(BASE_NB.read_text(encoding="utf-8")))

    checks = [
        (
            'TEST_DIR = COMP_DIR / "test"',
            'TEST_DIR = COMP_DIR / "train"',
            1,
        ),
        (
            "test_stems = list_test_stems()",
            f'_r3_all_stems = [s for s in list_test_stems() if s.startswith("{prefix}_")]\n'
            f'test_stems = _r3_all_stems[{shard_index}::{shard_count}]\n'
            f'assert test_stems, "No train stems found for {label}"\n'
            f'print("R3 shard {label}: movies", len(test_stems), "of", len(_r3_all_stems))',
            1,
        ),
        (
            '''_guard_expected = sorted(
    path.name.removesuffix(".zarr")
    for path in TEST_DIR.iterdir()
    if path.name.endswith(".zarr")
)''',
            '_guard_expected = sorted(test_stems)',
            1,
        ),
        (
            '_expected_sets = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))',
            '_expected_sets = sorted(test_stems)',
            1,
        ),
    ]
    for old, new, want in checks:
        got = replace_all(nb, old, new)
        if got != want:
            raise RuntimeError(f"replacement mismatch for {old!r}: expected {want}, got {got}")

    install = r'''
import glob as _r3_glob
import os as _r3_os
import subprocess as _r3_subprocess
import sys as _r3_sys
from pathlib import Path as _R3Path

_r3_wheels = _r3_glob.glob('/kaggle/input/**/hoct-0.2.0-py3-none-any.whl', recursive=True)
if not _r3_wheels:
    raise FileNotFoundError('HOCT wheel dataset not attached')
_r3_wheel_dir = _r3_os.path.dirname(_r3_wheels[0])
_r3_subprocess.run([
    _r3_sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', _r3_wheel_dir,
    'hoct==0.2.0', 'spatial-graph==0.1.1', 'pooch==1.9.0'
], check=True)

_r3_ckpts = list(_R3Path('/kaggle/input').rglob('general_v1.pt'))
if len(_r3_ckpts) != 1:
    raise RuntimeError(f'Expected exactly one general_v1.pt, found {_r3_ckpts}')
R3_HOCT_CHECKPOINT = _r3_ckpts[0]
print('R3 HOCT checkpoint:', R3_HOCT_CHECKPOINT)
'''
    nb["cells"].append(code_cell(install))

    extractor_src = EXTRACTOR.read_text(encoding="utf-8")
    extractor_src = extractor_src.split("\ndef main() -> None:", 1)[0]
    nb["cells"].append(code_cell(extractor_src))

    driver = f'''
import json as _r3_json
import numpy as _r3_np
import pandas as _r3_pd
import torch as _r3_torch

R3_HARD_K = 8
R3_PREFIX = "{prefix}"
R3_SHARD_LABEL = "{label}"
R3_SHARD_INDEX = {shard_index}
R3_SHARD_COUNT = {shard_count}
_r3_submission = _r3_pd.read_csv(SUBMISSION_PATH)
_r3_model = _r3_torch.jit.load(str(R3_HOCT_CHECKPOINT), map_location="cuda").eval()
_r3_out = WORKING_DIR / f"r3_fulltrain_{{R3_SHARD_LABEL}}.npz"
_r3_stats_out = WORKING_DIR / f"r3_fulltrain_{{R3_SHARD_LABEL}}.stats.json"

_r3_buckets = {{k: [] for k in (
    "features", "labels", "groups", "edge_ids", "source_ids", "target_ids",
    "base_scores", "views", "datasets",
)}}
_r3_stats = []
for _r3_name in test_stems:
    _x, _y, _g, _e, _s, _t, _b, _v, _st = extract_one(
        _r3_name, _r3_submission, TEST_DIR, TEST_DIR, _r3_model, R3_HARD_K
    )
    _vals = (_x, _y, _g, _e, _s, _t, _b, _v, _r3_np.full(len(_y), _r3_name, dtype="U64"))
    for _key, _value in zip(_r3_buckets, _vals):
        _r3_buckets[_key].append(_value)
    _r3_stats.append(_st)

_r3_arrays = {{k: _r3_np.concatenate(v, axis=0) for k, v in _r3_buckets.items()}}
_r3_np.savez_compressed(_r3_out, **_r3_arrays)
_r3_stats_out.write_text(_r3_json.dumps({{
    "prefix": R3_PREFIX,
    "shard_label": R3_SHARD_LABEL,
    "shard_index": R3_SHARD_INDEX,
    "shard_count": R3_SHARD_COUNT,
    "hard_k": R3_HARD_K,
    "movies": _r3_stats,
    "rows": int(len(_r3_arrays["labels"])),
    "feature_shape": list(_r3_arrays["features"].shape),
}}, indent=2), encoding="utf-8")
print("R3_FULLTRAIN_SAVED", _r3_out, _r3_arrays["features"].shape)
'''
    nb["cells"].append(code_cell(driver))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if src.strip():
                compile(src, f"r3_fulltrain_{label}.ipynb", "exec")

    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"biohub-r3-fulltrain-{label}.ipynb"
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = json.loads((BASE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = f"taeyangg4/biohub-r3-fulltrain-{label}"
    meta["title"] = f"Biohub R3 Full Train {label}"
    meta["code_file"] = out_nb.name
    sources = list(meta.get("dataset_sources", []))
    for source in ["sjlee101/biohub-hoct-020-wheels", "taeyangg4/biohub-hoct-general-v1-private"]:
        if source not in sources:
            sources.append(source)
    meta["dataset_sources"] = sources
    meta["machine_shape"] = "NvidiaTeslaT4"
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


for _prefix, _count in (("44b6", 2), ("6bba", 3)):
    for _index in range(_count):
        print("prepared", build(_prefix, _index, _count))
