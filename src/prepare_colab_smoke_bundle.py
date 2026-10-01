from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = "6bba_05b6850b"
BUNDLE = ROOT / "experiments" / "colab_r3_smoke"
BUNDLE.mkdir(parents=True, exist_ok=True)

submission_src = (
    ROOT
    / "experiments"
    / "exp_dctta_lite_det0965_public0947"
    / "output_api_v1"
    / "submission.csv"
)
submission_out = BUNDLE / "submission_6bba_05b6850b.csv"
archive_out = BUNDLE / "visible_6bba_05b6850b.tar"
manifest_out = BUNDLE / "manifest.json"

frame = pd.read_csv(submission_src)
sub = frame[frame["dataset"].astype(str).eq(DATASET)].copy()
if sub.empty:
    raise RuntimeError(f"no rows for {DATASET}")
sub.to_csv(submission_out, index=False)

image_dir = ROOT / "data" / "visible_test" / "test" / f"{DATASET}.zarr"
gt_dir = ROOT / "data" / "visible_gt" / "train" / f"{DATASET}.geff"
if not image_dir.is_dir() or not gt_dir.is_dir():
    raise FileNotFoundError((image_dir, gt_dir))

with tarfile.open(archive_out, mode="w") as tf:
    tf.add(image_dir, arcname=f"visible_test/test/{DATASET}.zarr")
    tf.add(gt_dir, arcname=f"visible_gt/train/{DATASET}.geff")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "dataset": DATASET,
    "archive": {
        "path": archive_out.name,
        "bytes": archive_out.stat().st_size,
        "sha256": sha256(archive_out),
    },
    "submission": {
        "path": submission_out.name,
        "bytes": submission_out.stat().st_size,
        "sha256": sha256(submission_out),
        "rows": int(len(sub)),
        "nodes": int(sub["row_type"].eq("node").sum()),
        "edges": int(sub["row_type"].eq("edge").sum()),
    },
    "hoct_checkpoint": {
        "path": "artifacts/hoct_general_v1_research/general_v1.pt",
        "sha256": "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33f",
    },
}
manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
