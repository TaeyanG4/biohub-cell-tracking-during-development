from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


root = Path("/content/biohub-r3/inputs")
parts = sorted(root.glob("6bba_05b6850b_smoke.part_*"))
if not parts:
    raise RuntimeError("no smoke parts found")

tar_path = root / "6bba_05b6850b_smoke.tar"
with tar_path.open("wb") as out:
    for part in parts:
        out.write(part.read_bytes())

h = hashlib.sha256()
with tar_path.open("rb") as f:
    while True:
        chunk = f.read(8 * 1024 * 1024)
        if not chunk:
            break
        h.update(chunk)

print("parts", len(parts))
print("bytes", tar_path.stat().st_size)
print("sha256", h.hexdigest())

extract_root = root / "smoke"
extract_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(tar_path, "r") as tf:
    tf.extractall(extract_root)

image = extract_root / "data" / "visible_test" / "test" / "6bba_05b6850b.zarr"
gt = extract_root / "data" / "visible_gt" / "train" / "6bba_05b6850b.geff"
print("image_exists", image.exists(), image)
print("gt_exists", gt.exists(), gt)
print("SMOKE_INPUT_READY")
