from __future__ import annotations

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
local_path = root / "reports" / "research_20260914" / "r3_compact_smoke.npz"
colab_path = root / "reports" / "research_20260914" / "colab_r3_compact_smoke.npz"

a = np.load(local_path, allow_pickle=True)
b = np.load(colab_path, allow_pickle=True)

print("local", local_path)
print("colab", colab_path)
print("keys_equal", set(a.files) == set(b.files), sorted(a.files))

for key in sorted(set(a.files) | set(b.files)):
    if key not in a or key not in b:
        print(key, "missing")
        continue
    x = a[key]
    y = b[key]
    same_shape = x.shape == y.shape
    if not same_shape:
        print(key, "shape", x.shape, y.shape)
        continue
    if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
        delta = np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))) if x.size else 0.0
        exact = np.array_equal(x, y)
        print(key, "shape", x.shape, "dtype", x.dtype, y.dtype, "exact", exact, "max_abs", float(delta))
    else:
        print(key, "shape", x.shape, "exact", np.array_equal(x.astype(str), y.astype(str)))

stats_path = root / "reports" / "research_20260914" / "colab_r3_compact_smoke.stats.json"
print("colab_stats", json.dumps(json.loads(stats_path.read_text(encoding="utf-8")), sort_keys=True))
