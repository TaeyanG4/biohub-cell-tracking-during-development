from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
local = np.load(root / "reports" / "research_20260914" / "r3_compact_smoke.npz", allow_pickle=True)
colab = np.load(root / "reports" / "research_20260914" / "colab_r3_compact_smoke.npz", allow_pickle=True)

la = local["base_scores"].astype(np.float64)
cb = colab["base_scores"].astype(np.float64)
fd = local["features"].astype(np.float32) - colab["features"].astype(np.float32)

print("base_mean_abs", float(np.mean(np.abs(la - cb))))
print("base_rmse", float(np.sqrt(np.mean((la - cb) ** 2))))
print("base_pearson", float(np.corrcoef(la, cb)[0, 1]))
print("feature_mean_abs", float(np.mean(np.abs(fd))))
print("feature_rmse", float(np.sqrt(np.mean(fd ** 2))))

df = pd.DataFrame({
    "dataset": local["datasets"].astype(str),
    "target_id": local["target_ids"].astype(np.int64),
    "edge_id": local["edge_ids"].astype(np.int64),
    "label": local["labels"].astype(np.int64),
    "local": la,
    "colab": cb,
})

n = same = local_correct = colab_correct = flips_fix = flips_break = 0
for (_, target), g in df.groupby(["dataset", "target_id"], sort=False):
    if g.label.sum() <= 0 or len(g) < 2:
        continue
    li = g.local.idxmax()
    ci = g.colab.idxmax()
    lrow = g.loc[li]
    crow = g.loc[ci]
    n += 1
    same += int(int(lrow.edge_id) == int(crow.edge_id))
    lc = bool(lrow.label > 0)
    cc = bool(crow.label > 0)
    local_correct += int(lc)
    colab_correct += int(cc)
    flips_fix += int((not lc) and cc)
    flips_break += int(lc and (not cc))

print("targets", n)
print("winner_agreement", same / n if n else float("nan"), "same", same)
print("local_top1", local_correct / n if n else float("nan"), local_correct)
print("colab_top1", colab_correct / n if n else float("nan"), colab_correct)
print("colab_vs_local_fixes", flips_fix, "breaks", flips_break)
