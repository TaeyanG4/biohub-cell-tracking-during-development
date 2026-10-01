from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "exp_dctta_lite_det0965"
DST = ROOT / "experiments" / "exp_dctta_lite_det0965_public0947"
SRC_NB = SRC / "biohub-dctta-lite-det0965.ipynb"
DST_NB = DST / "biohub-dctta-lite-det0965-public0947.ipynb"


DST.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

changes = {
    'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"':
        'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"',
    'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"':
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
}

for old, new in changes.items():
    hits = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if old in src:
            cell["source"] = src.replace(old, new, 1).splitlines(keepends=True)
            hits += 1
    if hits != 1:
        raise RuntimeError(f"expected one hit for {old!r}, got {hits}")

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if src.strip():
            compile(src, str(DST_NB), "exec")

DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
meta["id"] = "taeyangg4/biohub-dctta-lite-public0947-anchor"
meta["title"] = "Biohub DCTTA Lite Public0947 Anchor"
meta["code_file"] = DST_NB.name
(DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("prepared", DST)
print("motion_tight=6.0 safe_div_threshold=0.20")
