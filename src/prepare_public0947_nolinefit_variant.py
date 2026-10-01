from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "exp_dctta_hoct_det0965_nolinefit"
DST = ROOT / "experiments" / "exp_dctta_hoct_det0965_nolinefit_public0947"
SRC_NB = SRC / "biohub-dctta-hoct-det0965-nolinefit.ipynb"
DST_NB = DST / "biohub-dctta-hoct-det0965-nolinefit-public0947.ipynb"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return text.replace(old, new, 1)


DST.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
if 'os.environ["BIOHUB_OUTPUT_LINEFIT_SMOOTH"] = "0"' not in joined and \
   'os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "0")' not in joined:
    raise RuntimeError("source notebook is not the verified no-linefit variant")

changes = {
    'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"':
        'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"',
    'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"':
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
}

for old, new in changes.items():
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if old in src:
            cell["source"] = src.replace(old, new, 1).splitlines(keepends=True)
            hits += 1
    if hits != 1:
        raise RuntimeError(f"expected one notebook cell hit for {old!r}, got {hits}")

DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
meta["id"] = "taeyangg4/biohub-dctta-hoct-det0965-nolinefit-public0947"
meta["title"] = "Biohub DCTTA HOCT NoLinefit Public0947 Settings"
meta["code_file"] = DST_NB.name
(DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

compiled = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    compile(src, str(DST_NB), "exec")
    compiled += 1

print("prepared", DST)
print("compiled_code_cells", compiled)
print("motion_tight", "6.0")
print("safe_div_threshold", "0.20")
print("linefit", "off")
