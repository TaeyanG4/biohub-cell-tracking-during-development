from __future__ import annotations

import difflib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "exp_dctta_hoct_det0965"
DST = ROOT / "experiments" / "exp_dctta_hoct_det0965_nolinefit"
SRC_NB = SRC / "biohub-dctta-hoct-det0965.ipynb"
DST_NB = DST / "biohub-dctta-hoct-det0965-nolinefit.ipynb"


def source_text(nb: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


DST.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

old = 'os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1")'
new = 'os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "0")'
replacements = 0
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if old in src:
        cell["source"] = src.replace(old, new).splitlines(keepends=True)
        replacements += 1

if replacements != 1:
    raise RuntimeError(f"Expected exactly one linefit replacement, got {replacements}")

DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

metadata = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
metadata.update(
    {
        "id": "taeyangg4/biohub-dctta-hoct-det0965-nolinefit",
        "title": "Biohub DCTTA HOCT DET0965 NoLinefit",
        "code_file": DST_NB.name,
    }
)
(DST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

src_nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
dst_nb = json.loads(DST_NB.read_text(encoding="utf-8"))
diff = list(
    difflib.unified_diff(
        source_text(src_nb).splitlines(),
        source_text(dst_nb).splitlines(),
        fromfile="baseline",
        tofile="nolinefit",
        n=1,
    )
)
print("REPLACEMENTS", replacements)
print("DIFF_LINES", len(diff))
print("\n".join(diff[:40]))
