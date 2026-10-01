import json
from pathlib import Path

p = Path("experiments/exp_dctta_lite_det0965_public0947/biohub-dctta-lite-det0965-public0947.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
src = "".join(nb["cells"][6].get("source", []))
i = src.index("_guard_expected")
print(src[i:i+900])
