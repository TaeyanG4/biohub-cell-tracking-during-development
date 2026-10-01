import json
from pathlib import Path


path = Path("experiments/exp_dctta_dualhoct_det0965/biohub-dctta-dualhoct-det0965.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
cell = "".join(nb["cells"][0]["source"])
needle = 'os.environ["BIOHUB_HOCT_VETO"] = "2"'
insert = needle + '\nos.environ["BIOHUB_HOCT_MAX_VIDEO_S"] = "1800"'
if 'BIOHUB_HOCT_MAX_VIDEO_S' not in cell:
    if needle not in cell:
        raise RuntimeError("HOCT veto setting not found")
    cell = cell.replace(needle, insert, 1)
nb["cells"][0]["source"] = cell.splitlines(keepends=True)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print([line for line in cell.splitlines() if "BIOHUB_HOCT_MAX_VIDEO_S" in line])
