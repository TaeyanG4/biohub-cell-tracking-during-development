import json
from pathlib import Path

p = Path("experiments/exp_dctta_lite_det0965_public0947/biohub-dctta-lite-det0965-public0947.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    hits = [line for line in src.splitlines() if "test_stems" in line or "TEST_DIR =" in line or "TRAIN_DIR =" in line or "_guard_expected" in line or "_expected_sets" in line]
    if hits:
        print("CELL", i)
        for line in hits[:20]:
            print(line)
        if i in (2, 4):
            print("---FULL---")
            print(src[:12000])
