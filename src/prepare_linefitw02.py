import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "experiments" / "exp_dctta_hoct_det0965"
dst = root / "experiments" / "exp_dctta_hoct_det0965_linefitw02"
dst.mkdir(parents=True, exist_ok=True)

src_nb = src / "biohub-dctta-hoct-det0965.ipynb"
dst_nb = dst / "biohub-dctta-hoct-det0965-linefitw02.ipynb"
nb = json.loads(src_nb.read_text(encoding="utf-8"))
text = "".join(nb["cells"][0]["source"])
anchor = 'os.environ["BIOHUB_HOCT_VETO"] = "2"'
if anchor not in text:
    raise RuntimeError("config anchor missing")
text = text.replace(
    anchor,
    'os.environ["BIOHUB_OUTPUT_LINEFIT_WEIGHT"] = "0.2"\n' + anchor,
    1,
)
nb["cells"][0]["source"] = text
dst_nb.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
meta["id"] = "taeyangg4/biohub-dctta-hoct-det0965-linefitw02"
meta["title"] = "Biohub DCTTA HOCT DET0965 LinefitW02"
meta["code_file"] = dst_nb.name
(dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(dst)
