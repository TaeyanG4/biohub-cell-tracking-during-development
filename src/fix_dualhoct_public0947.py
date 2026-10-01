import json
from pathlib import Path

p = Path(r"experiments/exp_dctta_dualhoct_det0965_nolinefit_public0947/biohub-dctta-dualhoct-det0965-nolinefit.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

repls = {
    'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"': 'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"',
    'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"': 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
}

for old, new in repls.items():
    hits = 0
    for c in nb["cells"]:
        s = "".join(c.get("source", []))
        if old in s:
            c["source"] = s.replace(old, new, 1).splitlines(keepends=True)
            hits += 1
    print(old, hits)

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

for c in nb["cells"]:
    if c.get("cell_type") == "code":
        s = "".join(c.get("source", []))
        if s.strip():
            compile(s, str(p), "exec")

m = Path(r"experiments/exp_dctta_dualhoct_det0965_nolinefit_public0947/kernel-metadata.json")
meta = json.loads(m.read_text(encoding="utf-8"))
meta["id"] = "taeyangg4/biohub-dctta-dualhoct-nolinefit-public0947"
meta["title"] = "Biohub DCTTA DualHOCT NoLinefit Public0947"
meta["code_file"] = p.name
m.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("compile_ok")
