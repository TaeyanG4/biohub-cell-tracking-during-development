from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "experiments" / "exp_dctta_dualhoct_det0965"
DST_DIR = ROOT / "experiments" / "exp_dctta_dualhoct_det0965_nolinefit"
SRC_NB = SRC_DIR / "biohub-dctta-dualhoct-det0965.ipynb"
DST_NB = DST_DIR / "biohub-dctta-dualhoct-det0965-nolinefit.ipynb"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    # Keep the dual-HOCT v2 cap and only disable line-fit smoothing.
    src0 = "".join(nb["cells"][0]["source"])
    if 'os.environ["BIOHUB_HOCT_MAX_VIDEO_S"] = "1800"' not in src0:
        raise RuntimeError("dual-HOCT v2 cap=1800 not found")
    marker = 'os.environ["BIOHUB_HOCT_VETO"] = "2"\n'
    if marker not in src0:
        raise RuntimeError("HOCT veto marker not found")
    if 'BIOHUB_OUTPUT_LINEFIT_SMOOTH' not in src0:
        src0 = src0.replace(marker, 'os.environ["BIOHUB_OUTPUT_LINEFIT_SMOOTH"] = "0"\n' + marker, 1)
    else:
        raise RuntimeError("unexpected explicit linefit override already present")
    nb["cells"][0]["source"] = src0

    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "taeyangg4/biohub-dctta-dualhoct-det0965-nolinefit"
    meta["title"] = "Biohub DCTTA DualHOCT DET0965 NoLinefit"
    meta["code_file"] = DST_NB.name
    (DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(DST_DIR)


if __name__ == "__main__":
    main()
