from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "experiments" / "exp_dctta_hoct_det0965_nolinefit"
DST_DIR = ROOT / "experiments" / "exp_public0947_nolinefit_hoct"
SRC_NB = SRC_DIR / "biohub-dctta-hoct-det0965-nolinefit.ipynb"
DST_NB = DST_DIR / "biohub-public0947-nolinefit-hoct.ipynb"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    src = "".join(nb["cells"][0]["source"])
    replacements = [
        (
            'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"',
            'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"',
        ),
        (
            'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
            'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
        ),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(f"expected config not found: {old}")
        src = src.replace(old, new, 1)

    # Guard against a later duplicate override reintroducing 0.25.
    if 'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD' not in src:
        raise RuntimeError("safe-div threshold env missing")
    src = src.replace(
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
        'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"',
    )

    nb["cells"][0]["source"] = src
    DST_NB.write_text(
        json.dumps(nb, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "taeyangg4/biohub-public0947-nolinefit-hoct"
    meta["title"] = "Biohub Public0947 NoLinefit HOCT"
    meta["code_file"] = DST_NB.name
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print(DST_DIR)
    print("motion_tight_6", 'BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"' in src)
    print("safe_div_020", 'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"' in src)
    print("linefit_off", 'BIOHUB_OUTPUT_LINEFIT_SMOOTH"] = "0"' in src)


if __name__ == "__main__":
    main()
