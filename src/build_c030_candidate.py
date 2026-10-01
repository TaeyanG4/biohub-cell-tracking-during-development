#!/usr/bin/env python3
"""Build Candidate C030: C023 (x138 head + every-pair stabilization + ILP-edge restore) + AdaBN test-time adaptation.

Cell 4 of C023 patches the support pack's predict script in place (`_ps.write_text(...)`). C030 appends one more
text patch after the last existing one: the three anchored replacements of `src/build_adabn_repo.py` (module
helpers, per-movie adaptation before the window loop, restore after the movie) and sets BIOHUB_ADABN=1.
Verification: the patch strings embedded in the notebook, applied to the C022 kernel's predict script, must
give exactly the text `build_adabn_repo.py` produces.

    python src/build_c030_candidate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from build_adabn_repo import PATCHES, patch_text  # noqa: E402

C023_DIR = REPO_ROOT / "experiments" / "candidates" / "c023_x138_head_stabilize_restore"
C023_NB = C023_DIR / "biohub-c023-x138-head-stabilize-restore.ipynb"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c030_adabn"
KERNEL_SLUG = "biohub-c030-x138head-adabn"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
LAST_PATCH_LINE = "_ps.write_text(_trial_source)\n"


def adabn_cell_code() -> str:
    lines = ["", "# ----------------------------------------------------------------- C030 AdaBN test-time adaptation (src/build_adabn_repo.py)",
             "_adabn_src = _ps.read_text()", "_adabn_patches = ("]
    for label, old, new in PATCHES:
        lines.append(f"    ({label!r}, {old!r}, {new!r}),")
    lines += [")", "for _adabn_label, _adabn_old, _adabn_new in _adabn_patches:",
              "    if _adabn_src.count(_adabn_old) != 1:",
              "        raise RuntimeError(('AdaBN patch anchor mismatch', _adabn_label, _adabn_src.count(_adabn_old)))",
              "    _adabn_src = _adabn_src.replace(_adabn_old, _adabn_new, 1)",
              "_ps.write_text(_adabn_src)",
              "os.environ['BIOHUB_ADABN'] = '1'",
              "os.environ['BIOHUB_ADABN_SECONDARY'] = '1'",
              "print('C030: AdaBN patch applied (BIOHUB_ADABN=1)')", ""]
    return "\n".join(lines)


def main() -> None:
    nb = json.loads(C023_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    header = next(l for l in cell0.splitlines() if l.startswith("'''Biohub C023:"))
    cell0 = cell0.replace(header, header.replace("Biohub C023:", "Biohub C030:") + " + AdaBN test-time adaptation", 1)
    assert cell0.count("BIOHUB_PRESET = 'c023_x138_head_stabilize_restore'") == 1
    cell0 = cell0.replace("BIOHUB_PRESET = 'c023_x138_head_stabilize_restore'", "BIOHUB_PRESET = 'c030_x138head_adabn'", 1)
    axis = next(l for l in cell0.splitlines() if l.startswith("BIOHUB_SCORE_AXIS = "))
    cell0 = cell0.replace(axis, "BIOHUB_SCORE_AXIS = 'C023 + AdaBN (per-movie BatchNorm statistics of both UNets)'", 1)
    cell4 = "".join(cells[4]["source"])
    if cell4.count(LAST_PATCH_LINE) != 1:
        raise SystemExit(f"cell 4: expected one '{LAST_PATCH_LINE.strip()}', found {cell4.count(LAST_PATCH_LINE)}")
    cell4 = cell4.replace(LAST_PATCH_LINE, LAST_PATCH_LINE + adabn_cell_code(), 1)
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    nb["metadata"]["title"] = KERNEL_SLUG
    for k, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{k}", "exec")
    # verification: the embedded patches reproduce build_adabn_repo's output on the C022 kernel script
    ns: dict = {}
    exec("_adabn_patches = (\n" + "\n".join(f"    ({l!r}, {o!r}, {n!r})," for l, o, n in PATCHES) + "\n)", ns)
    src = (REPO_ROOT / "tmp" / "c022_output" / "tracking_repo" / "scripts" / "predict_unet_transformer.py").read_text(encoding="utf-8")
    out = src
    for _l, o, n in ns["_adabn_patches"]:
        assert out.count(o) == 1
        out = out.replace(o, n, 1)
    assert out == patch_text(src), "embedded patches differ from build_adabn_repo.patch_text"
    meta = json.loads((C023_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{KERNEL_SLUG}", title=KERNEL_SLUG, code_file=DEST_NB.name)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DEST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST_NB.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}; embedded patches verified against build_adabn_repo")
    print(f"notebook sha256 {hashlib.sha256(DEST_NB.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
