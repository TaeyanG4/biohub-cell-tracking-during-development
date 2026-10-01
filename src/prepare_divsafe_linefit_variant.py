from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "experiments" / "exp_dctta_hoct_det0965"
DST_DIR = ROOT / "experiments" / "exp_dctta_hoct_det0965_divsafe_linefit"
SRC_NB = SRC_DIR / "biohub-dctta-hoct-det0965.ipynb"
DST_NB = DST_DIR / "biohub-dctta-hoct-det0965-divsafe-linefit.ipynb"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    target_idx = None
    for idx, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if "def linefit_smooth_output_graph(" in source:
            target_idx = idx
            break
    if target_idx is None:
        raise RuntimeError("linefit_smooth_output_graph not found")

    source = "".join(nb["cells"][target_idx]["source"])

    old = '''    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))
'''

    new = '''    for node_id in sorted(nodes_by_id):
        # Division-safe smoothing: do not move branch parents, and never let a
        # daughter's backward neighbourhood cross through a branch parent into
        # the pre-division lineage. The branch parent itself may be used as the
        # daughter's immediate endpoint so the first post-division step remains
        # geometrically anchored.
        if len(successor.get(node_id, [])) >= 2:
            stats["linefit_skipped_nodes"] += 1
            continue

        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            # Once we have stepped onto a branch parent, stop before crossing
            # it. This is the key difference from the production smoother.
            if current != node_id and len(successor.get(current, [])) != 1:
                break
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))
'''

    if old not in source:
        raise RuntimeError("expected linefit traversal block not found")
    source = source.replace(old, new, 1)
    nb["cells"][target_idx]["source"] = source
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "taeyangg4/biohub-dctta-hoct-det0965-divsafe-linefit"
    meta["title"] = "Biohub DCTTA HOCT DET0965 DivSafe Linefit"
    meta["code_file"] = DST_NB.name
    (DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(DST_DIR)
    print("changed_cell", target_idx)


if __name__ == "__main__":
    main()
