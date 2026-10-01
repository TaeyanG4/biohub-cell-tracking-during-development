#!/usr/bin/env python3
"""Build Candidate C019: C017 (C012 + jump-stabilized relink) + ILP-seeded relink flow.

x138's relink seeds each frame pair with the previous pair's flow field (`previous_flow`), which is wrong
whenever the motion changes between pairs (jumps, drift changes, the double step after a frozen frame). The
ILP's own confident links of the SAME pair already encode that pair's motion (the raw ILP misses 11 % of GT
edges on 5-8 um jumps where C012's relink missed 20 %). C019 builds the seed prior of pair t from the ILP links
of pair t (learned_edge_probs with p >= ILP_SEED_MIN_PROB, the same k-NN median field x138 uses) and falls back
to the previous pair's field where the ILP gives fewer than MOTION_RELINK_FLOW_MIN_SAMPLES links.
One change vs C017 (cell 5, inside motion_relink_edges). Toggle BIOHUB_ILP_SEED_FLOW (1 = on); off equals C017.

    python src/build_c019_candidate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C017_DIR = REPO_ROOT / "experiments" / "candidates" / "c017_jump_stabilized_relink"
C017_NB = C017_DIR / "biohub-c017-jump-stabilized-relink.ipynb"
C017_META = C017_DIR / "kernel-metadata.json"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c019_ilp_seeded_flow"
KERNEL_SLUG = "biohub-c019-ilp-seeded-flow"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
SUBMISSION_CALL = '\nwrite_test_submission("base")\n'

PATCHES = [
    ("ilp links by frame",
     "    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}\n",
     "    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}\n"
     "    _ilp_links_by_t: dict[int, list[tuple[int, int]]] = {}  # C019: the ILP's own confident links per frame pair\n"
     "    if ILP_SEED_FLOW:\n"
     "        for (_s, _g), _p in learned_edge_probs.items():\n"
     "            if _s in position_um and _g in position_um and float(_p) >= ILP_SEED_MIN_PROB \\\n"
     "                    and int(nodes_by_id[_g][\"t\"]) == int(nodes_by_id[_s][\"t\"]) + 1:\n"
     "                _ilp_links_by_t.setdefault(int(nodes_by_id[_s][\"t\"]), []).append((_s, _g))\n"),
    ("seed prior",
     "            seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)\n",
     "            _seed_prior = previous_flow\n"
     "            if ILP_SEED_FLOW and _ilp_links_by_t.get(t):\n"
     "                _ilp_field = _field_from(_ilp_links_by_t[t])\n"
     "                if _ilp_field is not None:\n"
     "                    _seed_prior = _ilp_field\n"
     "                    stats[\"ilp_seeded_pairs\"] = stats.get(\"ilp_seeded_pairs\", 0) + 1\n"
     "            seed = assign_pass(source_ids, target_ids, seed_gate_um, _seed_prior)\n"),
]

SETTINGS_BLOCK = '''

# ----------------------------------------------------------------- C019 ILP-seeded relink flow (read at call time)
ILP_SEED_FLOW = os.environ.get("BIOHUB_ILP_SEED_FLOW", "1") != "0"
ILP_SEED_MIN_PROB = float(os.environ.get("BIOHUB_ILP_SEED_MIN_PROB", "0.5"))
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    nb = json.loads(C017_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    cell0 = replace_once(cell0, "'''Biohub C017: x138 pipeline + own V1284 head v1 + jump-stabilized motion relink",
                         "'''Biohub C019: x138 pipeline + own V1284 head v1 + jump-stabilized relink + ILP-seeded relink flow", "cell0 header")
    cell0 = replace_once(cell0, "BIOHUB_PRESET = 'c017_jump_stabilized_relink'", "BIOHUB_PRESET = 'c019_ilp_seeded_flow'", "cell0 preset")
    cell0 = replace_once(cell0, "BIOHUB_SCORE_AXIS = 'C012 + jump-stabilized motion relink'",
                         "BIOHUB_SCORE_AXIS = 'C017 + ILP-seeded relink flow'", "cell0 axis")
    cell5 = "".join(cells[5]["source"])
    for label, old, new in PATCHES:
        cell5 = replace_once(cell5, old, new, label)
    cell5 = replace_once(cell5, SUBMISSION_CALL, SETTINGS_BLOCK + SUBMISSION_CALL, "cell5 submission call")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[5]["source"] = cell5.splitlines(keepends=True)
    nb["metadata"]["title"] = KERNEL_SLUG
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    meta = json.loads(C017_META.read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{KERNEL_SLUG}", title=KERNEL_SLUG, code_file=DEST_NB.name)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DEST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST_NB.relative_to(REPO_ROOT)}; datasets {meta['dataset_sources']}")
    print(f"notebook sha256 {hashlib.sha256(DEST_NB.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
