#!/usr/bin/env python3
"""Build Candidate C018: C017 (C012 + jump-stabilized relink) + frozen-frame catch-up prior for the relink.

4.8 % of train frame pairs are byte-identical duplicates (acquisition hiccups, 6bba); the GT does not move on
them and the NEXT pair carries a double step (GT median 3.3 um vs 1.8 um). x138's relink seeds every pair
with the previous pair's flow field, which after a frozen pair is ~0, so the catch-up pair is seeded with the
wrong motion (C017 still misses 7.3 % of GT edges there vs 3.0 % on normal pairs, 14 % of all misses).

One change vs C017, all in cell 5:
  * `filter_output_graph` is wrapped to find the movie's frozen pairs (byte-identical consecutive raw frames via
    the notebook's own read_test_frame) before post-processing;
  * inside `motion_relink_edges` the seed pass of a catch-up pair (the pair right after k frozen pairs) uses
    the flow field of the last clean pair scaled by (k + 1) instead of the frozen pair's ~0 field.
Toggle: BIOHUB_AFTER_FROZEN_PRIOR (1 = on). With it off the notebook equals C017.

    python src/build_c018_candidate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C017_DIR = REPO_ROOT / "experiments" / "candidates" / "c017_jump_stabilized_relink"
C017_NB = C017_DIR / "biohub-c017-jump-stabilized-relink.ipynb"
C017_META = C017_DIR / "kernel-metadata.json"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c018_frozen_catchup_prior"
KERNEL_SLUG = "biohub-c018-frozen-catchup-prior"
DEST_NB = DEST_DIR / f"{KERNEL_SLUG}.ipynb"
SUBMISSION_CALL = '\nwrite_test_submission("base")\n'

PATCHES = [
    ("relink init",
     "    times = sorted(ids_by_t)\n    previous_flow = None\n    for t in times:\n",
     "    times = sorted(ids_by_t)\n    previous_flow = None\n    _clean_flow_matches = None  # C018: last single-interval pair\n    _frozen_run = 0  # C018: consecutive frozen pairs right before t\n    for t in times:\n"),
    ("relink seed prior",
     "            seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)\n",
     "            _seed_prior = previous_flow\n"
     "            if AFTER_FROZEN_PRIOR and _frozen_run > 0 and t not in _FROZEN_PAIR_T and _clean_flow_matches:\n"
     "                # C018: catch-up pair after k frozen pairs moves over k + 1 intervals\n"
     "                _catchup = _flow_predictor(\n"
     "                    [position_um[s] for s, g in _clean_flow_matches],\n"
     "                    [(_frozen_run + 1) * (position_um[g] - position_um[s]) for s, g in _clean_flow_matches],\n"
     "                )\n"
     "                if _catchup is not None:\n"
     "                    _seed_prior = _catchup\n"
     "                    stats[\"after_frozen_prior_pairs\"] = stats.get(\"after_frozen_prior_pairs\", 0) + 1\n"
     "            seed = assign_pass(source_ids, target_ids, seed_gate_um, _seed_prior)\n"),
    ("relink bookkeeping",
     "        if MOTION_RELINK_FLOW_MODE != \"off\":\n            previous_flow = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])\n        stats[\"motion_relink_frames\"] += 1\n",
     "        if MOTION_RELINK_FLOW_MODE != \"off\":\n            previous_flow = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])\n"
     "        if t in _FROZEN_PAIR_T:\n"
     "            _frozen_run += 1\n"
     "        else:\n"
     "            if _frozen_run == 0:\n"
     "                _clean_flow_matches = [(s, g) for s, g, _r, _m, _n, _p in frame_matches]\n"
     "            _frozen_run = 0\n"
     "        stats[\"motion_relink_frames\"] += 1\n"),
]

FROZEN_BLOCK = '''

# ----------------------------------------------------------------- C018 frozen-frame catch-up prior
AFTER_FROZEN_PRIOR = os.environ.get("BIOHUB_AFTER_FROZEN_PRIOR", "1") != "0"
_FROZEN_PAIR_T = set()  # t such that raw frames t and t + 1 of the current movie are byte-identical
_unframed_filter_output_graph = filter_output_graph


def _find_frozen_pairs(dataset, times):
    frozen, prev = set(), None
    for t in range(min(times), max(times) + 1):
        try:
            cur = read_test_frame(dataset, int(t), {})
        except Exception:
            prev = None
            continue
        if prev is not None and prev.shape == cur.shape and np.array_equal(prev, cur):
            frozen.add(int(t) - 1)
        prev = cur
    return frozen


def filter_output_graph(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
    global _FROZEN_PAIR_T
    _FROZEN_PAIR_T = set()
    if AFTER_FROZEN_PRIOR and dataset is not None and nodes_by_id:
        _FROZEN_PAIR_T = _find_frozen_pairs(dataset, {int(n["t"]) for n in nodes_by_id.values()})
        print(f"  [{dataset}] C018 frozen frame pairs: {len(_FROZEN_PAIR_T)}", flush=True)
    try:
        return _unframed_filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
    finally:
        _FROZEN_PAIR_T = set()

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
                         "'''Biohub C018: x138 pipeline + own V1284 head v1 + jump-stabilized relink + frozen-frame catch-up prior", "cell0 header")
    cell0 = replace_once(cell0, "BIOHUB_PRESET = 'c017_jump_stabilized_relink'", "BIOHUB_PRESET = 'c018_frozen_catchup_prior'", "cell0 preset")
    cell0 = replace_once(cell0, "BIOHUB_SCORE_AXIS = 'C012 + jump-stabilized motion relink'",
                         "BIOHUB_SCORE_AXIS = 'C017 + frozen-frame catch-up prior'", "cell0 axis")
    cell5 = "".join(cells[5]["source"])
    for label, old, new in PATCHES:
        cell5 = replace_once(cell5, old, new, label)
    cell5 = replace_once(cell5, SUBMISSION_CALL, FROZEN_BLOCK + SUBMISSION_CALL, "cell5 submission call")
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
