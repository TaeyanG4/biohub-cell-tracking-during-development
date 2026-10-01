#!/usr/bin/env python3
"""Build Candidate C016: C012 (x138 + our head v1) with the learned division scorer.

Starts from the C012 notebook and makes two additions:
  * cell 4: mount the scorer file from the private dataset (SHA256-pinned) and export
    BIOHUB_DIVISION_SCORER plus the acceptance thresholds;
  * cell 5: embed src/division_scorer_stage.py verbatim and wrap add_safe_divisions_postlink:
    x138's rule runs unchanged, then the scored stage adds divisions among the parents the rule
    left single-child (additive; without a scorer or dump the output equals C012's), inserted
    before write_test_submission.
The local harness runs the same cell 5, so `--env BIOHUB_DIVISION_SCORER=<file>` evaluates
exactly the shipped code.

    python src/build_c016_candidate.py --scorer <division_scorer.pt> --dataset-file division_scorer_v1.pt \
        --threshold 0.6 --reparent-threshold 0.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
C012_NB = REPO_ROOT / "experiments" / "candidates" / "c012_v1284_head" / "biohub-c012-v1284-head-v1.ipynb"
C012_META = REPO_ROOT / "experiments" / "candidates" / "c012_v1284_head" / "kernel-metadata.json"
STAGE_SRC = REPO_ROOT / "src" / "division_scorer_stage.py"
DEST_DIR = REPO_ROOT / "experiments" / "candidates" / "c016_division_scorer"
DEST_NB = DEST_DIR / "biohub-c016-division-scorer.ipynb"
KERNEL_SLUG = "biohub-c016-division-scorer"
DATASET_SLUG = "biohub-division-scorer"

HEAD_ANCHOR = "print('C012 V1284 head:', _head_found[0], _head_sha256)"
SCORER_BLOCK = """
# C016: learned division scorer (src/division_scorer_stage.py, embedded in cell 5).
_scorer_candidates = [
    Path('/kaggle/input/{dslug}/{fname}'),
    Path('/kaggle/input/datasets/taeyangg4/{dslug}/{fname}'),
]
_scorer_found = [p for p in _scorer_candidates if p.is_file()]
if not _scorer_found:
    _scorer_found = sorted(Path('/kaggle/input').rglob('{dslug}/{fname}'))
if len(_scorer_found) != 1:
    raise RuntimeError(('C016 division scorer mount mismatch', [str(p) for p in _scorer_found]))
_scorer_sha256 = _head_hashlib.sha256(_scorer_found[0].read_bytes()).hexdigest()
if _scorer_sha256 != '{sha256}':
    raise RuntimeError(('C016 division scorer checksum mismatch', _scorer_sha256))
os.environ['BIOHUB_DIVISION_SCORER'] = str(_scorer_found[0])
os.environ['BIOHUB_DIVISION_SCORE_THRESHOLD'] = '{threshold}'
os.environ['BIOHUB_DIVISION_REPARENT_THRESHOLD'] = '{reparent}'
os.environ['BIOHUB_DIVISION_REPARENT_MAX_PROB'] = '{reparent_max_prob}'
os.environ['BIOHUB_DIVISION_MAX_FRAC'] = '{max_frac}'
print('C016 division scorer:', _scorer_found[0], _scorer_sha256, 'threshold', os.environ['BIOHUB_DIVISION_SCORE_THRESHOLD'])"""

SUBMISSION_CALL = '\nwrite_test_submission("base")\n'
STAGE_WRAPPER = '''

# ----------------------------------------------------------------- C016 scored divisions
_DIVISION_SCORER = load_scorer(os.environ["BIOHUB_DIVISION_SCORER"]) if os.environ.get("BIOHUB_DIVISION_SCORER", "").strip() else None
_DIVISION_THRESHOLD = float(os.environ.get("BIOHUB_DIVISION_SCORE_THRESHOLD", "0.6"))
_DIVISION_REPARENT_THRESHOLD = float(os.environ.get("BIOHUB_DIVISION_REPARENT_THRESHOLD", "0.8"))
_DIVISION_REPARENT_MAX_PROB = float(os.environ.get("BIOHUB_DIVISION_REPARENT_MAX_PROB", "0.95"))
_DIVISION_MAX_FRAC = float(os.environ.get("BIOHUB_DIVISION_MAX_FRAC", "0.00375"))
_rule_based_safe_divisions = add_safe_divisions_postlink


def add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
    """C016 (additive): x138's rule runs unchanged first, then the scored stage adds divisions among the parents
    it left single-child. Without a scorer or a low-detection dump the output equals C012's."""
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    edges = _rule_based_safe_divisions(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                                       frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)
    rule_added = stats.get("safe_divisions_added", 0)
    cache_dir = os.environ.get("BIOHUB_CACHE_DIR", "").strip()
    dump = Path(cache_dir) / f"{dataset}.npz" if cache_dir and dataset else None
    if _DIVISION_SCORER is None or dump is None or not dump.exists():
        print(f"  [{dataset}] scored divisions unavailable (scorer={_DIVISION_SCORER is not None}, dump={dump}); rule only ({rule_added})")
        return edges
    admitted, low_by_t = load_dump(np.load(dump))

    def _dc_point(node):
        value = deepcenter_score_point(dataset, int(node["t"]), (float(node["z"]), float(node["y"]), float(node["x"])),
                                       deepcenter_bundle, frame_cache, deepcenter_cache)
        return float(value) if value is not None else 0.0

    candidates = enumerate_candidates(nodes_by_id, edges, admitted, low_by_t, _dc_point,
                                      make_intensity_fn(dataset, read_test_frame, frame_cache))
    scores = score_candidates(_DIVISION_SCORER, candidates)
    max_added = max(1, int(round(len(nodes_by_id) * _DIVISION_MAX_FRAC)))
    before = len(edges)
    edges = add_scored_divisions(nodes_by_id, edges, stats, candidates, scores, _DIVISION_THRESHOLD, _DIVISION_REPARENT_THRESHOLD, max_added,
                                 reparent_max_prob=_DIVISION_REPARENT_MAX_PROB)
    print(f"  [{dataset}] scored divisions: candidates={len(candidates)} rule={rule_added} "
          f"scored={stats.get('safe_divisions_added', 0) - rule_added} reparented={stats.get('scored_division_reparented', 0)} "
          f"edges {before}->{len(edges)}", flush=True)
    return edges

'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


FUTURE_IMPORT = "from __future__ import annotations\n"


def embedded_stage_source() -> str:
    """The stage module as embedded in cell 5: identical except for the __future__ import, which is a
    SyntaxError anywhere but the top of a cell (ast.parse does not catch it; compile() does)."""
    src = STAGE_SRC.read_text(encoding="utf-8")
    if src.count(FUTURE_IMPORT) != 1:
        raise SystemExit("stage module: expected exactly one __future__ import line")
    return src.replace(FUTURE_IMPORT, "", 1)


def compile_cells(cells) -> None:
    for i, cell in enumerate(cells):
        compile("".join(cell["source"]), f"cell{i}", "exec")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--dataset-file", default="division_scorer_v1.pt")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--reparent-threshold", type=float, default=0.8)
    parser.add_argument("--reparent-max-prob", type=float, default=0.95, help="never re-parent a D2 whose ILP edge probability is >= this")
    parser.add_argument("--max-frac", type=float, default=0.00375)
    args = parser.parse_args()
    sha256 = hashlib.sha256(args.scorer.read_bytes()).hexdigest()

    nb = json.loads(C012_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cell0 = "".join(cells[0]["source"])
    cell0 = replace_once(cell0, "'''Biohub C012: x138 pipeline + own V1284-compatible head (v1)",
                         "'''Biohub C016: x138 pipeline + own V1284 head v1 + learned division scorer", "cell0 header")
    cell0 = replace_once(cell0, "BIOHUB_PRESET = 'c012_v1284_head'", "BIOHUB_PRESET = 'c016_division_scorer'", "cell0 preset")
    cell0 = replace_once(cell0, "BIOHUB_SCORE_AXIS = 'x138 pipeline + own V1284-compatible head (v1)'",
                         "BIOHUB_SCORE_AXIS = 'C012 + learned division scorer'", "cell0 axis")
    block = (SCORER_BLOCK.replace("{dslug}", DATASET_SLUG).replace("{fname}", args.dataset_file).replace("{sha256}", sha256)
             .replace("{threshold}", str(args.threshold)).replace("{reparent}", str(args.reparent_threshold))
             .replace("{reparent_max_prob}", str(args.reparent_max_prob)).replace("{max_frac}", str(args.max_frac)))
    cell4 = replace_once("".join(cells[4]["source"]), HEAD_ANCHOR, HEAD_ANCHOR + block, "cell4 head anchor")
    cell5 = "".join(cells[5]["source"])
    insertion = "\n\n# ---- embedded src/division_scorer_stage.py (C016) ----\n" + embedded_stage_source() + STAGE_WRAPPER + SUBMISSION_CALL
    cell5 = replace_once(cell5, SUBMISSION_CALL, insertion, "cell5 submission call")
    cells[0]["source"] = cell0.splitlines(keepends=True)
    cells[4]["source"] = cell4.splitlines(keepends=True)
    cells[5]["source"] = cell5.splitlines(keepends=True)
    nb["metadata"]["title"] = KERNEL_SLUG
    compile_cells(cells)

    meta = json.loads(C012_META.read_text(encoding="utf-8"))
    meta.update(id=f"taeyangg4/{KERNEL_SLUG}", title=KERNEL_SLUG, code_file=DEST_NB.name)
    meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) | {f"taeyangg4/{DATASET_SLUG}"})
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DEST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST_NB.relative_to(REPO_ROOT)} (scorer {args.dataset_file} sha256 {sha256}; threshold {args.threshold}, "
          f"reparent {args.reparent_threshold}, reparent_max_prob {args.reparent_max_prob}, max_frac {args.max_frac})")
    print(f"notebook sha256 {hashlib.sha256(DEST_NB.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
