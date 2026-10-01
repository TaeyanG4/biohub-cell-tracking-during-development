#!/usr/bin/env python3
"""Verify Candidate C016 (C012 + learned division scorer) before any push.

Checks:
  1. every code cell parses; cells other than 0, 4 and 5 are byte-identical to C012; cell 0
     differs only in its label strings (environment identical); cell 4 differs only by the
     appended scorer-mount block; cell 5 differs only by the embedded stage
     (src/division_scorer_stage.py verbatim) + wrapper inserted before write_test_submission;
  2. the scorer file loads through the embedded code with weights_only=True, its feature
     list equals the stage's FEATURES, and it scores a synthetic candidate finitely;
  3. kernel-metadata.json: private, T4, internet off, pilkwang + head + scorer datasets;
  4. with --replay: the local harness runs the built notebook on the 12 held-out movies
     (scorer via --env) and prints the 6bba proxy next to C012's.

    python src/verify_c016.py --scorer experiments/candidates/c016_division_scorer/dataset/division_scorer_v1.pt [--replay]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from build_c016_candidate import (  # noqa: E402
    C012_NB, DATASET_SLUG, DEST_DIR, DEST_NB, KERNEL_SLUG, STAGE_WRAPPER, SUBMISSION_CALL, embedded_stage_source,
)

ENV_ASSIGN = re.compile(r"""^os\.environ\[["']([A-Z0-9_]+)["']\]\s*=\s*(.+?)\s*$""", re.M)


def cells_of(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--threshold", default="0.6")
    parser.add_argument("--reparent-threshold", default="0.8")
    parser.add_argument("--reparent-max-prob", default="0.95")
    args = parser.parse_args()

    c012, c016 = cells_of(C012_NB), cells_of(DEST_NB)
    assert len(c016) == len(c012) == 12
    for i, src in enumerate(c016):
        compile(src, f"cell{i}", "exec")  # compile(), not ast.parse(): a misplaced __future__ import only fails here
    for i in range(12):
        if i not in (0, 4, 5):
            assert c016[i] == c012[i], f"cell {i} differs from C012"
    assert ENV_ASSIGN.findall(c016[0]) == ENV_ASSIGN.findall(c012[0]), "cell 0 environment drifted"
    added4 = [l[1:] for l in difflib.unified_diff(c012[4].splitlines(), c016[4].splitlines(), lineterm="", n=0)
              if l.startswith("+") and not l.startswith("+++")]
    removed4 = [l for l in difflib.unified_diff(c012[4].splitlines(), c016[4].splitlines(), lineterm="", n=0)
                if l.startswith("-") and not l.startswith("---")]
    sha = hashlib.sha256(args.scorer.read_bytes()).hexdigest()
    assert not removed4 and f"if _scorer_sha256 != '{sha}':" in "\n".join(added4), "cell 4: scorer block missing or SHA not pinned"
    assert "os.environ['BIOHUB_DIVISION_SCORER']" in "\n".join(added4)
    stage_src = embedded_stage_source()
    expected5 = c012[5].replace(SUBMISSION_CALL, "\n\n# ---- embedded src/division_scorer_stage.py (C016) ----\n" + stage_src + STAGE_WRAPPER + SUBMISSION_CALL, 1)
    assert c016[5] == expected5, "cell 5 is not C012 cell 5 + embedded stage + wrapper"
    assert "from __future__" not in c016[5], "cell 5 must not carry a __future__ import mid-cell"
    print("PASS text: cells 1-3, 6-11 identical to C012; cell 0 labels only; cell 4 = + scorer mount (SHA", sha[:16] + "); cell 5 = + embedded stage")

    import torch
    ns: dict = {"__name__": "c016_stage_check", "np": np, "torch": torch}
    exec(compile(stage_src, "division_scorer_stage.py", "exec"), ns)
    scorer = ns["load_scorer"](args.scorer)
    assert scorer["features"] == ns["FEATURES"], "scorer feature list != stage FEATURES"
    cand = {f: 0.5 for f in ns["FEATURES"]}
    cand.update(P=1, D1=2, D2=3, t=0, Q=None)
    s = ns["score_candidates"](scorer, [cand])
    assert s.shape == (1,) and np.isfinite(s).all() and 0.0 <= float(s[0]) <= 1.0
    nodes = {1: {"t": 0, "z": 10, "y": 100, "x": 100}, 2: {"t": 1, "z": 10, "y": 102, "x": 100}, 3: {"t": 1, "z": 10, "y": 100, "x": 120}}
    edges = [{"source_id": 1, "target_id": 2, "edge_prob": 0.9, "distance_um": 0.8}]
    stats: dict = {}
    out = ns["add_scored_divisions"](nodes, edges, stats, [cand], np.array([0.99], dtype=np.float32), 0.6, 0.8, 5)
    assert len(out) == 2 and stats["safe_divisions_added"] == 1 and out[-1]["source_id"] == 1 and out[-1]["target_id"] == 3
    print(f"PASS scorer: weights_only load, {len(scorer['features'])} features, synthetic candidate score {float(s[0]):.3f}, acceptance path OK")

    meta = json.loads((DEST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == f"taeyangg4/{KERNEL_SLUG}" and meta["code_file"] == DEST_NB.name
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert f"taeyangg4/{DATASET_SLUG}" in meta["dataset_sources"] and "taeyangg4/biohub-c012-v1284-head" in meta["dataset_sources"]
    print("PASS kernel-metadata:", meta["dataset_sources"])

    if args.replay:
        variants = REPO_ROOT / "reports" / "pp_replay" / "variants_c011_as_is.json"
        out = REPO_ROOT / "reports" / "pp_replay" / "c016_replay_all12.csv"
        subprocess.run([sys.executable, str(REPO_ROOT / "src" / "eval_pp_variants_local.py"), "--notebook", str(DEST_NB),
                        "--variants", str(variants), "--stems", "all12", "--round-coords",
                        "--pred-root", str(REPO_ROOT / "experiments/candidates/c012_v1284_head/e2e/head_v1/predictions"),
                        "--lowdet-dir", str(REPO_ROOT / "experiments/candidates/c012_v1284_head/e2e/head_v1/edge_cache"),
                        "--env", f"BIOHUB_DIVISION_SCORER={args.scorer.resolve()}",
                        "--env", f"BIOHUB_DIVISION_SCORE_THRESHOLD={args.threshold}",
                        "--env", f"BIOHUB_DIVISION_REPARENT_THRESHOLD={args.reparent_threshold}",
                        "--env", f"BIOHUB_DIVISION_REPARENT_MAX_PROB={args.reparent_max_prob}",
                        "--out", str(out)], check=True, cwd=REPO_ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        subprocess.run([sys.executable, str(REPO_ROOT / "src" / "rank_by_lb_proxy.py"), str(out),
                        str(REPO_ROOT / "reports/pp_replay/e2e_head_v1_all12.csv")], check=True, cwd=REPO_ROOT)
    print("C016 verification: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
