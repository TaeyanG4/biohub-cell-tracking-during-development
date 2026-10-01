#!/usr/bin/env python3
"""Verify Candidate C017 (C012 + jump-stabilized motion relink) before any push.

  1. every code cell compiles; cells other than 0 and 5 are byte-identical to C012; cell 0 differs only in label
     strings (identical environment); cell 5 = C012 cell 5 + the stabilization block before write_test_submission;
  2. kernel-metadata.json: private, T4, internet off, exactly C012's datasets;
  3. with --replay: the harness runs the built notebook as configured on the 12 held-out movies and the result
     must equal the harness option `--stabilize-relink` at the same threshold (reports/pp_replay/stabilize_head_v1_all12.csv).

    python src/verify_c017.py [--replay]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from build_c017_candidate import C012_META, C012_NB, DEST_DIR, DEST_NB, KERNEL_SLUG, SUBMISSION_CALL  # noqa: E402

ENV_ASSIGN = re.compile(r"""^os\.environ\[["']([A-Z0-9_]+)["']\]\s*=\s*(.+?)\s*$""", re.M)


def cells_of(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    c012, c017 = cells_of(C012_NB), cells_of(DEST_NB)
    assert len(c012) == len(c017) == 12
    for i, src in enumerate(c017):
        compile(src, f"cell{i}", "exec")
    for i in range(12):
        if i not in (0, 5):
            assert c017[i] == c012[i], f"cell {i} differs from C012"
    assert ENV_ASSIGN.findall(c017[0]) == ENV_ASSIGN.findall(c012[0]), "cell 0 environment drifted"
    before, after = c012[5].split(SUBMISSION_CALL), c017[5].split(SUBMISSION_CALL)
    assert len(before) == len(after) == 2 and after[1] == before[1] and after[0].startswith(before[0]), "cell 5: change outside the inserted block"
    block = after[0][len(before[0]):]
    assert "def motion_relink_edges(" in block and "_unstabilized_motion_relink_edges = motion_relink_edges" in block
    added = [l for l in difflib.unified_diff(c012[5].splitlines(), c017[5].splitlines(), lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    min_um = re.search(r'BIOHUB_STAB_MIN_UM", "([0-9.]+)"', block).group(1)
    print(f"PASS text: cells 1-4, 6-11 identical to C012; cell 0 labels only; cell 5 = + stabilization block ({len(added)} lines, STAB_MIN_UM {min_um})")
    meta = json.loads((DEST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    base = json.loads(C012_META.read_text(encoding="utf-8"))
    assert meta["id"] == f"taeyangg4/{KERNEL_SLUG}" and meta["code_file"] == DEST_NB.name
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert sorted(meta["dataset_sources"]) == sorted(base["dataset_sources"]), "datasets differ from C012"
    print("PASS kernel-metadata:", meta["dataset_sources"])
    if args.replay:
        c = REPO_ROOT / "experiments" / "candidates" / "c012_v1284_head" / "e2e" / "head_v1"
        out = REPO_ROOT / "reports" / "pp_replay" / "c017_replay_all12.csv"
        subprocess.run([sys.executable, str(REPO_ROOT / "src" / "eval_pp_variants_local.py"), "--notebook", str(DEST_NB),
                        "--variants", str(REPO_ROOT / "reports" / "pp_replay" / "variants_c011_as_is.json"), "--stems", "all12",
                        "--round-coords", "--pred-root", str(c / "predictions"), "--lowdet-dir", str(c / "edge_cache"), "--out", str(out)],
                       check=True, cwd=REPO_ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        ref = {r["stem"]: r for r in csv.DictReader((REPO_ROOT / "reports" / "pp_replay" / "stabilize_head_v1_all12.csv").open(encoding="utf-8"))
               if r["config"] == f"stab_{float(min_um):.1f}"}
        got = {r["stem"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
        diffs = [s for s in ref if abs(float(ref[s]["adjusted_edge_jaccard"]) - float(got[s]["adjusted_edge_jaccard"])) > 1e-9
                 or ref[s]["edges"] != got[s]["edges"] or ref[s]["div_tp"] != got[s]["div_tp"] or ref[s]["div_fp"] != got[s]["div_fp"]]
        assert not diffs, f"built notebook differs from the harness option on {diffs}"
        print(f"PASS replay: built notebook == harness --stabilize-relink stab_{float(min_um):.1f} on all 12 movies (edges, adj, divisions)")
        subprocess.run([sys.executable, str(REPO_ROOT / "src" / "rank_by_lb_proxy.py"), str(out),
                        str(REPO_ROOT / "reports" / "pp_replay" / "e2e_head_v1_all12.csv")], check=True, cwd=REPO_ROOT)
    print("C017 verification: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
