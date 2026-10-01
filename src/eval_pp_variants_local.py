#!/usr/bin/env python3
"""Replay a harmonic-fusion-lineage notebook's post-processing on cached ILP graphs.

Executes the notebook's own configuration, post-processing and validator-scoring
code (code cells 0-2, 5 and 8 of the 12-cell HF / x138 / C010 layout), skips
package install, inference and submission writing, and scores post-processing
variants against local GT with the notebook's implementation of the official
metric: size-weighted adjusted edge Jaccard + 0.1 x micro-averaged division
Jaccard (the same numbers the in-notebook validator prints as PROXY_SCORE).

Default inputs are the ILP graphs saved by C004's Kaggle run
(tmp/c004_log/tracking_repo/predictions): the 8 held-out train movies of its
validator and the 4 visible-test movies (train copies), GT from
data/train/<stem>.geff. The Reyhan (C004) and HF/x138 post-processing functions
are logically identical, so C004's validator table is a calibration target.

The x138 readmit and low-detection gap-filler stages read the low-detection
dump written during inference; without --lowdet-dir they stay idle, exactly as
on Kaggle when the dump is missing.

Needs the global Python (torch + tracksdata 0.1.0rc6), not .venv:

    python src/eval_pp_variants_local.py \
        --notebook state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb \
        --variants reports/pp_replay/variants_div.json --stems all12 \
        --out reports/pp_replay/x138_div.csv
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_ROOT = REPO_ROOT / "tmp" / "c004_log" / "tracking_repo" / "predictions"
DEFAULT_GT_DIR = REPO_ROOT / "data" / "train"
DEEPCENTER_CHECKPOINT = REPO_ROOT / "artifacts" / "pilkwang_deepcenter" / "weights" / "full_frame_center" / "best.pt"
DEEPCENTER_MANIFEST = REPO_ROOT / "artifacts" / "pilkwang_deepcenter" / "ARTIFACT_MANIFEST.json"

STEM_SETS = {
    # C004 validator held-out set (VALIDATOR_N_PER_TYPE=4, division-bearing first).
    "val8": [
        "44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f",
        "6bba_062c8d37", "6bba_07e24132", "6bba_085bf656", "6bba_09961292",
    ],
    # Visible test movies (train copies with local GT).
    "vis4": ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"],
}
STEM_SETS["all12"] = STEM_SETS["val8"] + STEM_SETS["vis4"]

SUBMISSION_CALL = '\nwrite_test_submission("base")\n'


def code_cells(notebook: Path) -> list[str]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def build_namespace(notebook: Path, env_overrides: dict[str, str], lowdet_dir: Path | None) -> dict:
    cells = code_cells(notebook)
    if len(cells) != 12:
        raise SystemExit(f"{notebook}: expected the 12-cell harmonic-fusion layout, found {len(cells)} code cells")
    if cells[5].count(SUBMISSION_CALL) != 1:
        raise SystemExit(f"{notebook}: could not isolate the write_test_submission call in cell 5")

    ns: dict = {"__name__": "__pp_replay__"}
    exec(compile(cells[0], f"{notebook.name}:cell0", "exec"), ns)
    # Local paths for what cell 0 points at /kaggle/input or /kaggle/working.
    os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(DEEPCENTER_CHECKPOINT)
    os.environ["BIOHUB_DEEPCENTER_MANIFEST"] = str(DEEPCENTER_MANIFEST)
    os.environ["BIOHUB_CACHE_DIR"] = str(lowdet_dir) if lowdet_dir else ""
    os.environ["BIOHUB_REPAIR_DEADLINE_S"] = "1e12"
    os.environ.update(env_overrides)
    exec(compile(cells[1], f"{notebook.name}:cell1", "exec"), ns)
    exec(compile(cells[2], f"{notebook.name}:cell2", "exec"), ns)
    exec(compile(cells[5].replace(SUBMISSION_CALL, "\n"), f"{notebook.name}:cell5", "exec"), ns)
    if ns.get("DEEPCENTER_VETO_DETECTOR") is None and ns.get("USE_DEEPCENTER_VETO"):
        raise SystemExit("DeepCenter add-only gate did not load; results would not match Kaggle")
    ns.update(VALIDATOR_MATCH_RADIUS_UM=7.0, VALIDATOR_NODE_COUNT_PENALTY_A=0.1, VALIDATOR_DIVISION_WEIGHT=0.1)
    exec(compile(cells[8], f"{notebook.name}:cell8", "exec"), ns)
    return ns


def install_frame_caches(ns: dict) -> tuple[dict, dict]:
    """Share frames and DeepCenter heatmaps across variants of one movie.

    Both depend only on (movie, t), so reusing them keeps every variant on the
    same DeepCenter scores and avoids recomputing ~100 heatmaps per variant.
    """
    frames: dict = {}
    heatmaps: dict = {}
    read_frame = ns["read_test_frame"]
    heatmap_for_frame = ns["deepcenter_heatmap_for_frame"]

    def cached_read_test_frame(dataset, t, frame_cache):
        key = (dataset, int(t))
        if key not in frames:
            frames[key] = read_frame(dataset, t, {})
        return frames[key]

    def cached_heatmap_for_frame(dataset, t, detector_bundle, frame_cache, heatmap_cache):
        key = (dataset, int(t))
        if key not in heatmaps:
            heatmap = heatmap_for_frame(dataset, t, detector_bundle, frame_cache, {})
            if heatmap is None:
                return None
            heatmaps[key] = heatmap
        return heatmaps[key]

    ns["read_test_frame"] = cached_read_test_frame
    ns["deepcenter_heatmap_for_frame"] = cached_heatmap_for_frame
    return frames, heatmaps


def install_weak_leaf_prune(ns: dict, source_notebook: Path) -> None:
    """Graft amanatar's prune_weak_leaf_nodes (verbatim) into the replayed notebook.

    It runs where amanatar's filter_output_graph runs it: after short-track
    filtering, before line-fit smoothing. Inert until a variant sets
    LEAF_PRUNE_MIN_EDGE_PROB > 0.
    """
    import ast

    for src in code_cells(source_notebook):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "prune_weak_leaf_nodes":
                exec(compile(ast.get_source_segment(src, node), f"{source_notebook.name}:prune_weak_leaf_nodes", "exec"), ns)
                break
    if "prune_weak_leaf_nodes" not in ns:
        raise SystemExit(f"{source_notebook}: prune_weak_leaf_nodes not found")
    ns.setdefault("LEAF_PRUNE_MIN_EDGE_PROB", 0.0)
    linefit = ns["linefit_smooth_output_graph"]

    def linefit_after_leaf_prune(nodes_by_id, edges, stats):
        if ns["LEAF_PRUNE_MIN_EDGE_PROB"] > 0.0:
            nodes_by_id, kept_edges = ns["prune_weak_leaf_nodes"](nodes_by_id, edges, stats)
            edges[:] = kept_edges  # filter_output_graph returns this same list
        return linefit(nodes_by_id, edges, stats)

    ns["linefit_smooth_output_graph"] = linefit_after_leaf_prune


def ilp_forks_of(raw_edges: list) -> dict:
    """{parent: [(child, prob), ...]} for every ILP parent with two or more children (transformer-scored forks)."""
    by_source: dict[int, list] = {}
    for e in raw_edges:
        by_source.setdefault(int(e["source_id"]), []).append((int(e["target_id"]), float(e["edge_prob"] or 0.0)))
    return {p: sorted(kids, key=lambda k: -k[1]) for p, kids in by_source.items() if len(kids) >= 2}


def install_ilp_fork_reinjection(ns: dict) -> None:
    """Re-add the ILP's own forks after the one-to-one motion relink discarded them.

    x138's final edges are the Hungarian relink output, so a fork the ILP made (second child with
    transformer probability above the ILP division weight) never reaches the output. This wrapper runs
    right before add_safe_divisions_postlink: for each ILP fork parent that now has exactly one child, it
    adds the missing child edge when that child's probability >= ILP_FORK_MIN_PROB and the child is free or
    (ILP_FORK_REPARENT) held by a weaker link; x138's division geometry filter still validates the fork.
    Inert while ILP_FORK_MIN_PROB <= 0. Variants set ILP_FORK_MIN_PROB / ILP_FORK_REPARENT.
    """
    ns.setdefault("ILP_FORK_MIN_PROB", 0.0)
    ns.setdefault("ILP_FORK_REPARENT", 1)
    ns.setdefault("ILP_FORK_MAX_PARENT_UM", 14.0)
    ns.setdefault("ILP_FORK_MAX_SISTER_UM", 16.0)
    ns.setdefault("_ILP_FORKS", {})
    rule = ns["add_safe_divisions_postlink"]
    edge_distance_um = ns["edge_distance_um"]

    def reinject(nodes_by_id, edges, stats):
        min_prob = float(ns["ILP_FORK_MIN_PROB"])
        forks = ns.get("_ILP_FORKS") or {}
        if min_prob <= 0 or not forks:
            return edges
        out_by_source: dict[int, list] = {}
        in_edge: dict[int, dict] = {}
        for e in edges:
            out_by_source.setdefault(int(e["source_id"]), []).append(e)
            in_edge[int(e["target_id"])] = e
        added, reparented, removed = 0, 0, set()
        new_edges = []
        for parent, kids in forks.items():
            if parent not in nodes_by_id:
                continue
            current = out_by_source.get(parent, [])
            if len(current) != 1:
                continue
            current_child = int(current[0]["target_id"])
            for child, prob in kids:
                if child == current_child or child not in nodes_by_id or prob < min_prob:
                    continue
                if int(nodes_by_id[child]["t"]) != int(nodes_by_id[parent]["t"]) + 1:
                    continue
                # loose sanity envelope (the C016 candidate gate); when x138's division geometry filter is ON
                # (off by default) only forks it will keep are added, otherwise DIV_DROP_TO_SINGLE_IF_BAD could
                # drop the correct first child instead of the re-injected one
                d_new = edge_distance_um(nodes_by_id[parent], nodes_by_id[child])
                d_cur = edge_distance_um(nodes_by_id[parent], nodes_by_id[current_child])
                sister = edge_distance_um(nodes_by_id[current_child], nodes_by_id[child])
                max_parent, max_sister = float(ns["ILP_FORK_MAX_PARENT_UM"]), float(ns["ILP_FORK_MAX_SISTER_UM"])
                if ns.get("OUTPUT_DIVISION_GEOMETRY_FILTER"):
                    max_parent, max_sister = min(max_parent, float(ns["DIV_PARENT_MAX_UM"])), min(max_sister, float(ns["DIV_SISTER_MAX_UM"]))
                if max(d_new, d_cur) > max_parent or sister > max_sister:
                    stats["ilp_forks_geometry_rejected"] = stats.get("ilp_forks_geometry_rejected", 0) + 1
                    continue
                other = in_edge.get(child)
                if other is not None:
                    other_prob = other.get("edge_prob")
                    if not int(ns["ILP_FORK_REPARENT"]) or (other_prob is not None and float(other_prob) >= prob):
                        continue
                    if len(out_by_source.get(int(other["source_id"]), [])) <= 1 and int(other["source_id"]) in forks:
                        pass  # the competitor is itself a fork parent; still allow, its own fork is handled on its turn
                    removed.add(id(other)); reparented += 1
                new_edges.append({"source_id": parent, "target_id": child, "edge_prob": prob,
                                  "distance_um": edge_distance_um(nodes_by_id[parent], nodes_by_id[child]), "ilp_fork": 1})
                added += 1
                break
        stats["ilp_forks_added"] = added
        stats["ilp_forks_reparented"] = reparented
        return [e for e in edges if id(e) not in removed] + new_edges

    def safe_divisions_after_reinject(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        edges = reinject(nodes_by_id, edges, stats)
        return rule(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                    frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)

    ns["add_safe_divisions_postlink"] = safe_divisions_after_reinject


def install_stabilized_relink(ns: dict) -> None:
    """Remove whole-field jumps before the motion relink (forum: hengck23 'beware of jumps', 724283).

    Acquisition hiccups make every cell of a frame pair move together by several micrometres; x138's
    distance-gated Hungarian relink (tight 5.5 um, seed flow from same-pair tight matches) then breaks links
    the ILP had right (C012, 22 movies: 20 % / 48 % of GT edges missed on 5-8 um / >8 um jumps vs 3 % on
    normal pairs). Per frame pair the global shift g_t is the component-wise median displacement of the ILP's
    own links (learned_edge_probs, prob >= STAB_MIN_PROB); pairs with |g_t| >= STAB_MIN_UM are treated as jumps.
    Node positions are shifted by the cumulative jump sum before the relink (geometry only), so each jump pair
    looks like a normal pair; returned edges get their distances recomputed on the original coordinates.
    Inert while STAB_MIN_UM <= 0.
    """
    ns.setdefault("STAB_MIN_UM", 0.0)
    ns.setdefault("STAB_MIN_PROB", 0.5)
    ns.setdefault("STAB_MIN_EDGES", 8)
    relink = ns["motion_relink_edges"]
    edge_distance_um = ns["edge_distance_um"]
    vox = np.array([1.625, 0.40625, 0.40625])

    def stabilized(nodes_by_id, stats, learned_edge_probs=None):
        min_um = float(ns["STAB_MIN_UM"])
        if min_um <= 0 or not learned_edge_probs:
            return relink(nodes_by_id, stats, learned_edge_probs)
        disp: dict[int, list] = {}
        for (s, d), p in learned_edge_probs.items():
            a, b = nodes_by_id.get(s), nodes_by_id.get(d)
            if a is None or b is None or int(b["t"]) != int(a["t"]) + 1 or float(p) < float(ns["STAB_MIN_PROB"]):
                continue
            disp.setdefault(int(a["t"]), []).append(
                (np.array([float(b["z"]), float(b["y"]), float(b["x"])]) - np.array([float(a["z"]), float(a["y"]), float(a["x"])])) * vox)
        offsets = {}
        for t, vs in disp.items():
            if len(vs) >= int(ns["STAB_MIN_EDGES"]):
                med = np.median(np.array(vs), axis=0)
                if float(np.linalg.norm(med)) >= min_um:
                    offsets[t] = med
        if not offsets:
            return relink(nodes_by_id, stats, learned_edge_probs)
        times = sorted({int(n["t"]) for n in nodes_by_id.values()})
        cum, acc = {}, np.zeros(3)
        for t in times:
            cum[t] = acc.copy()
            if t in offsets:
                acc = acc + offsets[t]
        shifted = {}
        for nid, n in nodes_by_id.items():
            c = cum[int(n["t"])] / vox
            shifted[nid] = {**n, "z": float(n["z"]) - c[0], "y": float(n["y"]) - c[1], "x": float(n["x"]) - c[2]}
        edges = relink(shifted, stats, learned_edge_probs)
        for e in edges:
            a, b = nodes_by_id.get(int(e["source_id"])), nodes_by_id.get(int(e["target_id"]))
            if a is not None and b is not None:
                e["distance_um"] = edge_distance_um(a, b)
        stats["stabilized_jump_pairs"] = len(offsets)
        return edges

    ns["motion_relink_edges"] = stabilized


def install_ilp_edge_restore(ns: dict) -> None:
    """josephadamski V1057 reconciliation (audit D), run after all of filter_output_graph.

    Every ILP edge with p >= V1057_MIN_PROB that is not in the final graph is put back, conflicts resolved by
    descending probability (each source and target used once); nodes with two children and their daughters
    are never touched; the final edges it displaces are dropped; no nodes are added or removed.
    Inert while V1057_MIN_PROB <= 0.
    """
    ns.setdefault("V1057_MIN_PROB", 0.0)
    inner = ns["filter_output_graph"]

    def restored(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
        raw = [(int(e["source_id"]), int(e["target_id"]), e.get("edge_prob")) for e in raw_edges]
        nodes, edges, stats = inner(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
        tau = float(ns["V1057_MIN_PROB"])
        if tau <= 0 or not edges:
            return nodes, edges, stats
        node_time = {int(k): int(v["t"]) for k, v in nodes.items()}
        anchor = {(int(e["source_id"]), int(e["target_id"])): e for e in edges}
        outgoing: dict[int, list] = {k: [] for k in node_time}
        incoming: dict[int, list] = {k: [] for k in node_time}
        for s, t in anchor:
            outgoing.setdefault(s, []).append(t); incoming.setdefault(t, []).append(s)
        cand: dict[tuple, float] = {}
        for s, t, p in raw:
            if s not in node_time or t not in node_time or (s, t) in anchor or node_time[t] != node_time[s] + 1:
                continue
            if len(outgoing.get(s, [])) > 1:
                continue
            owner = incoming[t][0] if len(incoming.get(t, [])) == 1 else None
            if owner is not None and len(outgoing.get(owner, [])) > 1:
                continue
            if p is None or not np.isfinite(float(p)) or float(p) < tau:
                continue
            cand[(s, t)] = max(float(p), cand.get((s, t), float("-inf")))
        used_s, used_t, selected = set(), set(), []
        for (s, t), p in sorted(cand.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
            if s in used_s or t in used_t:
                continue
            selected.append((s, t, p)); used_s.add(s); used_t.add(t)
        kept = [e for (s, t), e in anchor.items() if s not in used_s and t not in used_t]
        new = [{"source_id": s, "target_id": t, "edge_prob": p, "v1057_restored": 1} for s, t, p in selected]
        stats["v1057_restored"] = len(new)
        stats["v1057_displaced"] = len(anchor) - len(kept)
        return nodes, kept + new, stats

    ns["filter_output_graph"] = restored


def install_weak_edge_filter(ns: dict) -> None:
    """amanatar 'optimized-biohub-max-score' L1 weak-edge output filter (review 2026-09-25), run after all of
    filter_output_graph: drop edges whose learned edge_prob < WEAK_EDGE_MIN_PROB (edges without a probability are
    exempt), then drop nodes left without any edge except on the first / last frame. Inert while <= 0."""
    ns.setdefault("WEAK_EDGE_MIN_PROB", 0.0)
    inner = ns["filter_output_graph"]

    def filtered(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
        nodes, edges, stats = inner(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
        tau = float(ns["WEAK_EDGE_MIN_PROB"])
        if tau <= 0 or not edges:
            return nodes, edges, stats
        kept, dropped = [], 0
        for e in edges:
            p = e.get("edge_prob")
            if p is None or float(p) >= tau:
                kept.append(e)
            else:
                dropped += 1
        if dropped:
            t_min = min(int(n["t"]) for n in nodes.values()); t_max = max(int(n["t"]) for n in nodes.values())
            linked = {int(e["source_id"]) for e in kept} | {int(e["target_id"]) for e in kept}
            nodes = {nid: n for nid, n in nodes.items() if nid in linked or int(n["t"]) in (t_min, t_max)}
        stats["weak_edges_dropped"] = dropped
        return nodes, kept, stats

    ns["filter_output_graph"] = filtered


def apply_overrides(ns: dict, overrides: dict) -> dict:
    saved = {}
    for key, value in overrides.items():
        if key not in ns:
            raise KeyError(f"{key} is not a notebook global")
        current = ns[key]
        saved[key] = current
        if isinstance(current, bool):
            ns[key] = value if isinstance(value, bool) else str(value).strip() not in ("0", "false", "False", "")
        elif isinstance(current, (int, float, str)):
            ns[key] = type(current)(value)
        else:
            ns[key] = value
    return saved


def load_raw_graph(ns: dict, path: Path) -> tuple[dict, list]:
    graph = ns["graph_from_geff"](path)
    nodes: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {"node_id": node_id, "t": int(row["t"]),
                          "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    edges: list[dict] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        prob = row.get("edge_prob")
        edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                      "edge_prob": None if prob is None else float(prob)})
    return nodes, edges


def summarize(ns: dict, rows: list[dict], labels: list[str], groups: dict[str, list[str]]) -> list[dict]:
    out = []
    for label in labels:
        for group, stems in groups.items():
            sel = [r for r in rows if r["config"] == label and r["stem"] in stems]
            if not sel:
                continue
            agg = ns["aggregate_official"](sel)
            agg.update(config=label, group=group, n=len(sel),
                       safe_divisions_added=sum(r["safe_divisions_added"] for r in sel),
                       readmitted_nodes=sum(r["readmitted_nodes"] for r in sel),
                       gapfill_added_nodes=sum(r["gapfill_added_nodes"] for r in sel))
            out.append(agg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True,
                        help='JSON {"label": {"NOTEBOOK_GLOBAL": value, ...}, ...}; {} is the notebook as configured')
    parser.add_argument("--stems", default="all12", help="val8 | vis4 | all12 | comma-separated stems")
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--lowdet-dir", type=Path, default=None,
                        help="directory with <stem>.npz low-detection dumps (enables readmit / gap filler)")
    parser.add_argument("--env", action="append", default=[], metavar="BIOHUB_KEY=VALUE",
                        help="extra environment override applied after cell 0")
    parser.add_argument("--round-coords", action="store_true",
                        help="score node coordinates rounded like write_test_submission (max(0, round(v)))")
    parser.add_argument("--weak-leaf-prune-from", type=Path, default=None, metavar="NOTEBOOK",
                        help="graft prune_weak_leaf_nodes from this notebook (amanatar geometric fusion); "
                             "variants then set LEAF_PRUNE_MIN_EDGE_PROB")
    parser.add_argument("--weak-edge-filter", action="store_true", help="amanatar L1 weak-edge output filter (variants set WEAK_EDGE_MIN_PROB)")
    parser.add_argument("--structured-trajectory", type=Path, default=None, metavar="ASSETS_DIR",
                        help="C033 fixed public structured stage; variants set STRUCTURED_TRAJECTORY_MODE off/raw/stabilized")
    parser.add_argument("--ilp-edge-restore", action="store_true",
                        help="josephadamski V1057: put back ILP edges displaced by post-processing (variants set V1057_MIN_PROB)")
    parser.add_argument("--stabilize-relink", action="store_true",
                        help="remove whole-field jumps before the motion relink (variants set STAB_MIN_UM / STAB_MIN_PROB)")
    parser.add_argument("--reinject-ilp-forks", action="store_true",
                        help="re-add the ILP's forks after the motion relink (variants set ILP_FORK_MIN_PROB / ILP_FORK_REPARENT)")
    parser.add_argument("--out", type=Path, required=True, help="per-movie CSV; the summary goes next to it")
    args = parser.parse_args()

    stems = STEM_SETS.get(args.stems) or [s.strip() for s in args.stems.split(",") if s.strip()]
    variants: dict[str, dict] = json.loads(args.variants.read_text(encoding="utf-8"))
    env_overrides = dict(item.split("=", 1) for item in args.env)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = build_namespace(args.notebook.resolve(), env_overrides, args.lowdet_dir)
    ns["TEST_DIR"] = args.gt_dir.resolve()  # read_test_frame -> local <stem>.zarr frames
    frames, heatmaps = install_frame_caches(ns)
    if args.weak_leaf_prune_from:
        install_weak_leaf_prune(ns, args.weak_leaf_prune_from)
    if args.reinject_ilp_forks:
        install_ilp_fork_reinjection(ns)
    if args.stabilize_relink:
        install_stabilized_relink(ns)
    if args.ilp_edge_restore:
        install_ilp_edge_restore(ns)
    if args.weak_edge_filter:
        install_weak_edge_filter(ns)
    if args.structured_trajectory:
        from structured_trajectory_stage import install_structured_trajectory
        folder = args.structured_trajectory
        install_structured_trajectory(ns, (folder / 'structured-trajectory.py').read_text(encoding='utf-8'),
                                      json.loads((folder / 'structured-trajectory-model.json').read_text(encoding='utf-8')))

    import pandas as pd

    rows: list[dict] = []
    t_start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for stem in stems:
            pred_path = next(args.pred_root.rglob(f"{stem}.geff"), None)
            gt_path = args.gt_dir / f"{stem}.geff"
            if pred_path is None or not gt_path.exists():
                raise SystemExit(f"{stem}: missing prediction ({pred_path}) or GT ({gt_path})")
            raw_nodes, raw_edges = load_raw_graph(ns, pred_path)
            ns["_ILP_FORKS"] = ilp_forks_of(raw_edges) if args.reinject_ilp_forks else {}
            gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](gt_path))
            t_true = ns["read_estimated_true_node_count"](gt_path)
            for label, overrides in variants.items():
                saved = apply_overrides(ns, overrides)
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(log):
                        print(f"===== {stem} | {label} | {overrides}")
                        nodes, edges, stats = ns["filter_output_graph"](
                            copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                            deepcenter_bundle=ns.get("DEEPCENTER_VETO_DETECTOR"),
                        )
                finally:
                    for key, value in saved.items():
                        ns[key] = value
                pred_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
                pred_nodes = ns["nodes_by_id_to_plain"](nodes)
                if args.round_coords:
                    pred_nodes = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in pred_nodes.items()}
                row = ns["score_sample"](pred_nodes, pred_edges, gt_nodes, gt_edges, t_true)
                row.update(
                    stem=stem, config=label, nodes=len(nodes), edges=len(edges),
                    safe_divisions_added=int(stats.get("safe_divisions_added", 0)),
                    readmitted_nodes=int(stats.get("readmitted_nodes", 0)),
                    gapfill_added_nodes=int(stats.get("gapfill_added_nodes", 0)),
                    leaf_pruned_nodes=int(stats.get("leaf_prune_nodes", 0)),
                    flow_frames=int(stats.get("motion_relink_flow_frames", 0)),
                    seconds=round(time.time() - t0, 1),
                )
                rows.append(row)
                print(f"{stem:15s} {label:28s} adj={row['adjusted_edge_jaccard']:.4f} "
                      f"div(tp/fp/fn)={row['div_tp']}/{row['div_fp']}/{row['div_fn']} "
                      f"nodes={row['nodes']} edges={row['edges']} [{row['seconds']:.0f}s]", flush=True)
            frames.clear()
            heatmaps.clear()

    groups = {"all": stems}
    for name in ("val8", "vis4"):
        members = [s for s in stems if s in STEM_SETS[name]]
        if members and len(members) != len(stems):
            groups[name] = members
    for prefix in sorted({s.split("_")[0] for s in stems}):
        groups[prefix] = [s for s in stems if s.startswith(prefix + "_")]
    summary = summarize(ns, rows, list(variants), groups)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print(f"\n{'config':28s} {'group':6s} {'score':>7s} {'adj':>7s} {'divJ':>6s}  div tp/fp/fn   safe_div  readmit  gapfill")
    for s in summary:
        print(f"{s['config']:28s} {s['group']:6s} {s['proxy_score']:.4f}  {s['adjusted_edge_jaccard']:.4f} "
              f"{s['division_jaccard']:.4f}  {s['div_tp']:>3d}/{s['div_fp']:>3d}/{s['div_fn']:>3d}  "
              f"{s['safe_divisions_added']:>8d} {s['readmitted_nodes']:>8d} {s['gapfill_added_nodes']:>8d}")
    print(f"\nrows -> {args.out}\nsummary -> {summary_path}\nlog -> {log_path}\n"
          f"total {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
