from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td


SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def load_graph(path: Path):
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g


def dist(pos: dict[int, np.ndarray], a: int, b: int) -> float:
    return float(np.linalg.norm(pos[a] - pos[b]))


def simulate(
    path: Path,
    *,
    existing_mode: str,
    parent_max: float = 9.0,
    sister_max: float = 14.0,
    existing_max: float = 10.0,
    symmetry_tau: float = 0.6,
    diverge_um: float = 2.25,
    require_mutual_nn: bool = True,
    require_divergence: bool = True,
    frame_frac_cap: float = 0.0076,
    global_frac_cap: float = 0.00375,
) -> dict:
    dataset = path.stem
    group = dataset.split("_", 1)[0]
    g = load_graph(path)
    ndf = g.node_attrs().to_pandas()[["node_id", "t", "z", "y", "x"]]
    edf = g.edge_attrs().to_pandas()[["source_id", "target_id"]]

    node_t = {int(r.node_id): int(r.t) for r in ndf.itertuples(index=False)}
    pos = {
        int(r.node_id): np.asarray([r.z, r.y, r.x], dtype=np.float32) * SCALE
        for r in ndf.itertuples(index=False)
    }
    gt_children: dict[int, list[int]] = defaultdict(list)
    for r in edf.itertuples(index=False):
        gt_children[int(r.source_id)].append(int(r.target_id))

    division_sources = {s: c for s, c in gt_children.items() if len(c) >= 2}
    missing_true: dict[int, int] = {}
    base_edges: list[tuple[int, int]] = []
    for s, children in gt_children.items():
        if len(children) < 2:
            for c in children:
                base_edges.append((s, c))
            continue
        ranked = sorted(children, key=lambda c: dist(pos, s, c))
        if existing_mode == "nearest":
            keep = ranked[0]
            drop = ranked[1]
        elif existing_mode == "farther":
            keep = ranked[-1]
            drop = ranked[0]
        else:
            raise ValueError(existing_mode)
        base_edges.append((s, keep))
        missing_true[s] = drop

    out_by_source: dict[int, list[int]] = defaultdict(list)
    incoming: set[int] = set()
    for s, t in base_edges:
        out_by_source[s].append(t)
        incoming.add(t)

    ids_by_t: dict[int, list[int]] = defaultdict(list)
    for nid, t in node_t.items():
        ids_by_t[t].append(nid)

    stats = defaultdict(int)
    fail_reason: dict[int, str] = {}
    true_pass_stage = defaultdict(int)
    global_cap = max(1, int(round(max(1, len(base_edges)) * global_frac_cap)))
    added: list[tuple[int, int]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()

    true_pass_stage["division_total"] = len(missing_true)

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [nid for nid in ids_by_t[t] if len(out_by_source.get(nid, [])) == 1]
        candidate_ids = [nid for nid in child_frame_ids if nid not in incoming and nid not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        candidate_tree = cKDTree(np.stack([pos[c] for c in candidate_ids])) if require_mutual_nn else None
        frame_cap = max(1, int(round(len(source_ids) * frame_frac_cap)))
        proposals: list[tuple[float, int, int]] = []

        for source_id in source_ids:
            existing_child_id = out_by_source[source_id][0]
            child_dist = dist(pos, source_id, existing_child_id)
            true_cand = missing_true.get(source_id)
            if true_cand is not None:
                true_pass_stage["source_eligible"] += 1

            if child_dist > existing_max:
                if true_cand is not None:
                    fail_reason[source_id] = "existing_max"
                continue
            if true_cand is not None:
                true_pass_stage["existing_max"] += 1

            mutual_nn_id = None
            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(pos[existing_child_id])
                mutual_nn_id = candidate_ids[int(nn_idx)]

            for candidate_id in candidate_ids:
                parent_dist = dist(pos, source_id, candidate_id)
                if parent_dist > parent_max:
                    if candidate_id == true_cand and source_id not in fail_reason:
                        fail_reason[source_id] = "parent_max"
                    continue
                if candidate_id == true_cand:
                    true_pass_stage["parent_max"] += 1

                sister_dist = dist(pos, existing_child_id, candidate_id)
                if sister_dist > sister_max:
                    if candidate_id == true_cand and source_id not in fail_reason:
                        fail_reason[source_id] = "sister_max"
                    continue
                if candidate_id == true_cand:
                    true_pass_stage["sister_max"] += 1

                if require_mutual_nn and candidate_id != mutual_nn_id:
                    stats["mutual_nn_rejected"] += 1
                    if candidate_id == true_cand and source_id not in fail_reason:
                        fail_reason[source_id] = "mutual_nn"
                    continue
                if candidate_id == true_cand:
                    true_pass_stage["mutual_nn"] += 1

                if require_divergence:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats["divergence_rejected"] += 1
                        if candidate_id == true_cand and source_id not in fail_reason:
                            fail_reason[source_id] = "divergence_context"
                        continue
                    gc1, gc2 = c1_succ[0], q_succ[0]
                    if node_t.get(gc1) != t + 2 or node_t.get(gc2) != t + 2:
                        stats["divergence_rejected"] += 1
                        if candidate_id == true_cand and source_id not in fail_reason:
                            fail_reason[source_id] = "divergence_context"
                        continue
                    grandchild_dist = dist(pos, gc1, gc2)
                    if grandchild_dist - sister_dist < diverge_um:
                        stats["divergence_rejected"] += 1
                        if candidate_id == true_cand and source_id not in fail_reason:
                            fail_reason[source_id] = "divergence_amount"
                        continue
                if candidate_id == true_cand:
                    true_pass_stage["divergence"] += 1

                stats["geometric_candidates"] += 1
                if symmetry_tau > 0.0:
                    denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / denom > symmetry_tau:
                        stats["symmetry_rejected"] += 1
                        if candidate_id == true_cand and source_id not in fail_reason:
                            fail_reason[source_id] = "symmetry"
                        continue
                if candidate_id == true_cand:
                    true_pass_stage["symmetry"] += 1
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id))

        stats["proposals"] += len(proposals)
        proposals.sort(key=lambda x: x[0])
        added_this_frame = 0
        for _, source_id, candidate_id in proposals:
            if len(added) >= global_cap:
                stats["cap_skipped"] += 1
                break
            if added_this_frame >= frame_cap:
                stats["cap_skipped"] += 1
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            added.append((source_id, candidate_id))
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1

    added_set = set(added)
    truth_set = {(s, c) for s, c in missing_true.items()}
    tp = len(added_set & truth_set)
    fp = len(added_set - truth_set)
    fn = len(truth_set - added_set)
    for s, c in truth_set:
        if (s, c) in added_set:
            fail_reason[s] = "recovered"
        elif s not in fail_reason:
            fail_reason[s] = "proposal_lost_or_cap"
    reason_counts = defaultdict(int)
    for r in fail_reason.values():
        reason_counts[r] += 1

    return {
        "dataset": dataset,
        "group": group,
        "existing_mode": existing_mode,
        "nodes": len(node_t),
        "gt_edges": len(edf),
        "division_total": len(truth_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / len(truth_set) if truth_set else 1.0,
        "added": len(added),
        "global_cap": global_cap,
        "fail_reasons": dict(reason_counts),
        "true_pass_stage": dict(true_pass_stage),
        "internal_stats": dict(stats),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for mode in sorted({r["existing_mode"] for r in rows}):
        rr = [r for r in rows if r["existing_mode"] == mode]
        by_group = {}
        for group in sorted({r["group"] for r in rr}):
            gg = [r for r in rr if r["group"] == group]
            tp = sum(r["tp"] for r in gg)
            fp = sum(r["fp"] for r in gg)
            fn = sum(r["fn"] for r in gg)
            reasons = defaultdict(int)
            stages = defaultdict(int)
            for r in gg:
                for k, v in r["fail_reasons"].items():
                    reasons[k] += v
                for k, v in r["true_pass_stage"].items():
                    stages[k] += v
            by_group[group] = {
                "movies": len(gg),
                "division_total": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 1.0,
                "recall": tp / (tp + fn) if tp + fn else 1.0,
                "fail_reasons": dict(reasons),
                "true_pass_stage": dict(stages),
            }
        result[mode] = by_group
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    paths = sorted(args.gt_root.glob("*.geff"))
    for i, p in enumerate(paths, 1):
        for mode in ("nearest", "farther"):
            rows.append(simulate(p, existing_mode=mode))
        if i % 25 == 0 or i == len(paths):
            print("processed", i, "of", len(paths), flush=True)
    report = {
        "config": {
            "parent_max": 9.0,
            "sister_max": 14.0,
            "existing_max": 10.0,
            "symmetry_tau": 0.6,
            "diverge_um": 2.25,
            "require_mutual_nn": True,
            "require_divergence": True,
            "frame_frac_cap": 0.0076,
            "global_frac_cap": 0.00375,
            "deepcenter_not_simulated": True,
        },
        "aggregate": aggregate(rows),
        "per_movie": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
