#!/usr/bin/env python3
"""C055 stage: Lineage Forge V12 guarded endpoint readmission on the C023 replay.

Ports ``readmit_open_track_ends`` from flexonafft's Lineage Forge V12
(state/public_lineage_review_20260929/latest_cell_05.txt, lines 1366-1477) with its
fixed constants (score >= 0.94, step <= 3 um, motion residual <= 1.5 um,
separation >= 1.5 um, ambiguity margin 0.5, bridge preference, budget
min(100, floor(0.002 * nodes))) and places it where V12 places it: after the
motion relink and the single-parent / single-child repairs, before the
single-frame gap closer.  C023's own radius readmission is switched off by the
variant (READMIT_RADIUS_UM = 0) so the two stages are never applied together.

Candidate source: C023's existing low-detection dump (``<stem>.npz`` with
``low_coords`` int16 [t, z, y, x] already scaled to the node voxel grid and
``low_score`` = sigmoid at the peak).  The dump and V12's capture call the same
``_detect_cells_pooled``; its local-maximum mask does not depend on the
threshold, so the dump filtered at >= 0.94 is V12's candidate set.

One documented coordinate-semantics adaptation: C023 node coordinates are
V1284-head-refined, so an existing node no longer sits exactly on its raw peak
(measured up to 1.61 um away).  V12 relies on the 1.5 um separation test to
reject a node's own peak; here the raw peaks that belong to existing graph
nodes are excluded by identity first (nearest peak of the same frame within
``NODE_PEAK_IDENTITY_UM``), then V12's separation test runs unchanged.

The stage never edits the notebook or the pinned harness; it is installed into
the replay namespace built by ``eval_pp_variants_local.build_namespace``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

READMIT_MIN_SCORE = 0.94
READMIT_MAX_STEP_UM = 3.0
READMIT_MAX_RESIDUAL_UM = 1.5
READMIT_MIN_SEPARATION_UM = 1.5
READMIT_MAX_ADDED_FRACTION = 0.002
READMIT_MAX_ADDED_ABS = 100
# Only used for the identity exclusion of existing nodes' own raw peaks (see module docstring).
NODE_PEAK_IDENTITY_UM = 2.5
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def load_guarded_candidates(lowdet_dir, dataset, min_score=READMIT_MIN_SCORE):
    """V12 ``load_readmit_candidates`` on C023's dump: score >= min_score, one entry per
    rounded (t, z, y, x) keeping the maximum score."""
    path = Path(lowdet_dir) / f"{dataset}.npz"
    if not path.exists():
        raise RuntimeError(f"Readmit capture missing for {dataset}: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "low_coords" not in data.files or "low_score" not in data.files:
            raise RuntimeError(f"{path} has no low-detection dump")
        low = np.asarray(data["low_coords"]).reshape(-1, 4)
        score = np.asarray(data["low_score"]).reshape(-1)
    result = {}
    for (t, z, y, x), s in zip(low.tolist(), score.tolist()):
        if float(s) >= min_score:
            key = tuple(int(round(v)) for v in (t, z, y, x))
            result[key] = max(result.get(key, 0.0), float(s))
    return [{"t": t, "z": z, "y": y, "x": x, "readmit_score": s} for (t, z, y, x), s in result.items()]


def assign_node_peaks(node_um, peak_um, identity_um=NODE_PEAK_IDENTITY_UM, k=3):
    """Greedy one-to-one assignment of nodes to their own raw peaks within one frame.

    Every (node, peak) pair among each node's ``k`` nearest peaks within
    ``identity_um`` is taken in ascending distance order, each node and each
    peak at most once.  Returns (assigned_peak_indices, unassigned_nodes,
    conflicts) where conflicts counts nodes whose nearest peak was already taken
    by a closer node (resolved by the next nearest peak).
    """
    if len(node_um) == 0 or len(peak_um) == 0:
        return set(), len(node_um), 0
    kk = min(k, len(peak_um))
    d, j = cKDTree(peak_um).query(node_um, k=kk)
    d = np.asarray(d).reshape(len(node_um), kk)
    j = np.asarray(j).reshape(len(node_um), kk)
    pairs = sorted((float(d[n, r]), n, int(j[n, r])) for n in range(len(node_um)) for r in range(kk) if d[n, r] <= identity_um)
    taken_nodes, taken_peaks, conflicts = set(), set(), 0
    for dist, n, p in pairs:
        if n in taken_nodes:
            continue
        if p in taken_peaks:
            conflicts += 1
            continue
        taken_nodes.add(n)
        taken_peaks.add(p)
    return taken_peaks, len(node_um) - len(taken_nodes), conflicts


def exclude_node_peaks(candidates, nodes_by_id, stats, identity_um=NODE_PEAK_IDENTITY_UM):
    """Drop candidates that are the raw peaks of existing graph nodes (one-to-one identity per
    frame within identity_um; see assign_node_peaks)."""
    scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
    by_t = {}
    for index, c in enumerate(candidates):
        by_t.setdefault(int(c["t"]), []).append(index)
    node_um = {}
    for node in nodes_by_id.values():
        node_um.setdefault(int(node["t"]), []).append(
            np.array([float(node["z"]), float(node["y"]), float(node["x"])]) * scale)
    owned = set()
    unassigned = 0
    conflicts = 0
    for t, indices in by_t.items():
        nodes = node_um.get(t)
        if not nodes:
            continue
        pts = np.array([[candidates[i]["z"], candidates[i]["y"], candidates[i]["x"]] for i in indices], dtype=np.float64) * scale
        taken, missing, conf = assign_node_peaks(np.stack(nodes), pts, identity_um)
        owned.update(indices[p] for p in taken)
        unassigned += missing
        conflicts += conf
    stats["readmit_candidates_all"] = len(candidates)
    stats["readmit_node_peaks_excluded"] = len(owned)
    stats["readmit_node_peak_unassigned"] = unassigned
    stats["readmit_node_peak_conflicts"] = conflicts
    return [c for i, c in enumerate(candidates) if i not in owned]


def make_readmit(ns):
    """Return V12's readmit_open_track_ends bound to the notebook's own distance helpers."""
    edge_distance_um = ns["edge_distance_um"]
    point_distance_um = ns["point_distance_um"]
    node_point = ns["node_point"]

    def readmit_open_track_ends(nodes_by_id, edges, stats, candidates):
        stats["readmit_candidates"] = len(candidates)
        stats["readmit_nodes"] = 0
        stats["readmit_edges"] = 0
        stats["readmit_bridges"] = 0
        stats["readmit_proposals"] = 0
        stats["readmit_budget"] = 0
        stats["readmit_added_ids"] = []
        stats["readmit_added_edges"] = []
        if not nodes_by_id or not edges or not candidates:
            return nodes_by_id, edges
        incoming, outgoing, frames, candidate_frames = {}, {}, {}, {}
        for node_id, node in nodes_by_id.items():
            frames.setdefault(int(node["t"]), []).append(node_id)
        for edge in edges:
            source, target = int(edge["source_id"]), int(edge["target_id"])
            outgoing.setdefault(source, []).append(target)
            incoming.setdefault(target, []).append(source)
        for index, candidate in enumerate(candidates):
            if float(candidate.get("readmit_score", 0)) >= READMIT_MIN_SCORE:
                candidate_frames.setdefault(int(candidate["t"]), []).append(index)
        ends = {i for i in nodes_by_id if len(incoming.get(i, [])) == 1 and not outgoing.get(i)}
        starts = {i for i in nodes_by_id if len(outgoing.get(i, [])) == 1 and not incoming.get(i)}
        proposals = []
        for endpoint in sorted(ends | starts):
            direction = 1 if endpoint in ends else -1
            node = nodes_by_id[endpoint]
            context_ids = incoming.get(endpoint, []) if direction == 1 else outgoing.get(endpoint, [])
            context = nodes_by_id[context_ids[0]]
            if int(context["t"]) != int(node["t"]) - direction:
                continue
            if len(outgoing.get(context_ids[0], [])) != 1 or len(incoming.get(context_ids[0], [])) > 1:
                continue
            if edge_distance_um(node, context) > READMIT_MAX_STEP_UM:
                continue
            predicted = tuple(2.0 * a - b for a, b in zip(node_point(node), node_point(context)))
            frame = int(node["t"]) + direction
            matches = []
            for index in candidate_frames.get(frame, []):
                candidate = candidates[index]
                step = edge_distance_um(node, candidate)
                residual = point_distance_um(predicted, node_point(candidate))
                if step > READMIT_MAX_STEP_UM or residual > READMIT_MAX_RESIDUAL_UM:
                    continue
                if any(edge_distance_um(candidate, nodes_by_id[j]) < READMIT_MIN_SEPARATION_UM for j in frames.get(frame, [])):
                    continue
                matches.append((residual + 0.25 * step, -candidate["readmit_score"], index))
            matches.sort()
            if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < 0.5):
                continue
            cost, _, index = matches[0]
            candidate = candidates[index]
            opposite = starts if direction == 1 else ends
            reconnect = []
            for other in sorted(opposite):
                other_node = nodes_by_id[other]
                if int(other_node["t"]) != frame + direction:
                    continue
                distance = edge_distance_um(candidate, other_node)
                if distance <= READMIT_MAX_STEP_UM:
                    target_prediction = tuple(2.0 * a - b for a, b in zip(node_point(candidate), node_point(node)))
                    error = point_distance_um(target_prediction, node_point(other_node))
                    if error <= READMIT_MAX_RESIDUAL_UM:
                        reconnect.append((error + 0.25 * distance, other))
            reconnect.sort()
            other = reconnect[0][1] if reconnect and (len(reconnect) == 1 or reconnect[1][0] - reconnect[0][0] >= 0.5) else None
            proposals.append((0 if other is not None else 1, cost, endpoint, direction, index, other))
        budget = min(READMIT_MAX_ADDED_ABS, int(len(nodes_by_id) * READMIT_MAX_ADDED_FRACTION))
        stats["readmit_proposals"] = len(proposals)
        stats["readmit_budget"] = budget
        used_candidates, used_endpoints = set(), set()
        next_id = max(nodes_by_id) + 1
        for _, _, endpoint, direction, index, other in sorted(proposals):
            if stats["readmit_nodes"] >= budget:
                break
            if index in used_candidates or endpoint in used_endpoints or other in used_endpoints:
                continue
            candidate = candidates[index]
            if any(edge_distance_um(candidate, nodes_by_id[j]) < READMIT_MIN_SEPARATION_UM for j in frames.get(int(candidate["t"]), [])):
                continue
            node_id = next_id
            next_id += 1
            nodes_by_id[node_id] = {"node_id": node_id, **candidate, "readmitted": 1}
            frames.setdefault(int(candidate["t"]), []).append(node_id)
            pairs = [(endpoint, node_id)] if direction == 1 else [(node_id, endpoint)]
            if other is not None:
                pairs.append((node_id, other) if direction == 1 else (other, node_id))
                used_endpoints.add(other)
                stats["readmit_bridges"] += 1
            for source, target in pairs:
                edges.append({"source_id": source, "target_id": target, "edge_prob": None,
                              "distance_um": edge_distance_um(nodes_by_id[source], nodes_by_id[target]), "readmitted": 1})
                stats["readmit_added_edges"].append((int(source), int(target)))
            used_candidates.add(index)
            used_endpoints.add(endpoint)
            stats["readmit_added_ids"].append(int(node_id))
            stats["readmit_nodes"] += 1
            stats["readmit_edges"] += len(pairs)
        return nodes_by_id, edges

    return readmit_open_track_ends


def install_guarded_readmit(ns, lowdet_dir, graph_folder=None):
    """Install the stage into a replay namespace.

    Inert while ``ns['GUARDED_READMIT']`` is 0 (variants toggle it).  Wraps
    ``close_single_frame_gaps`` (V12's insertion point) and the outer
    ``filter_output_graph`` to record per-movie survival of the added nodes /
    edges through every later stage and, when ``graph_folder`` is given, to save
    the final graph as ids / txyz / edges (rounded like the submission writer).
    Variant labels come from ``ns['_C055_LABEL']``.
    """
    ns.setdefault("GUARDED_READMIT", 0)
    ns.setdefault("_C055_LABEL", "unlabelled")
    ns["_C055_RECORDS"] = {}
    readmit = make_readmit(ns)
    close_gaps = ns["close_single_frame_gaps"]
    outer = ns["filter_output_graph"]
    last = {}

    def close_after_guarded(nodes_by_id, edges, stats, *args, **kwargs):
        last.clear()
        if int(ns["GUARDED_READMIT"]):
            dataset = kwargs.get("dataset")
            candidates = exclude_node_peaks(load_guarded_candidates(lowdet_dir, dataset), nodes_by_id, stats)
            before_nodes, before_edges = len(nodes_by_id), len(edges)
            nodes_by_id, edges = readmit(nodes_by_id, edges, stats, candidates)
            print(f"  [{dataset}] guarded readmit: candidates={stats['readmit_candidates']} "
                  f"(all>=0.94 {stats['readmit_candidates_all']}, node peaks excluded {stats['readmit_node_peaks_excluded']}) "
                  f"proposals={stats['readmit_proposals']} budget={stats['readmit_budget']} "
                  f"added={stats['readmit_nodes']} bridges={stats['readmit_bridges']} edges={stats['readmit_edges']} "
                  f"nodes {before_nodes}->{len(nodes_by_id)} edges {before_edges}->{len(edges)}")
            last.update(added_ids=list(stats["readmit_added_ids"]), added_edges=list(stats["readmit_added_edges"]),
                        candidates=stats["readmit_candidates"], candidates_all=stats["readmit_candidates_all"],
                        node_peaks_excluded=stats["readmit_node_peaks_excluded"], proposals=stats["readmit_proposals"],
                        budget=stats["readmit_budget"], bridges=stats["readmit_bridges"],
                        node_peak_unassigned=stats["readmit_node_peak_unassigned"], node_peak_conflicts=stats["readmit_node_peak_conflicts"])
        return close_gaps(nodes_by_id, edges, stats, *args, **kwargs)

    def filter_with_record(nodes_by_id, raw_edges, *args, **kwargs):
        dataset = kwargs.get("dataset")
        nodes, edges, stats = outer(nodes_by_id, raw_edges, *args, **kwargs)
        label = str(ns["_C055_LABEL"])
        added = set(last.get("added_ids", []))
        added_edges = set(tuple(e) for e in last.get("added_edges", []))
        final_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
        record = dict(stem=dataset, label=label,
                      readmit_candidates=int(last.get("candidates", 0)), readmit_candidates_all=int(last.get("candidates_all", 0)),
                      readmit_node_peaks_excluded=int(last.get("node_peaks_excluded", 0)),
                      readmit_node_peak_unassigned=int(last.get("node_peak_unassigned", 0)),
                      readmit_node_peak_conflicts=int(last.get("node_peak_conflicts", 0)),
                      readmit_proposals=int(last.get("proposals", 0)), readmit_budget=int(last.get("budget", 0)),
                      readmit_nodes=len(added), readmit_bridges=int(last.get("bridges", 0)), readmit_edges=len(added_edges),
                      readmit_nodes_final=sum(1 for i in added if i in nodes),
                      readmit_edges_final=sum(1 for p in added_edges if p in final_pairs),
                      readmit_incident_final=sum(1 for s, t in final_pairs if s in added or t in added),
                      final_nodes=len(nodes), final_edges=len(edges))
        for key in ("readmitted_nodes", "gapfill_added_nodes", "safe_divisions_added", "v1057_restored", "v1057_displaced"):
            record["stat_" + key] = int(stats.get(key, 0))
        ns["_C055_RECORDS"][(dataset, label)] = record
        if graph_folder is not None:
            folder = Path(graph_folder)
            folder.mkdir(parents=True, exist_ok=True)
            plain = ns["nodes_by_id_to_plain"](nodes)
            ids = sorted(plain)
            np.savez_compressed(folder / f"{dataset}_{label}.npz", ids=np.array(ids, np.int64),
                                txyz=np.array([[int(plain[i][0]), *[max(0, int(round(v))) for v in plain[i][1:]]] for i in ids], np.int64),
                                edges=np.array(sorted(final_pairs), np.int64).reshape(-1, 2),
                                readmitted_ids=np.array(sorted(added), np.int64))
        return nodes, edges, stats

    ns["close_single_frame_gaps"] = close_after_guarded
    ns["filter_output_graph"] = filter_with_record
