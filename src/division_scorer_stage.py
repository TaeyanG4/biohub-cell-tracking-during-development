"""C016 division scorer: candidate enumeration, features, torch model and the post-processing stage.

This file is imported by the local tools (candidate generation, training) AND embedded verbatim
into the C016 notebook, so the features seen at training time are exactly the features computed
on Kaggle. It only needs numpy, scipy and torch, plus what x138's post-processing already has:
nodes_by_id / edges dicts, the low-detection dump (`admitted` candidate edges with learned
probabilities, `low_coords` / `low_score` detection peaks), a DeepCenter point scorer and the raw
frame reader (`read_test_frame`).

Candidate: P (node at t, exactly one child D1 at t+1) + D2 (another node at t+1 within
PARENT_UM of P, sister distance D1-D2 <= SISTER_UM). D2 may be free or parented by Q != P.

Feature groups (v3):
  geometry      dist_p_d1, dist_p_d2, sister, asym, cos_d1_d2, cos_vel_d2, vel_um, dz_p_d2, dz_p_d1
  learned edges prob_p_d1, prob_p_d2 (transformer probability of P->D2 if the ILP ever saw it)
  D2's parent Q d2_has_parent, prob_q_d2, dist_q_d2, q_d2_relinked, q_d2_gap (edge came from
                motion relink / gap filling rather than the ILP), q_has_parent, q_n_children,
                q_track_len, score_q, q_synthetic
  continuity    d2_has_child, d2_child_step, d2_track_len_after, p_track_len,
                det_t_at_d2 / det_t_at_d2_dist (is there a detection peak at D2's position one
                frame earlier - a newborn daughter should have none, a continuing cell has one),
                sister_t2 (distance between the daughters' own children at t+2; sisters keep separating)
  detections    score_p, score_d1, score_d2, dc_p, dc_d1, dc_d2, dc_mid, p_synthetic, d1_synthetic,
                d2_synthetic (node created by readmit / gap filling)
  appearance    raw-frame intensity, normalised per frame ((v - median) / (p99.5 - median) of a
                subsample): int_p_peak / int_p_mean (a dividing parent is bright and condensed),
                int_p_prev_peak (its predecessor at t-1), int_d1_peak, int_d2_peak, int_d2_mean,
                int_mid_mean (between the daughters at t+1: dark if they really are two cells)
  context       density_t, density_t1, frame_frac
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

VOX = np.array([1.625, 0.40625, 0.40625])
PARENT_UM = 14.0
SISTER_UM = 16.0
TRACK_LEN_CAP = 10
INT_WIN = (1, 3, 3)  # +- voxels (z, y, x) around a node for the intensity window (~3.3 x 2.4 x 2.4 um)
FEATURES = [
    "dist_p_d1", "dist_p_d2", "sister", "asym", "cos_d1_d2", "cos_vel_d2", "vel_um", "dz_p_d2", "dz_p_d1",
    "prob_p_d1", "prob_p_d2",
    "d2_has_parent", "prob_q_d2", "dist_q_d2", "q_d2_relinked", "q_d2_gap", "q_has_parent", "q_n_children",
    "q_track_len", "score_q", "q_synthetic",
    "d2_has_child", "d2_child_step", "d2_track_len_after", "p_track_len", "det_t_at_d2", "det_t_at_d2_dist", "sister_t2",
    "score_p", "score_d1", "score_d2", "dc_p", "dc_d1", "dc_d2", "dc_mid", "p_synthetic", "d1_synthetic", "d2_synthetic",
    "int_p_peak", "int_p_mean", "int_p_prev_peak", "int_d1_peak", "int_d2_peak", "int_d2_mean", "int_mid_mean",
    "density_t", "density_t1", "frame_frac",
]
SYNTHETIC_FLAGS = ("readmitted", "gap_synthetic", "gapfill_peak")


def make_model(n_in: int, hidden: int = 32) -> torch.nn.Sequential:
    """hidden=0 gives plain logistic regression (the most robust choice with ~100 positives)."""
    if hidden <= 0:
        return torch.nn.Sequential(torch.nn.Linear(n_in, 1))
    return torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                               torch.nn.SiLU(), torch.nn.Linear(hidden, 1))


def load_scorer(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(len(saved["features"]), int(saved["hidden"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return {"model": model, "mean": saved["mean"].float(), "scale": saved["scale"].float(), "features": list(saved["features"])}


def score_candidates(scorer, candidates):
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    x = torch.tensor([[float(c[f]) for f in scorer["features"]] for c in candidates], dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(scorer["model"]((x - scorer["mean"]) / scorer["scale"]).squeeze(1)).numpy()


def pos_um(node) -> np.ndarray:
    return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * VOX


def is_synthetic(node) -> int:
    return int(any(int(node.get(flag, 0) or 0) == 1 for flag in SYNTHETIC_FLAGS))


def load_dump(npz):
    """Candidate-edge probabilities and per-frame detection-peak trees from an edge_cache dump."""
    admitted = {(int(a), int(b)): float(p) for a, b, p, _ in np.asarray(npz["admitted"]).reshape(-1, 4)}
    low_by_t = {}
    lc, ls = np.asarray(npz["low_coords"], dtype=np.float64).reshape(-1, 4), np.asarray(npz["low_score"], dtype=np.float64).reshape(-1)
    for t in np.unique(lc[:, 0]).astype(int):
        sel = lc[:, 0] == t
        low_by_t[int(t)] = (cKDTree(lc[sel, 1:] * VOX), ls[sel])
    return admitted, low_by_t


def make_intensity_fn(dataset, read_frame, frame_cache):
    """intensity(t, z, y, x) -> (peak, mean) of the raw frame in a small window, normalised per frame by
    (v - median) / (p99.5 - median) of a strided subsample. `read_frame(dataset, t, frame_cache)` is the
    notebook's read_test_frame; frames are requested in increasing t so its bounded cache is enough."""
    norm = {}

    def frame_stats(t):
        if t not in norm:
            vol = read_frame(dataset, int(t), frame_cache)
            sub = vol[::2, ::8, ::8].astype(np.float32)
            med = float(np.median(sub))
            norm[t] = (med, max(float(np.percentile(sub, 99.5)) - med, 1.0))
        return norm[t]

    def intensity(t, z, y, x):
        vol = read_frame(dataset, int(t), frame_cache)
        if vol is None or vol.ndim != 3:
            return 0.0, 0.0
        med, scale = frame_stats(int(t))
        zi, yi, xi = int(round(float(z))), int(round(float(y))), int(round(float(x)))
        zi = min(max(zi, 0), vol.shape[0] - 1); yi = min(max(yi, 0), vol.shape[1] - 1); xi = min(max(xi, 0), vol.shape[2] - 1)
        patch = vol[max(0, zi - INT_WIN[0]):zi + INT_WIN[0] + 1, max(0, yi - INT_WIN[1]):yi + INT_WIN[1] + 1,
                    max(0, xi - INT_WIN[2]):xi + INT_WIN[2] + 1]
        if patch.size == 0:
            return 0.0, 0.0
        return (float(patch.max()) - med) / scale, (float(patch.mean()) - med) / scale

    return intensity


def enumerate_candidates(nodes_by_id, edges, admitted, low_by_t, deepcenter_point, intensity_point=None,
                         parent_um=PARENT_UM, sister_um=SISTER_UM):
    """All (P, D1, D2) triples with their feature dicts. `deepcenter_point(node_dict) -> float`,
    `intensity_point(t, z, y, x) -> (peak, mean)` (None -> zeros). Parents are visited in frame order."""
    out_edges, in_edge = {}, {}
    for e in edges:
        out_edges.setdefault(int(e["source_id"]), []).append(e)
        in_edge[int(e["target_id"])] = e
    by_t = {}
    for nid, n in nodes_by_id.items():
        by_t.setdefault(int(n["t"]), []).append(nid)
    trees = {t: (cKDTree(np.stack([pos_um(nodes_by_id[i]) for i in ids])), ids) for t, ids in by_t.items()}
    t_max = max(by_t) if by_t else 1
    ancestors_cache, descendants_cache = {}, {}

    def ancestors(nid):
        """Number of consecutive predecessors (capped), i.e. how long the track has existed before nid."""
        if nid in ancestors_cache:
            return ancestors_cache[nid]
        n, cur = 0, nid
        while n < TRACK_LEN_CAP:
            e = in_edge.get(cur)
            if e is None:
                break
            cur = int(e["source_id"]); n += 1
        ancestors_cache[nid] = n
        return n

    def descendants(nid):
        """Number of consecutive successors following the first child (capped)."""
        if nid in descendants_cache:
            return descendants_cache[nid]
        n, cur = 0, nid
        while n < TRACK_LEN_CAP:
            kids = out_edges.get(cur, [])
            if not kids:
                break
            cur = int(kids[0]["target_id"]); n += 1
        descendants_cache[nid] = n
        return n

    def det_score(node):
        entry = low_by_t.get(int(node["t"]))
        if entry is None:
            return 0.0
        d, j = entry[0].query(pos_um(node), k=1)
        return float(entry[1][j]) if d <= 2.0 else 0.0

    def det_at(t, um):
        """(score, distance) of the nearest detection peak of frame t to position `um`; distance capped at 6."""
        entry = low_by_t.get(int(t))
        if entry is None:
            return 0.0, 6.0
        d, j = entry[0].query(um, k=1)
        return (float(entry[1][j]) if d <= 2.0 else 0.0), float(min(d, 6.0))

    def edge_prob(e):
        p = e.get("edge_prob")
        return float(p) if p is not None else 0.0

    def intensity(t, z, y, x):
        return intensity_point(t, z, y, x) if intensity_point is not None else (0.0, 0.0)

    def first_child(nid):
        kids = out_edges.get(nid, [])
        return int(kids[0]["target_id"]) if kids else None

    cands = []
    for t in sorted(by_t):
        if t + 1 not in trees:
            continue
        tree, ids = trees[t + 1]
        for P in by_t[t]:
            n_p = nodes_by_id[P]
            kids = out_edges.get(P, [])
            if len(kids) != 1:
                continue
            D1 = int(kids[0]["target_id"])
            if int(nodes_by_id[D1]["t"]) != t + 1:
                continue
            p_um, d1_um = pos_um(n_p), pos_um(nodes_by_id[D1])
            near = tree.query_ball_point(p_um, r=parent_um)
            prev = in_edge.get(P)
            prev_node = nodes_by_id[int(prev["source_id"])] if prev else None
            prev_um = pos_um(prev_node) if prev_node is not None else None
            vel = (p_um - prev_um) if prev_um is not None else np.zeros(3)
            vel_n = float(np.linalg.norm(vel))
            density_t = len(trees[t][0].query_ball_point(p_um, r=10.0)) - 1
            density_t1 = len(near)
            d1_child = first_child(D1)
            d1_child_um = pos_um(nodes_by_id[d1_child]) if d1_child is not None else None
            dc_p = score_p = None
            for j in near:
                D2 = ids[j]
                if D2 == D1:
                    continue
                n_d2 = nodes_by_id[D2]
                d2_um = pos_um(n_d2)
                sister = float(np.linalg.norm(d1_um - d2_um))
                if sister > sister_um:
                    continue
                if dc_p is None:
                    dc_p, score_p = deepcenter_point(n_p), det_score(n_p)
                    dc_d1, score_d1 = deepcenter_point(nodes_by_id[D1]), det_score(nodes_by_id[D1])
                    p_track_len, p_syn, d1_syn = ancestors(P), is_synthetic(n_p), is_synthetic(nodes_by_id[D1])
                    int_p_peak, int_p_mean = intensity(t, n_p["z"], n_p["y"], n_p["x"])
                    int_p_prev_peak = intensity(t - 1, prev_node["z"], prev_node["y"], prev_node["x"])[0] if prev_node is not None else -1.0
                    int_d1_peak = intensity(t + 1, nodes_by_id[D1]["z"], nodes_by_id[D1]["y"], nodes_by_id[D1]["x"])[0]
                other = in_edge.get(D2)
                Q = int(other["source_id"]) if other else None
                n_q = nodes_by_id[Q] if Q is not None else None
                v1, v2 = d1_um - p_um, d2_um - p_um
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                cos12 = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-6 and n2 > 1e-6 else 0.0
                cos_vel = float(np.dot(vel, v2) / (vel_n * n2)) if vel_n > 1e-6 and n2 > 1e-6 else 0.0
                d2_kids = out_edges.get(D2, [])
                d2_child = int(d2_kids[0]["target_id"]) if d2_kids else None
                det_t_score, det_t_dist = det_at(t, d2_um)
                mid_z, mid_y, mid_x = ((float(nodes_by_id[D1][k]) + float(n_d2[k])) / 2 for k in ("z", "y", "x"))
                mid = {"t": t + 1, "z": mid_z, "y": mid_y, "x": mid_x}
                int_d2_peak, int_d2_mean = intensity(t + 1, n_d2["z"], n_d2["y"], n_d2["x"])
                sister_t2 = (float(np.linalg.norm(d1_child_um - pos_um(nodes_by_id[d2_child])))
                             if d1_child_um is not None and d2_child is not None else -1.0)
                cands.append({
                    "P": P, "D1": D1, "D2": D2, "t": t, "Q": Q,
                    "dist_p_d1": n1, "dist_p_d2": n2, "sister": sister, "asym": abs(n1 - n2) / max((n1 + n2) / 2, 1e-6),
                    "cos_d1_d2": cos12, "cos_vel_d2": cos_vel, "vel_um": vel_n, "dz_p_d2": float(abs(v2[0])), "dz_p_d1": float(abs(v1[0])),
                    "prob_p_d1": admitted.get((P, D1), edge_prob(kids[0])), "prob_p_d2": admitted.get((P, D2), 0.0),
                    "d2_has_parent": int(Q is not None),
                    "prob_q_d2": (admitted.get((Q, D2), edge_prob(other)) if Q is not None else 0.0),
                    "dist_q_d2": float(np.linalg.norm(pos_um(n_q) - d2_um)) if Q is not None else -1.0,
                    "q_d2_relinked": int(other is not None and int(other.get("motion_relinked", 0) or 0) == 1),
                    "q_d2_gap": int(other is not None and any(int(other.get(k, 0) or 0) == 1 for k in ("gap_closed", "gap2_recovered", "gap_filled"))),
                    "q_has_parent": int(Q is not None and Q in in_edge),
                    "q_n_children": len(out_edges.get(Q, [])) if Q is not None else 0,
                    "q_track_len": ancestors(Q) if Q is not None else -1,
                    "score_q": det_score(n_q) if Q is not None else 0.0,
                    "q_synthetic": is_synthetic(n_q) if Q is not None else 0,
                    "d2_has_child": int(d2_child is not None),
                    "d2_child_step": float(np.linalg.norm(pos_um(nodes_by_id[d2_child]) - d2_um)) if d2_child is not None else -1.0,
                    "d2_track_len_after": descendants(D2), "p_track_len": p_track_len,
                    "det_t_at_d2": det_t_score, "det_t_at_d2_dist": det_t_dist, "sister_t2": sister_t2,
                    "score_p": score_p, "score_d1": score_d1, "score_d2": det_score(n_d2),
                    "dc_p": dc_p, "dc_d1": dc_d1, "dc_d2": deepcenter_point(n_d2), "dc_mid": deepcenter_point(mid),
                    "p_synthetic": p_syn, "d1_synthetic": d1_syn, "d2_synthetic": is_synthetic(n_d2),
                    "int_p_peak": int_p_peak, "int_p_mean": int_p_mean, "int_p_prev_peak": int_p_prev_peak,
                    "int_d1_peak": int_d1_peak, "int_d2_peak": int_d2_peak, "int_d2_mean": int_d2_mean,
                    "int_mid_mean": intensity(t + 1, mid_z, mid_y, mid_x)[1],
                    "density_t": density_t, "density_t1": density_t1, "frame_frac": t / max(1, t_max),
                })
    return cands


def add_scored_divisions(nodes_by_id, edges, stats, candidates, scores, threshold, reparent_threshold, max_added,
                         reparent_max_prob=1.01):
    """Accept scored candidates greedily: best score first, one division per parent, each D2 used once;
    a D2 that already has a parent Q is re-parented only above `reparent_threshold` (its Q->D2 edge is removed)
    and never when the ILP gave that Q->D2 edge a probability >= `reparent_max_prob` (confident links stay)."""
    order = np.argsort(-scores)
    used_parent, used_d2, removed = set(), set(), set()
    new_edges = []
    for i in order:
        c, s = candidates[int(i)], float(scores[int(i)])
        if s < threshold or len(new_edges) >= max_added:
            break
        # one fork per parent; a node never forks in two consecutive frames (P after its own parent forked,
        # or D1 after P) - the metric would pair only one of them with the GT division and count the other as FP
        if c["P"] in used_parent or c["D1"] in used_parent or c["D2"] in used_d2 or c["D2"] in used_parent or c["D1"] in used_d2:
            continue
        if c["Q"] is not None:
            if s < reparent_threshold or c["Q"] in used_parent or float(c["prob_q_d2"]) >= reparent_max_prob:
                continue
            removed.add((c["Q"], c["D2"]))
        used_parent.add(c["P"]); used_parent.add(c["D1"]); used_d2.add(c["D2"])
        p, d2 = nodes_by_id[c["P"]], nodes_by_id[c["D2"]]
        dist = math.sqrt(sum(((float(p[k]) - float(d2[k])) * v) ** 2 for k, v in zip(("z", "y", "x"), VOX)))
        new_edges.append({"source_id": c["P"], "target_id": c["D2"], "edge_prob": None, "distance_um": dist,
                          "safe_division": 1, "scored_division": 1, "division_score": s})
    kept = [e for e in edges if (int(e["source_id"]), int(e["target_id"])) not in removed]
    stats["safe_divisions_added"] = stats.get("safe_divisions_added", 0) + len(new_edges)
    stats["scored_division_candidates"] = len(candidates)
    stats["scored_division_reparented"] = len(removed)
    return kept + new_edges
