"""C060 track-conditioned raw-image data, reusing immutable C058 inputs.

Runtime context eligibility uses only actual C023 graph topology, synthetic
provenance and image bounds. ``source_labels`` additionally requires known GT
agreement on both context edges, strictly as source-embryo training supervision.
That training filter must never select nodes for inference or evaluation.
"""
from __future__ import annotations

import collections
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from c055_guarded_readmit import save_json, sha
from c058_raw_localizer_study import (
    ROOT, DEST as C058_DEST, metadata, get_frame, bounded_crop,
    load_baseline, eligible as central_eligible,
)

CROP = (13, 49, 49)
EXPANDED = (15, 57, 57)
VOXEL_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
CHANNELS = ("previous", "current", "next")


def load_graph(stem: str) -> dict[str, np.ndarray]:
    """Load a copy of C058's unchanged, provenance-labelled C023 final graph."""
    return load_baseline(C058_DEST, stem)


def context_indices(graph: dict, shape, crop=CROP):
    """Return ``(previous_rows, next_rows, eligible)`` without reading GT.

    All arrays have one element per input graph row. Invalid context indices
    are -1. Eligibility requires three real nodes at consecutive times, unique
    incoming/outgoing edges along the triple, no branching at any of its three
    nodes, and full real image support for ``crop`` around all three centers.
    Topology and coordinate arrays are never modified.
    """
    ids = np.asarray(graph["ids"], dtype=np.int64)
    points = np.asarray(graph["txyz"])
    edges = np.asarray(graph["edges"], dtype=np.int64).reshape(-1, 2)
    synthetic = np.asarray(graph["gap_synthetic"], dtype=bool)
    shape = tuple(int(v) for v in shape)
    crop = tuple(int(v) for v in crop)
    if len(shape) != 4 or len(crop) != 3 or any(v <= 0 or v % 2 == 0 for v in crop):
        raise ValueError("Expected TZYX image shape and a positive odd ZYX crop")
    if points.shape != (len(ids), 4) or synthetic.shape != (len(ids),):
        raise ValueError("Graph arrays have inconsistent lengths")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(points).all():
        raise ValueError("Graph IDs must be unique and coordinates finite")
    if not np.equal(points, np.round(points)).all():
        raise ValueError("Context must use the original integer-writer coordinates")
    if len(points) and ((points[:, 0] < 0).any() or (points[:, 0] >= shape[0]).any()):
        raise ValueError("Graph time lies outside the image")

    lookup = {int(node): row for row, node in enumerate(ids)}
    previous = np.full(len(ids), -1, dtype=np.int64)
    following = previous.copy()
    incoming = np.zeros(len(ids), dtype=np.int32)
    outgoing = incoming.copy()
    for source, target in edges:
        if int(source) not in lookup or int(target) not in lookup:
            raise ValueError("Graph has a dangling edge")
        a, b = lookup[int(source)], lookup[int(target)]
        outgoing[a] += 1
        incoming[b] += 1
        previous[b] = a
        following[a] = b
    if not len(ids):
        return previous, following, np.zeros(0, dtype=bool)

    simple_real = (~synthetic) & (incoming <= 1) & (outgoing <= 1)
    before = np.maximum(previous, 0)
    after = np.maximum(following, 0)
    valid_previous = ((previous >= 0) & simple_real & simple_real[before]
                      & (points[before, 0] == points[:, 0] - 1))
    valid_next = ((following >= 0) & simple_real & simple_real[after]
                  & (points[after, 0] == points[:, 0] + 1))
    support = central_eligible(graph, shape, crop)
    eligible = (valid_previous & valid_next & support
                & support[before] & support[after])
    previous[~valid_previous] = -1
    following[~valid_next] = -1
    return previous, following, eligible


def source_labels(stem: str, graph: dict, context) -> pd.DataFrame:
    """Known, temporally consistent source labels; never a runtime node mask.

    Reuse saved C058 official node matches, preserving their original offset
    targets. Keep an eligible central node only when both actual predicted
    context neighbors also have known matches and both corresponding directed
    GT edges exist. Unknown points are ignored, never negative labels.
    The caller owns source-embryo separation and must not call this to determine
    which target-embryo nodes receive corrections.
    """
    previous, following, mask = context
    previous = np.asarray(previous, dtype=np.int64)
    following = np.asarray(following, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    ids = np.asarray(graph["ids"], dtype=np.int64)
    if any(a.shape != (len(ids),) for a in (previous, following, mask)):
        raise ValueError("Context arrays must match graph row count")
    if ((previous[mask] < 0).any() or (following[mask] < 0).any()
            or (previous[mask] >= len(ids)).any() or (following[mask] >= len(ids)).any()):
        raise ValueError("Eligible context index is missing or out of bounds")

    frame = pd.read_csv(C058_DEST / "labels" / f"{stem}.csv")
    if frame["node_id"].duplicated().any() or frame["gt_id"].duplicated().any():
        raise ValueError("Saved official labels must be one-to-one")
    row_ids = frame["row"].to_numpy(np.int64)
    if (row_ids < 0).any() or (row_ids >= len(ids)).any():
        raise ValueError("Saved label row lies outside graph")
    if not np.array_equal(ids[row_ids], frame["node_id"].to_numpy(np.int64)):
        raise ValueError("Saved C058 labels do not index this immutable graph")
    if not np.array_equal(np.asarray(graph["txyz"])[row_ids, 0], frame["t"].to_numpy()):
        raise ValueError("Saved label times do not match graph")
    mapping = dict(zip(frame["node_id"].astype(np.int64), frame["gt_id"].astype(np.int64)))
    gt_edge_path = ROOT / "data/train" / f"{stem}.geff" / "edges/ids"
    gt_edges = set(map(tuple, np.asarray(zarr.open(str(gt_edge_path), mode="r")[:],
                                        dtype=np.int64).reshape(-1, 2).tolist()))
    selected, previous_rows, next_rows = [], [], []
    for index, row in enumerate(frame.itertuples(index=False)):
        central = int(row.row)
        if not mask[central]:
            continue
        before, after = int(previous[central]), int(following[central])
        gt_before = mapping.get(int(ids[before]))
        gt_after = mapping.get(int(ids[after]))
        gt_current = int(row.gt_id)
        if (gt_before is None or gt_after is None
                or (gt_before, gt_current) not in gt_edges
                or (gt_current, gt_after) not in gt_edges):
            continue
        selected.append(index)
        previous_rows.append(before)
        next_rows.append(after)
    result = frame.iloc[selected].copy().reset_index(drop=True)
    result["previous_row"] = np.asarray(previous_rows, dtype=np.int64)
    result["next_row"] = np.asarray(next_rows, dtype=np.int64)
    result["source_gt_both_context_edges_agree"] = True
    return result


def extract_one(out, stem: str):
    """Write training crops/labels for one source movie, reading each frame once.

    Outputs:
      train/<stem>.npy: float16 (N, 3, 15, 57, 57), previous/current/next.
      train/<stem>.npz: targets (N,3; micrometres), node_ids, gt_ids, rows,
        context_rows/context_node_ids (N,3), context_center_delta_um (N,3,3).
      train/<stem>.csv/json: exact labels, counts, and extraction receipt.

    Each channel is centered on its OWN actual predicted node. The physical
    context center displacement relative to the central node is supplied as
    metadata, not silently treated as image registration. All crop support is
    real; no padding, interpolation, synthetic neighbor, or GT recentering.
    """
    start = time.perf_counter()
    out = Path(out)
    folder = out / "train"
    folder.mkdir(parents=True, exist_ok=True)
    graph = load_graph(stem)
    meta = metadata(stem)
    context = context_indices(graph, meta[1]["shape"], EXPANDED)
    labels = source_labels(stem, graph, context)
    if not len(labels):
        raise ValueError(f"No source training labels with real consistent context: {stem}")
    rows = labels["row"].to_numpy(np.int64)
    context_rows = np.stack([labels["previous_row"].to_numpy(np.int64), rows,
                             labels["next_row"].to_numpy(np.int64)], axis=1)
    points = np.asarray(graph["txyz"])
    context_points = points[context_rows]
    if not np.array_equal(context_points[:, :, 0] - points[rows, 0, None],
                          np.broadcast_to(np.array([-1, 0, 1]), (len(rows), 3))):
        raise AssertionError("Context time order drift")
    targets_um = labels[["dz_um", "dy_um", "dx_um"]].to_numpy(np.float32)
    if not np.isfinite(targets_um).all() or (np.linalg.norm(targets_um, axis=1) > 7.00001).any():
        raise AssertionError("Saved official targets must remain inside the original7um gate")
    destination = folder / f"{stem}.npy"
    crops = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16,
                                    shape=(len(labels), len(CHANNELS), *EXPANDED))
    requests = collections.defaultdict(list)
    for index in range(len(labels)):
        for channel in range(len(CHANNELS)):
            requests[int(context_points[index, channel, 0])].append((index, channel))
    for t, actions in sorted(requests.items()):
        frame = get_frame(stem, t, meta)
        for index, channel in actions:
            center = context_points[index, channel, 1:]
            block, actual_center = bounded_crop(frame, center, EXPANDED)
            if block is None or not np.array_equal(actual_center, center):
                raise AssertionError("Declared real crop support or integer center drift")
            if block[tuple(np.array(EXPANDED) // 2)] != frame[tuple(center.astype(int))]:
                raise AssertionError("Channel center pixel disagrees with original frame")
            crops[index, channel] = block
        del frame
    crops.flush()
    del crops
    context_node_ids = np.asarray(graph["ids"])[context_rows]
    context_delta_um = ((context_points[:, :, 1:] - points[rows, None, 1:])
                        * VOXEL_UM).astype(np.float32)
    if not np.equal(context_delta_um[:, 1], 0).all():
        raise AssertionError("Central context displacement must be exactly zero")
    np.savez_compressed(folder / f"{stem}.npz", targets=targets_um,
                        node_ids=np.asarray(graph["ids"])[rows],
                        gt_ids=labels["gt_id"].to_numpy(np.int64), rows=rows,
                        context_rows=context_rows, context_node_ids=context_node_ids,
                        context_center_delta_um=context_delta_um)
    labels.to_csv(folder / f"{stem}.csv", index=False)
    receipt = dict(stem=stem, embryo=stem[:4], labels=len(labels),
                   tails_gt3_5=int((labels.residual_um > 3.5).sum()),
                   crop=list(CROP), expanded=list(EXPANDED), channels=list(CHANNELS),
                   shape=[len(labels), len(CHANNELS), *EXPANDED], dtype="float16",
                   frame_reads=len(requests), source_known_only=True,
                   source_both_context_edges_agree=True, runtime_mask_uses_gt=False,
                   input_graph_sha256=sha(C058_DEST / "baseline" / f"{stem}.npz"),
                   input_labels_sha256=sha(C058_DEST / "labels" / f"{stem}.csv"),
                   crop_sha256=sha(destination),
                   targets_sha256=sha(folder / f"{stem}.npz"),
                   seconds=time.perf_counter() - start)
    save_json(folder / f"{stem}.json", receipt)
    return receipt
