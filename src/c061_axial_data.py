"""C061 direct-GT axial controls and real-anchor diagnostic data.

No model, graph edits or queue is defined here. Selection reads only GEFF
points and image metadata. Known direct points are selected independently of
intensity, C058 residual and pipeline outcome. Extraction is explicit and
separate. Noninteger GT coordinates stop preparation: no implicit rounding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from c055_guarded_readmit import sha, save_json
from c058_raw_localizer_study import (
    ROOT, DEST as C058_DEST, metadata, get_frame, load_baseline, eligible,
)
from local_registration_probe import bounded_crop

DEST = ROOT / "experiments/candidates/c061_direct_axial"
CROP = (13, 49, 49)
EXPANDED = (21, 49, 49)
Z_SHIFTS = tuple(range(-4, 5))
PER_MOVIE_CAP = 32
VOXEL_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
DIRECT_COLUMNS = ["stem", "embryo", "gt_id", "t", "center_z", "center_y", "center_x",
                  "true_z", "true_y", "true_x", "selection_sha256"]
REAL_COLUMNS = ["stem", "embryo", "node_id", "gt_id", "row", "t", "center_z", "center_y",
                "center_x", "true_z", "true_y", "true_x", "dz_vox", "dy_vox", "dx_vox",
                "dz_um", "dy_um", "dx_um", "residual_um", "target_z_class"]


def _record(path: Path, hashes: dict):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    digest = sha(path)
    if key in hashes and hashes[key] != digest:
        raise AssertionError(f"Input changed during read: {key}")
    hashes[key] = digest


def _sources(hashes):
    for filename in ["c061_axial_data.py", "c058_raw_localizer_study.py",
                     "c058_raw_localizer_model.py", "c055_guarded_readmit.py",
                     "frame_motion_audit.py", "local_registration_probe.py"]:
        _record(ROOT / "src" / filename, hashes)


def _metadata(stem, hashes):
    folder = ROOT / "data/train" / f"{stem}.zarr"
    for p in [folder / "zarr.json", folder / "0/zarr.json"]:
        _record(p, hashes)
    return metadata(stem)


def _array(path: Path, hashes):
    """Full-array GEFF read; record its metadata and every present data chunk."""
    _record(path / "zarr.json", hashes)
    for p in sorted((path / "c").rglob("*")):
        if p.is_file():
            _record(p, hashes)
    # Full slice reads every stored chunk; no edges or unrelated GT properties.
    return np.asarray(zarr.open(str(path), mode="r")[:])


def gt_points(stem: str, hashes: dict):
    """Read original GEFF ids/t/z/y/x with exact source-file receipts."""
    folder = ROOT / "data/train" / f"{stem}.geff"
    ids = _array(folder / "nodes/ids", hashes)
    values = [_array(folder / "nodes/props" / axis / "values", hashes)
              for axis in ("t", "z", "y", "x")]
    points = np.stack(values, axis=1).astype(np.float64)
    if points.shape != (len(ids), 4) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Invalid GEFF point arrays: {stem}")
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite GT coordinates: {stem}")
    return ids.astype(np.int64), points


def _support(centers, spatial_shape, crop):
    centers = np.asarray(centers)
    half = np.array(crop, dtype=np.int64) // 2
    lower = centers - half
    return ((lower >= 0) & (lower + crop <= np.array(spatial_shape))).all(axis=1)


def select(out=DEST):
    """Save manifests/counts/input hashes without reading any image pixels.

    Files: coordinate_audit.json, selection.csv, real_selection.csv,
    data_inventory.csv, data_input_hashes.json and data_plan.json. All direct
    GT coordinates must be integer before any center is cast to integer.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data_plan.json").exists():
        raise FileExistsError("Data manifests are already registered; reuse them")
    hashes = {}
    _sources(hashes)
    stems = sorted(p.stem for p in (ROOT / "data/train").glob("*.geff"))
    audit_rows, inventory, cache = [], [], {}
    for stem in stems:
        ids, points = gt_points(stem, hashes)
        meta = _metadata(stem, hashes)
        noninteger = ~np.equal(points, np.round(points))
        audit_rows.append(dict(stem=stem, embryo=stem[:4], gt_points=len(ids),
            noninteger_tzyx=noninteger.sum(axis=0).astype(int).tolist(),
            examples=[dict(gt_id=int(ids[i]), tzyx=points[i].tolist())
                      for i in np.flatnonzero(noninteger.any(axis=1))[:5]]))
        cache[stem] = (ids, points, meta)
    all_integer = not any(any(r["noninteger_tzyx"]) for r in audit_rows)
    coordinate_audit = dict(status="passed" if all_integer else "rounding_decision_required",
        movies=len(stems), points=sum(r["gt_points"] for r in audit_rows),
        all_tzyx_integer=all_integer, coordinate_policy="Exact original integer GEFF values; no rounding",
        rows=audit_rows)
    save_json(out / "coordinate_audit.json", coordinate_audit)
    save_json(out / "data_input_hashes.json", hashes)
    if not all_integer:
        raise ValueError("Noninteger GEFF coordinates found; inspect coordinate_audit.json before selecting centers")

    direct = []
    for stem in stems:
        ids, points, meta = cache[stem]
        points = points.astype(np.int64)  # equality to integers checked above
        shape = meta[1]["shape"]
        time_ok = (points[:, 0] >= 0) & (points[:, 0] < shape[0])
        support = time_ok & _support(points[:, 1:], shape[1:], EXPANDED)
        pool = []
        for row in np.flatnonzero(support):
            t, z, y, x = points[row]
            key = hashlib.sha256(f"C061:{stem}:{int(t)}:{int(ids[row])}".encode()).hexdigest()
            pool.append(dict(stem=stem, embryo=stem[:4], gt_id=int(ids[row]), t=int(t),
                center_z=int(z), center_y=int(y), center_x=int(x),
                true_z=int(z), true_y=int(y), true_x=int(x), selection_sha256=key))
        pool.sort(key=lambda r: r["selection_sha256"])
        direct.extend(pool[:PER_MOVIE_CAP])
        inventory.append(dict(stem=stem, embryo=stem[:4], gt_points=len(ids),
                              direct_supported=int(support.sum()), direct_selected=min(len(pool), PER_MOVIE_CAP)))

    real = []
    audit_path = C058_DEST / "label_audit.csv"
    _record(audit_path, hashes)
    real_stems = pd.read_csv(audit_path)["stem"].tolist()
    for stem in real_stems:
        graph_path = C058_DEST / "baseline" / f"{stem}.npz"
        label_path = C058_DEST / "labels" / f"{stem}.csv"
        _record(graph_path, hashes)
        _record(label_path, hashes)
        graph = load_baseline(C058_DEST, stem)
        labels = pd.read_csv(label_path)
        gt_ids, gt_tzyx, meta = cache[stem]
        truth = dict(zip(gt_ids.tolist(), gt_tzyx))
        support = eligible(graph, meta[1]["shape"], CROP)
        for row in labels.itertuples(index=False):
            i = int(row.row)
            if int(graph["ids"][i]) != int(row.node_id):
                raise AssertionError("Saved C058 row/node identity drift")
            if not support[i]:
                continue
            center = graph["txyz"][i]
            true = truth[int(row.gt_id)]
            if center[0] != true[0] or center[0] != row.t:
                raise AssertionError("Saved C058/GT time identity drift")
            offset = true[1:] - center[1:]
            offset_um = offset * VOXEL_UM
            saved = np.array([row.dz_um, row.dy_um, row.dx_um])
            if not np.array_equal(saved, offset_um):
                raise AssertionError("Original known-pair offsets changed")
            if not np.equal(offset, np.round(offset)).all() or abs(offset[0]) > 4:
                raise AssertionError("Real known axial target outside exact integer -4..4 support")
            real.append(dict(stem=stem, embryo=stem[:4], node_id=int(row.node_id),
                gt_id=int(row.gt_id), row=i, t=int(center[0]),
                center_z=int(center[1]), center_y=int(center[2]), center_x=int(center[3]),
                true_z=int(true[1]), true_y=int(true[2]), true_x=int(true[3]),
                dz_vox=int(offset[0]), dy_vox=int(offset[1]), dx_vox=int(offset[2]),
                dz_um=float(offset_um[0]), dy_um=float(offset_um[1]), dx_um=float(offset_um[2]),
                residual_um=float(np.linalg.norm(offset_um)), target_z_class=int(offset[0] + 4)))
    direct_df = pd.DataFrame(direct, columns=DIRECT_COLUMNS)
    real_df = pd.DataFrame(real, columns=REAL_COLUMNS)
    direct_df.to_csv(out / "selection.csv", index=False)
    real_df.to_csv(out / "real_selection.csv", index=False)
    pd.DataFrame(inventory).to_csv(out / "data_inventory.csv", index=False)
    counts = []
    for embryo in ["44b6", "6bba"]:
        d = direct_df[direct_df.embryo == embryo]
        r = real_df[real_df.embryo == embryo]
        counts.append(dict(embryo=embryo, direct_movies=int(d.stem.nunique()),
                           direct_points=len(d), shifted_training_views=len(d) * len(Z_SHIFTS),
                           real_movies=int(r.stem.nunique()), real_points=len(r),
                           real_tails_gt3_5=int((r.residual_um > 3.5).sum()),
                           real_axial_classes={str(int(k)): int(v) for k, v in r.target_z_class.value_counts().sort_index().items()}))
    plan = dict(status="manifests_prepared_no_pixel_extraction", gt_movies=len(stems),
        crop=list(CROP), expanded=list(EXPANDED), z_shifts=list(Z_SHIFTS), per_movie_cap=PER_MOVIE_CAP,
        direct_selection="Full21x49x49 real support, then SHA256(C061:stem:t:gt_id) first32 per movie; no intensity/residual outcome selection",
        direct_labels="Crop-center shift s means true target z offset -s; class index -s+4",
        coordinate_policy="All original GEFF t,z,y,x exactly integer; cast only after complete audit, no rounding",
        real_selection="All original C058 known pairs with nonsynthetic13x49x49 support; no continuity or stable-track filter",
        real_stems=real_stems, counts=counts,
        direct_manifest_sha256=sha(out / "selection.csv"),
        real_manifest_sha256=sha(out / "real_selection.csv"),
        source_hashes={k: v for k, v in hashes.items() if k.startswith("src")},
        limitations=["Two biological embryos with overlapping crops; selected points are not independent biological samples.",
                     "Direct known labels do not label surrounding unannotated cells as background.",
                     "Real-anchor diagnostic is paired to unchanged original official identities and is not a graph modification."])
    save_json(out / "data_input_hashes.json", hashes)
    save_json(out / "data_plan.json", plan)
    return plan


def load_manifest(out, kind):
    if kind not in ("direct", "real"):
        raise ValueError("Expected direct or real manifest")
    out = Path(out)
    plan = json.loads((out / "data_plan.json").read_text(encoding="utf-8"))
    path = out / ("selection.csv" if kind == "direct" else "real_selection.csv")
    if sha(path) != plan[f"{kind}_manifest_sha256"]:
        raise AssertionError("Registered manifest changed")
    return pd.read_csv(path)


def shifted_direct_crop(expanded, shift_z):
    """Exact view around GT+shift_z; no interpolation or additional image IO."""
    shift_z = int(shift_z)
    if shift_z not in Z_SHIFTS or tuple(expanded.shape[-3:]) != EXPANDED:
        raise ValueError("Unexpected expanded crop or axial shift")
    start = (EXPANDED[0] - CROP[0]) // 2 + shift_z
    return expanded[..., start:start + CROP[0], :, :]


def extract_one(out, stem, kind):
    """Explicit pixel extraction; select() never calls this function.

    direct/<stem>.npy is float16 N×21×49×49 (nine13-plane views are sliced).
    real/<stem>.npy is float16 N×13×49×49. Companion NPZ preserves all target,
    center and identity fields; receipt JSON tracks read source/chunk hashes.
    """
    out = Path(out)
    manifest = load_manifest(out, kind)
    selected = manifest[manifest.stem == stem].copy().reset_index(drop=True)
    if not len(selected):
        raise ValueError(f"No selected {kind} rows: {stem}")
    shape = EXPANDED if kind == "direct" else CROP
    folder = out / kind
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    _sources(hashes)
    manifest_path = out / ("selection.csv" if kind == "direct" else "real_selection.csv")
    _record(manifest_path, hashes)
    meta = _metadata(stem, hashes)
    registered = json.loads((out / "data_input_hashes.json").read_text(encoding="utf-8"))
    for path, digest in hashes.items():
        if path in registered and registered[path] != digest:
            raise AssertionError(f"Registered input changed: {path}")
    destination = folder / f"{stem}.npy"
    if destination.exists():
        raise FileExistsError(f"Do not overwrite extracted crops: {destination}")
    start_time = time.perf_counter()
    crops = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16,
                                    shape=(len(selected), *shape))
    control_count = 0
    frame_paths = []
    for t, rows in selected.groupby("t", sort=True):
        chunk = meta[0] / "0/c" / str(int(t)) / "0/0/0"
        _record(chunk, hashes)
        frame_paths.append(str(chunk.relative_to(ROOT)))
        frame = get_frame(stem, int(t), meta)
        for i in rows.index:
            center = selected.loc[i, ["center_z", "center_y", "center_x"]].to_numpy(np.int64)
            block, actual = bounded_crop(frame, center, shape)
            if block is None or not np.array_equal(actual, center):
                raise AssertionError("Manifest real-support or integer-center drift")
            # Compare saved float16 to original pixels converted to that SAME dtype.
            # A float16 saved value is not expected to equal its float32 source.
            expected = block.astype(np.float16)
            crops[i] = expected
            if not np.array_equal(crops[i], expected):
                raise AssertionError("Saved raw crop differs after declared float16 conversion")
            control_count += 1
            if kind == "direct":
                for shift in Z_SHIFTS:
                    shifted_center = center + np.array([shift, 0, 0])
                    direct, verified_center = bounded_crop(frame, shifted_center, CROP)
                    if direct is None or not np.array_equal(verified_center, shifted_center):
                        raise AssertionError("Expanded direct crop lacks a shifted view")
                    if not np.array_equal(shifted_direct_crop(crops[i], shift), direct.astype(np.float16)):
                        raise AssertionError("Axial shifted view disagrees with original normalized pixels")
                    control_count += 1
        del frame
    crops.flush()
    del crops
    center = selected[["center_z", "center_y", "center_x"]].to_numpy(np.int64)
    truth = selected[["true_z", "true_y", "true_x"]].to_numpy(np.int64)
    payload = dict(gt_ids=selected.gt_id.to_numpy(np.int64), times=selected.t.to_numpy(np.int64),
                   center_zyx=center, true_zyx=truth, targets_um=((truth - center) * VOXEL_UM).astype(np.float32))
    if kind == "direct":
        if not np.array_equal(center, truth):
            raise AssertionError("Direct center is not the exact known GT point")
        shifts = np.array(Z_SHIFTS, np.int64)
        payload.update(shift_z_vox=shifts, target_z_vox=-shifts, target_z_class=-shifts + 4,
                       selection_sha256=selected.selection_sha256.to_numpy(dtype=str))
    else:
        payload.update(node_ids=selected.node_id.to_numpy(np.int64), rows=selected.row.to_numpy(np.int64),
                       target_z_vox=selected.dz_vox.to_numpy(np.int64),
                       target_z_class=selected.target_z_class.to_numpy(np.int64),
                       original_residual_um=selected.residual_um.to_numpy(np.float64))
    np.savez_compressed(folder / f"{stem}.npz", **payload)
    selected.to_csv(folder / f"{stem}.csv", index=False)
    receipt = dict(status="passed", stem=stem, embryo=stem[:4], kind=kind,
        points=len(selected), shape=[len(selected), *shape], dtype="float16",
        frames_read=len(frame_paths), image_chunks=frame_paths, input_hashes=hashes,
        raw_float16_pixel_controls=control_count, graph_modified=False,
        source_manifest_sha256=sha(manifest_path),
        crop_sha256=sha(destination), target_sha256=sha(folder / f"{stem}.npz"),
        seconds=time.perf_counter() - start_time)
    save_json(folder / f"{stem}.json", receipt)
    return receipt


def input_paths(out=DEST):
    """Exact dependencies plus image chunks that selected extraction will read.

    This enumerates chunk paths without reading pixels. The caller may hash
    this finite set when registering its queue. Extraction also records the
    hashes of the actual chunks it reads in each output receipt.
    """
    out = Path(out)
    hashes = json.loads((out / "data_input_hashes.json").read_text(encoding="utf-8"))
    paths = {ROOT / p for p in hashes}
    paths.update(out / p for p in ["selection.csv", "real_selection.csv", "data_plan.json",
                                   "coordinate_audit.json", "data_inventory.csv", "data_input_hashes.json"])
    for kind in ["direct", "real"]:
        frame = load_manifest(out, kind)
        for stem, t in frame[["stem", "t"]].drop_duplicates().itertuples(index=False, name=None):
            paths.add(ROOT / "data/train" / f"{stem}.zarr" / "0/c" / str(int(t)) / "0/0/0")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("A declared selected-data input is missing")
    return sorted(paths)


def extract(out=DEST, stems=None):
    """Extract both registered data kinds for all or selected movie stems."""
    out = Path(out)
    wanted = None if stems is None else set(stems)
    receipts = []
    for kind in ["direct", "real"]:
        frame = load_manifest(out, kind)
        for stem in sorted(frame.stem.unique()):
            if wanted is None or stem in wanted:
                receipts.append(extract_one(out, stem, kind))
    return receipts


# Explicit backwards-compatible descriptive alias; neither function extracts.
prepare = select
