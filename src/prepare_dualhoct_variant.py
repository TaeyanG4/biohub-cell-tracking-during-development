from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "exp_dctta_hoct_det0965"
DST = ROOT / "experiments" / "exp_dctta_dualhoct_det0965"
SRC_NB = SRC / "biohub-dctta-hoct-det0965.ipynb"
DST_NB = DST / "biohub-dctta-dualhoct-det0965.ipynb"


APPEND = r'''

# ==== Dual-HOCT consensus override: shared graph, v0 ∩ v1 ============================
# This block intentionally overrides only the HOCT model loader, runtime estimate, and
# `_hv_hoct_pairs`.  The upstream DCTTA detector/linker and the veto application logic stay
# unchanged.  Candidate graph/regionprops construction is shared between v0 and v1; each model
# gets an independent graph copy before model_predict mutates similarity/orphan attributes.
_HV_STATE["model_v1"] = None
_HV_SEC_PER_1000_NODES = 13.5
_HV_FIXED_S = 15.0


def _hv_install_and_load_dual():
    if _HV_STATE.get("model") is not None and _HV_STATE.get("model_v1") is not None:
        return _HV_STATE["model"], _HV_STATE["model_v1"]
    wheel_dir = _hv_os.path.dirname(_hv_find("hoct-0.2.0-py3-none-any.whl"))
    weights_v0 = _hv_find("general_v0.pt")
    weights_v1 = _hv_find("general_v1.pt")
    cmd = [_hv_sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
           "--find-links", wheel_dir, "hoct==0.2.0", "spatial-graph==0.1.1", "pooch==1.9.0"]
    _hv_log("installing offline HOCT wheels from " + wheel_dir)
    _hv_subprocess.run(cmd, check=True)
    import importlib
    importlib.invalidate_caches()
    import hoct  # noqa: E402
    from hoct import load_model
    model_v0 = load_model(weights_v0, device="cuda")
    model_v1 = load_model(weights_v1, device="cuda")
    _hv_log(
        f"hoct {hoct.__version__} loaded dual consensus: "
        f"v0={_hv_os.path.basename(weights_v0)} v1={_hv_os.path.basename(weights_v1)}"
    )
    _HV_STATE["model"] = model_v0
    _HV_STATE["model_v1"] = model_v1
    return model_v0, model_v1


def _hv_solution_pairs(sol, det_sub, ids_sub, global_start: int, own_end: int) -> set:
    nodes = sol.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = sol.edge_attrs(attr_keys=[])
    t_loc = nodes["t"].to_numpy().astype(int)
    zyx = _hv_np.stack([nodes[c].to_numpy() for c in ("z", "y", "x")], 1).astype(float)
    snap = _hv_snap_to_nodes(t_loc, zyx, det_sub)
    node_ids = nodes["node_id"].to_list()
    hoct_to_pid = {int(n): int(ids_sub[k]) for n, k in zip(node_ids, snap)}
    src_t = {int(n): int(tt) + global_start for n, tt in zip(node_ids, t_loc)}
    pairs = set()
    for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list()):
        if global_start <= src_t[int(a)] < own_end:
            pairs.add((hoct_to_pid[int(a)], hoct_to_pid[int(b)]))
    return pairs


def _hv_predict_pairs_dual(model_v0, model_v1, labels, images, det, ids, n_chunks: int) -> set:
    import torch
    from hoct import predict
    from hoct.features import create_graph
    from tracksdata.functional import TilingScheme

    T = labels.shape[0]
    starts = [round(i * T / n_chunks) for i in range(n_chunks)] + [T]
    consensus = set()
    for i in range(n_chunks):
        s, e_own = starts[i], starts[i + 1]
        e_frames = min(T, e_own + 1)
        sub = _hv_np.where((det[:, 0] >= s) & (det[:, 0] < e_frames))[0]
        det_sub = det[sub].copy()
        det_sub[:, 0] -= s
        graph = create_graph(
            labels=labels[s:e_frames],
            images=None if images is None else images[s:e_frames],
            gt_graph=None,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=_HV_MAX_DELTA_T,
            scale=(1.0, *_HV_SCALE_ZYX),
        )
        graph_v0 = graph.copy()
        graph_v1 = graph.copy()
        del graph
        scheme = TilingScheme(tile_shape=_HV_TILE, overlap_shape=_HV_OVERLAP)
        with torch.inference_mode():
            sol_v0 = predict(model_v0, graph=graph_v0, max_delta_t=_HV_MAX_DELTA_T, tiling_scheme=scheme)
        p0 = _hv_solution_pairs(sol_v0, det_sub, ids[sub], s, e_own)
        del sol_v0, graph_v0
        torch.cuda.empty_cache()
        with torch.inference_mode():
            sol_v1 = predict(model_v1, graph=graph_v1, max_delta_t=_HV_MAX_DELTA_T, tiling_scheme=scheme)
        p1 = _hv_solution_pairs(sol_v1, det_sub, ids[sub], s, e_own)
        del sol_v1, graph_v1
        torch.cuda.empty_cache()
        consensus.update(p0 & p1)
    return consensus


def _hv_hoct_pairs(dataset: str, nodes_by_id: dict) -> set:
    ids = _hv_np.array(sorted(nodes_by_id), dtype=int)
    det = _hv_np.array([[float(nodes_by_id[i]["t"]), float(nodes_by_id[i]["z"]),
                         float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])] for i in ids])
    key = ("dual_v0v1", dataset, _hv_hashlib.sha256(det.tobytes()).hexdigest())
    if key in _HV_STATE["cache"]:
        _hv_log(f"[{dataset}] dual-HOCT consensus reused from cache")
        return _HV_STATE["cache"][key]
    model_v0, model_v1 = _hv_install_and_load_dual()
    t0 = _hv_time.time()
    volume = _hv_read_volume(dataset)
    labels = _hv_rasterize_spheres(det, volume.shape)
    _hv_log(
        f"[{dataset}] {len(ids)} final nodes -> shared spheres r={_HV_RADIUS_UM} um, "
        f"volume {volume.shape} read+painted in {_hv_time.time() - t0:.0f}s"
    )
    pairs = None
    deadline_s = _HV_DEADLINE_H * 3600.0
    for n_chunks in (1, 2, 4):
        if n_chunks > 1:
            elapsed = _hv_notebook_elapsed_s()
            if elapsed + _hv_estimate_s(len(ids)) > deadline_s:
                raise RuntimeError(
                    f"dual-HOCT retry with {n_chunks} chunks refused: elapsed {elapsed:.0f}s plus "
                    f"the predicted {_hv_estimate_s(len(ids)):.0f}s would pass the {deadline_s:.0f}s deadline"
                )
        try:
            t1 = _hv_time.time()
            pairs = _hv_predict_pairs_dual(model_v0, model_v1, labels, volume, det, ids, n_chunks)
            _hv_log(
                f"[{dataset}] dual-HOCT consensus proposed {len(pairs)} edges in "
                f"{_hv_time.time() - t1:.0f}s (time chunks={n_chunks})"
            )
            break
        except RuntimeError as exc:
            _hv_log(
                f"[{dataset}] dual-HOCT with {n_chunks} chunk(s) failed: "
                f"{str(exc)[:160]}; retrying with more chunks"
            )
            _hv_release_gpu()
    if pairs is None:
        raise RuntimeError(f"DUAL_HOCT_VETO_FAILED {dataset}: no consensus solution")
    _HV_STATE["cache"][key] = pairs
    return pairs
'''


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    old_nb = DST / SRC_NB.name
    if old_nb != DST_NB:
        old_nb.rename(DST_NB)

    nb = json.loads(DST_NB.read_text(encoding="utf-8"))
    matches = []
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "# ==== HOCT consensus veto (post-ILP stage): arming cell" in src:
            matches.append(i)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one HOCT arming cell, found {matches}")
    idx = matches[0]
    src = "".join(nb["cells"][idx]["source"])
    if "Dual-HOCT consensus override" in src:
        raise RuntimeError("Dual-HOCT override already present")
    src += APPEND
    nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    DST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta_path = DST / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = "taeyangg4/biohub-dctta-dualhoct-det0965"
    meta["title"] = "Biohub DCTTA DualHOCT DET0965"
    meta["code_file"] = DST_NB.name
    datasets = list(meta.get("dataset_sources", []))
    v1_ref = "taeyangg4/biohub-hoct-general-v1-official"
    if v1_ref not in datasets:
        datasets.append(v1_ref)
    meta["dataset_sources"] = datasets
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(DST)
    print("cell", idx, "chars", len(src))
    print("dataset_sources", meta["dataset_sources"])


if __name__ == "__main__":
    main()
