#!/usr/bin/env python3
"""Capture V1284-style coordinate-head training pairs with the exact Kaggle inference code.

x138's V1284 module refines every first-seen fused detection with a tiny MLP over
frozen primary-UNet features (224-d = centre feature + 6 neighbour differences).
Its trained weights are private, so this tool rebuilds the training set:

1. copies a patched ``tracking_repo`` (the support-pack scripts a Kaggle run of the
   harmonic-fusion lineage leaves in /kaggle/working) and, if the V1284 hook is not
   there yet, inserts it exactly as x138's cell 4 does;
2. runs that ``predict_video`` (dual-seed fused detection, 8-view edge-feature TTA,
   same public pilkwang weights as Kaggle) on local train movies with
   ``V1284_MODE='capture'``;
3. matches every captured frame's detections one-to-one to GT (Hungarian, <= --match-um)
   and keeps only matched pairs per movie:
   ``features (N, 224) float32``, ``offset_um (N, 3) = GT - detection``, ids and coords.

Environment settings are copied from the literal ``os.environ`` assignments of the
reference notebook (default: C011 = x138 settings), so detection and features follow
the Kaggle configuration. Run with the global Python (torch + tracksdata).

    python src/v1284_capture_local.py --repo tmp/c011_output/tracking_repo \
        --stems-file experiments/candidates/c012_v1284_head/capture_stems.txt \
        --out experiments/candidates/c012_v1284_head/pairs
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_WEIGHTS = REPO_ROOT / "artifacts" / "pilkwang_support50" / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
SECONDARY_WEIGHTS = REPO_ROOT / "artifacts" / "pilkwang_temporal_seed314159" / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
REFERENCE_NOTEBOOK = REPO_ROOT / "experiments" / "candidates" / "c011_x138_zero" / "biohub-c011-x138-zero.ipynb"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "train"

DOWNSAMPLED_UM = 1.625                                # grid spacing of detections: z 1.625, y/x 4 x 0.40625
VOXEL_UM = np.array([1.625, 0.40625, 0.40625])        # full-resolution GT voxel size (z, y, x)
ENV_ASSIGN = re.compile(r"""^os\.environ\[["'](BIOHUB_[A-Z0-9_]+)["']\]\s*=\s*["']([^"']*)["']""", re.M)
REFINE_ANCHOR = "                coord_offset[t] = (global_node_count, global_node_count + len(arr))"


def code_cells(notebook: Path) -> list[str]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def notebook_env(notebook: Path) -> dict[str, str]:
    """Literal BIOHUB_* settings of cells 0-4 (config + inference), last assignment wins."""
    env: dict[str, str] = {}
    for src in code_cells(notebook)[:5]:
        env.update(ENV_ASSIGN.findall(src))
    return env


def embedded_v1284_module(notebook: Path) -> str:
    for node in ast.walk(ast.parse(code_cells(notebook)[4])):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
                and "v1284_coordinate_refinement.py" in ast.unparse(node.func.value)):
            return node.args[0].value
    raise SystemExit(f"{notebook}: v1284_coordinate_refinement.py source not found in cell 4")


def prepare_repo(source_repo: Path, work_dir: Path, notebook: Path) -> Path:
    repo = work_dir / "tracking_repo"
    if repo.exists():
        shutil.rmtree(repo)
    for part in ("scripts", "src"):
        shutil.copytree(source_repo / part, repo / part, ignore=shutil.ignore_patterns("__pycache__"))
    script = repo / "scripts" / "predict_unet_transformer.py"
    text = script.read_text(encoding="utf-8")
    # Diagnostic jsonl logs only; keep them out of a non-existent /kaggle/working.
    for quoted in ("Path('/kaggle/working')", 'Path("/kaggle/working")'):
        text = text.replace(quoted, "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    if "_v1284_refine(" not in text:
        (repo / "scripts" / "v1284_coordinate_refinement.py").write_text(embedded_v1284_module(notebook), encoding="utf-8")
        if text.count("import tracksdata as td\n") != 1 or text.count(REFINE_ANCHOR) != 1:
            raise SystemExit("V1284 hook anchors not unique in the source repo's predict script")
        text = text.replace("import tracksdata as td\n", "import tracksdata as td\n"
                            "from v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index\n", 1)
        text = text.replace(REFINE_ANCHOR, "                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n" + REFINE_ANCHOR, 1)
    script.write_text(text, encoding="utf-8")
    return repo


def load_gt(gt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(gt_path)
    graph = graph[0] if isinstance(graph, tuple) else graph
    nodes = graph.node_attrs()
    t = nodes["t"].to_numpy().astype(np.int64)
    zyx = np.stack([nodes["z"].to_numpy(), nodes["y"].to_numpy(), nodes["x"].to_numpy()], axis=1).astype(np.float64)
    return t, zyx * VOXEL_UM, nodes["node_id"].to_numpy().astype(np.int64)


def match_movie(capture_dir: Path, gt_path: Path, match_um: float) -> dict[str, np.ndarray]:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    gt_t, gt_um, gt_ids = load_gt(gt_path)
    out = {k: [] for k in ("features", "offset_um", "det_coords", "gt_node_id", "t")}
    n_det = 0
    for frame_file in sorted(capture_dir.glob("*.npz")):
        with np.load(frame_file) as z:
            coords = z["coords"].astype(np.int64)
            feats = z["features"].astype(np.float32)
        n_det += len(coords)
        t = int(coords[0, 0])
        sel = np.nonzero(gt_t == t)[0]
        if not len(sel):
            continue
        det_um = coords[:, 1:].astype(np.float64) * DOWNSAMPLED_UM
        cost = cdist(det_um, gt_um[sel])
        gated = np.where(cost <= match_um, cost, 1e6)
        rows, cols = linear_sum_assignment(gated)
        keep = gated[rows, cols] < 1e6
        rows, cols = rows[keep], cols[keep]
        out["features"].append(feats[rows])
        out["offset_um"].append((gt_um[sel][cols] - det_um[rows]).astype(np.float32))
        out["det_coords"].append(coords[rows].astype(np.int16))
        out["gt_node_id"].append(gt_ids[sel][cols])
        out["t"].append(np.full(len(rows), t, dtype=np.int16))
    arrays = {k: (np.concatenate(v) if v else np.empty((0,), np.float32)) for k, v in out.items()}
    arrays["n_detections"] = np.array(n_det)
    arrays["n_gt_nodes"] = np.array(len(gt_t))
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="patched tracking_repo (scripts/ + src/) from a Kaggle run")
    parser.add_argument("--stems", default="", help="comma-separated train stems")
    parser.add_argument("--stems-file", type=Path, default=None, help="one stem per line")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, required=True, help="directory for <stem>.npz matched pairs")
    parser.add_argument("--work", type=Path, default=None, help="scratch dir (default: <out>/_work)")
    parser.add_argument("--notebook", type=Path, default=REFERENCE_NOTEBOOK, help="source of BIOHUB_* settings")
    parser.add_argument("--match-um", type=float, default=4.0)
    parser.add_argument("--keep-raw", action="store_true", help="keep per-frame capture files")
    parser.add_argument("--no-tf32", action="store_true",
                        help="disable TF32 convolutions/matmuls (Ada default for cuDNN) to match the T4's plain FP32")
    args = parser.parse_args()

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if args.stems_file:
        stems += [s.strip() for s in args.stems_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not stems:
        raise SystemExit("no stems given")
    work = (args.work or args.out / "_work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    repo = prepare_repo(args.repo.resolve(), work, args.notebook)

    env = notebook_env(args.notebook)
    env.update({
        "BIOHUB_SECONDARY_WEIGHTS": str(SECONDARY_WEIGHTS),
        "BIOHUB_CACHE_DIR": "",
        "BIOHUB_LOCAL_WORKING": str(work),
        "V1284_MODE": "capture",
        "V1284_CAPTURE": str(work / "capture"),
    })
    env.pop("BIOHUB_DIAGNOSTIC_ARM", None)
    os.environ.update(env)
    os.environ.pop("BIOHUB_DIAGNOSTIC_ARM", None)
    sys.path[:0] = [str(repo / "scripts"), str(repo / "src")]

    import torch
    import predict_unet_transformer as pu

    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window, downsample = pu.load_model(PRIMARY_WEIGHTS, device)
    secondary, window2, downsample2 = pu.load_model(SECONDARY_WEIGHTS, device)
    if (window2, tuple(downsample2)) != (window, tuple(downsample)):
        raise SystemExit("primary/secondary grids differ")
    cfg = pu.PredictConfig(
        det_threshold=float(env.get("BIOHUB_DET_THRESHOLD", "0.965")),
        use_ilp=True,
        ilp_edge_weight=float(env.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0")),
        ilp_appearance_weight=float(env.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.0")),
        ilp_disappearance_weight=float(env.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "2")),
        ilp_division_weight=float(env.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.2")),
    )
    cfg.threshold = float(env["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"])
    print(f"device={device} window={window} downsample={downsample} det_threshold={cfg.det_threshold} "
          f"edge_threshold={cfg.threshold} stems={len(stems)}", flush=True)

    for stem in stems:
        target = args.out / f"{stem}.npz"
        if target.exists():
            print(f"{stem}: exists, skipped", flush=True)
            continue
        capture_dir = work / "capture" / stem
        if capture_dir.exists():
            shutil.rmtree(capture_dir)
        t0 = time.time()
        pu.predict_video(
            model, args.data_dir / stem, device, cfg=cfg, window_size=window, unet_batch_size=4,
            downsample=downsample, secondary_model=secondary,
            secondary_edge_weight=float(env["BIOHUB_SECONDARY_EDGE_WEIGHT"]),
            secondary_detection_weight=float(env["BIOHUB_SECONDARY_DETECTION_WEIGHT"]),
            secondary_link_mode=env["BIOHUB_SECONDARY_LINK_MODE"],
            secondary_mix_temperature=float(env["BIOHUB_SECONDARY_MIX_TEMPERATURE"]),
            secondary_low_margin_max=float(env["BIOHUB_SECONDARY_LOW_MARGIN_MAX"]),
        )
        t_pred = time.time() - t0
        arrays = match_movie(capture_dir, args.data_dir / f"{stem}.geff", args.match_um)
        np.savez_compressed(target, **arrays)
        if not args.keep_raw:
            shutil.rmtree(capture_dir)
        n = len(arrays["offset_um"])
        mean_err = float(np.linalg.norm(arrays["offset_um"], axis=1).mean()) if n else float("nan")
        print(f"{stem}: detections={int(arrays['n_detections'])} gt={int(arrays['n_gt_nodes'])} pairs={n} "
              f"mean|offset|={mean_err:.3f}um predict={t_pred:.0f}s total={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
