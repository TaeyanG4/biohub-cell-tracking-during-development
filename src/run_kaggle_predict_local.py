#!/usr/bin/env python3
"""Run a Kaggle run's patched predict script locally (inference + ILP) on train movies.

Uses the ``tracking_repo`` a harmonic-fusion-lineage Kaggle run leaves in
/kaggle/working (e.g. ``tmp/c011_output/tracking_repo``), the same public pilkwang
weights, the notebook's literal BIOHUB_* settings and the notebook's CLI arguments,
and collects per movie:

    <out>/<label>/predictions/<stem>.geff   ILP graph  -> src/eval_pp_variants_local.py --pred-root
    <out>/<label>/edge_cache/<stem>.npz     low-detection dump -> its --lowdet-dir (readmit / gap filler)

The V1284 hook runs in ``--v1284-mode zero`` (x138 minus head, = C011) or
``candidate`` with ``--v1284-head <file>`` (a head trained by src/v1284_head_train.py).
Run with the global Python (torch + tracksdata + pyscipopt).

    python src/run_kaggle_predict_local.py --repo tmp/c011_output/tracking_repo \
        --stems-file experiments/candidates/c012_v1284_head/heldout_stems.txt \
        --out experiments/candidates/c012_v1284_head/e2e --label zero --v1284-mode zero
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1284_capture_local import (  # noqa: E402
    DEFAULT_DATA_DIR, PRIMARY_WEIGHTS, REFERENCE_NOTEBOOK, SECONDARY_WEIGHTS, notebook_env, prepare_repo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stems", default="")
    parser.add_argument("--stems-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True, help="run name; also the predict --method")
    parser.add_argument("--notebook", type=Path, default=REFERENCE_NOTEBOOK)
    parser.add_argument("--v1284-mode", choices=("zero", "candidate", "output"), default="zero",
                        help="output = refine only the emitted coordinates (src/v1284_patch_variants.py)")
    parser.add_argument("--v1284-head", type=Path, default=None)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="override a BIOHUB_* setting after the notebook's literal values (e.g. BIOHUB_CACHE_EDGE_THRESHOLD=0.02)")
    parser.add_argument("--t4-fp32", action="store_true",
                        help="disable Ada TF32 and use math SDPA (Windows short-window attention fix); log runtime details")
    args = parser.parse_args()

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if args.stems_file:
        stems += [s.strip() for s in args.stems_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    head_paths = [Path(h) for h in str(args.v1284_head).split(";") if h] if args.v1284_head else []
    if args.v1284_mode != "zero" and not (head_paths and all(h.is_file() for h in head_paths)):
        raise SystemExit(f"--v1284-mode {args.v1284_mode} needs existing --v1284-head file(s) (';'-separated for an ensemble)")

    run_dir = (args.out / args.label).resolve()
    work = run_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    repo = prepare_repo(args.repo.resolve(), work, args.notebook)
    if args.t4_fp32:
        script = repo / "scripts" / "predict_unet_transformer.py"
        source = script.read_text(encoding="utf-8")
        anchor = "import torch\n"
        if source.count(anchor) != 1:
            raise SystemExit("T4 FP32 runtime: torch import anchor is not unique")
        block = '''import torch
# Local Ada -> T4-like numerical policy; does not claim bitwise GPU equivalence.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = False
print('LOCAL_RUNTIME', {'torch': torch.__version__, 'cuda': torch.version.cuda,
      'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
      'cudnn_tf32': torch.backends.cudnn.allow_tf32,
      'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
      'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
      'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
      'math_sdp': torch.backends.cuda.math_sdp_enabled()}, flush=True)
'''
        source = source.replace(anchor, block, 1)
        compile(source, str(script), "exec")
        script.write_text(source, encoding="utf-8")
    if args.v1284_mode == "output":
        from v1284_patch_variants import output_mode_module, output_mode_script

        module = repo / "scripts" / "v1284_coordinate_refinement.py"
        module.write_text(output_mode_module(module.read_text(encoding="utf-8")), encoding="utf-8")
        script = repo / "scripts" / "predict_unet_transformer.py"
        script.write_text(output_mode_script(script.read_text(encoding="utf-8")), encoding="utf-8")
    splits = work / "splits.json"
    splits.write_text(json.dumps([{"split": 0, "train": [], "test": stems}], indent=2), encoding="utf-8")

    nb_env = notebook_env(args.notebook)
    for item in args.env:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        nb_env[key] = value
    env = {**os.environ, **nb_env}
    env.pop("BIOHUB_DIAGNOSTIC_ARM", None)
    env.update({
        "PYTHONPATH": "src",
        "PYTHONIOENCODING": "utf-8",
        "BIOHUB_SECONDARY_WEIGHTS": str(SECONDARY_WEIGHTS),
        "BIOHUB_CACHE_DIR": str(run_dir / "edge_cache"),
        "BIOHUB_LOCAL_WORKING": str(work),
        "V1284_MODE": args.v1284_mode,
    })
    if args.v1284_mode != "zero":
        env["V1284_HEAD"] = ";".join(str(h.resolve()) for h in head_paths)  # several = head ensemble (module averages the shifts)

    cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", str(args.data_dir.resolve()), "--splits", str(splits), "--split", "0",
        "--weights", str(PRIMARY_WEIGHTS), "--unet-batch-size", nb_env.get("BIOHUB_UNET_BATCH_SIZE", "4"),
        "--det-threshold", nb_env.get("BIOHUB_DET_THRESHOLD", "0.965"),
        "--ilp-edge-weight", nb_env.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"),
        "--ilp-appearance-weight", nb_env.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.0"),
        "--ilp-disappearance-weight", nb_env.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "2"),
        "--ilp-division-weight", nb_env.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.2"),
        "--use-ilp", "--method", args.label,
    ]
    t0 = time.time()
    with (run_dir / "predict.log").open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    produced = sorted((repo / "predictions").rglob(f"{args.label}/split_0/*.geff"))
    target = run_dir / "predictions"
    target.mkdir(exist_ok=True)
    for geff in produced:
        dest = target / geff.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(geff), str(dest))
    missing = sorted(set(stems) - {p.stem for p in target.glob("*.geff")})
    print(f"{args.label}: {len(stems) - len(missing)}/{len(stems)} graphs in {target} "
          f"({(time.time() - t0) / 60:.1f} min); missing={missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
