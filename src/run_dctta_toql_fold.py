from __future__ import annotations

import argparse
import gc
import json
import runpy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "artifacts" / "stabledet_code"
RUNTIME = ROOT / "artifacts" / "stabledet_runtime"
CKPTS = ROOT / "artifacts" / "stabledet_hoct_hn"
DATA = ROOT / "data" / "visible_test" / "test"
BACKBONE = ROOT / "experiments" / "metric_dctta0965"
GT = ROOT / "data" / "visible_gt" / "train"


def official_repo() -> Path:
    roots = sorted((ROOT / "vendor" / "royerlab_metric").glob("source-*/kaggle-cell-tracking-competition-*"))
    if not roots:
        raise FileNotFoundError("vendored official repo not found")
    return roots[0]


def configure_paths() -> None:
    official = official_repo()
    paths = [
        CODE,
        RUNTIME / "src",
        RUNTIME / "scripts",
        RUNTIME / "methods" / "sparse_point_query_lineage",
        official / "src",
        official / "scripts",
    ]
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def build_cache(fold: int, out_root: Path) -> None:
    from cache_official_hoct_features import PredictConfig, cache_sample, load_model

    weight = RUNTIME / f"fold{fold}" / "edge_predictor_best.pth"
    config = json.loads((RUNTIME / f"fold{fold}" / "config.json").read_text(encoding="utf-8"))
    toql = torch.load(CKPTS / f"fold{fold}_toql_9294.pt", map_location="cpu", weights_only=True)
    candidate_threshold = float(toql["candidate_threshold"])
    gate_um = float(toql["gate_um"])
    device = torch.device("cuda:0")
    model, window_size, downsample = load_model(weight, device)
    cfg = PredictConfig(
        det_threshold=candidate_threshold,
        pool_kernel_um=float(config["pool_kernel_um"]),
    )
    query_cache = out_root / "query_cache"
    query_cache.mkdir(parents=True, exist_ok=True)

    stems = (
        "44b6_0113de3b",
        "44b6_0b24845f",
        "6bba_05b6850b",
        "6bba_05db0fb1",
    )
    summaries = []
    for stem in stems:
        sample = DATA / f"{stem}.zarr"
        query_output = query_cache / f"{stem}.pt"
        if query_output.exists():
            print(f"fold{fold} reuse cache {stem}", flush=True)
            continue
        summary = cache_sample(
            sample,
            query_output,
            model,
            device,
            cfg,
            window_size,
            downsample,
            gate_um=gate_um,
            match_um=7.0,
            fork_topk=8,
            node_graph_path=None,
            inference_only=True,
            include_det_logits=True,
            compact_query_cache=True,
            query_output_path=None,
        )
        summaries.append(summary)
        print(f"fold{fold} cached {stem}: {summary}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    (out_root / "cache_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_script(path: Path, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [str(path), *argv]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    configure_paths()

    out_root = ROOT / "experiments" / f"dctta_toql_fold{args.fold}"
    out_root.mkdir(parents=True, exist_ok=True)
    build_cache(args.fold, out_root)

    toql_dir = out_root / "toql_graphs"
    run_script(
        CODE / "infer_track_object_query_lineage.py",
        [
            "--cache-dir", str(out_root / "query_cache"),
            "--checkpoint", str(CKPTS / f"fold{args.fold}_toql_9294.pt"),
            "--output-dir", str(toql_dir),
            "--continuation-deficit",
        ],
    )

    merged = out_root / "merged_dctta_toql"
    run_script(
        CODE / "merge_hn_backbone_toql_bridges.py",
        [
            "--hn-dir", str(BACKBONE),
            "--toql-dir", str(toql_dir),
            "--output-dir", str(merged),
            "--gt-dir", str(GT),
        ],
    )


if __name__ == "__main__":
    main()
