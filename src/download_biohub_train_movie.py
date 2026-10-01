from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "biohub-cell-tracking-during-development"


def chunk_keys(meta: dict) -> list[str]:
    shape = list(meta["shape"])
    chunk_shape = list(meta["chunk_grid"]["configuration"]["chunk_shape"])
    grid = [range(math.ceil(s / c)) for s, c in zip(shape, chunk_shape)]
    return ["c/" + "/".join(map(str, idx)) for idx in itertools.product(*grid)]


def download_exact(api: KaggleApi, remote: str, local: Path, retries: int = 6) -> None:
    if local.exists() and local.stat().st_size > 0:
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            api.competition_download_file(
                COMPETITION,
                remote,
                path=str(local.parent),
                force=True,
                quiet=True,
            )
            got = local.parent / Path(remote).name
            if got != local and got.exists():
                got.replace(local)
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(f"download produced no file for {remote}")
            return
        except Exception as exc:
            if attempt + 1 >= retries:
                raise
            delay = min(30, 2 ** attempt)
            print(f"retry {attempt + 1}/{retries} {remote}: {type(exc).__name__}: {exc}; sleep={delay}s", flush=True)
            time.sleep(delay)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download_array(api: KaggleApi, remote_base: str, local_base: Path) -> None:
    meta_remote = f"{remote_base}/zarr.json"
    meta_local = local_base / "zarr.json"
    download_exact(api, meta_remote, meta_local)
    meta = read_json(meta_local)
    if meta.get("node_type") != "array":
        return
    for key in chunk_keys(meta):
        download_exact(api, f"{remote_base}/{key}", local_base / key)


def download_movie(stem: str, root: Path) -> None:
    api = KaggleApi()
    api.authenticate()

    movie_root = root / stem
    image_local = movie_root / f"{stem}.zarr"
    gt_local = movie_root / f"{stem}.geff"
    image_remote = f"train/{stem}.zarr"
    gt_remote = f"train/{stem}.geff"

    download_exact(api, f"{image_remote}/zarr.json", image_local / "zarr.json")
    download_array(api, f"{image_remote}/0", image_local / "0")

    group_paths = [
        "zarr.json",
        "nodes/zarr.json",
        "nodes/props/zarr.json",
        "edges/zarr.json",
        "edges/props/zarr.json",
    ]
    for rel in group_paths:
        download_exact(api, f"{gt_remote}/{rel}", gt_local / rel)

    download_array(api, f"{gt_remote}/nodes/ids", gt_local / "nodes" / "ids")
    download_array(api, f"{gt_remote}/edges/ids", gt_local / "edges" / "ids")
    for attr in ("t", "z", "y", "x"):
        download_exact(
            api,
            f"{gt_remote}/nodes/props/{attr}/zarr.json",
            gt_local / "nodes" / "props" / attr / "zarr.json",
        )
        download_array(
            api,
            f"{gt_remote}/nodes/props/{attr}/values",
            gt_local / "nodes" / "props" / attr / "values",
        )

    image_bytes = sum(p.stat().st_size for p in image_local.rglob("*") if p.is_file())
    gt_bytes = sum(p.stat().st_size for p in gt_local.rglob("*") if p.is_file())
    report = {
        "stem": stem,
        "image_bytes": image_bytes,
        "gt_bytes": gt_bytes,
        "image_files": sum(1 for p in image_local.rglob("*") if p.is_file()),
        "gt_files": sum(1 for p in gt_local.rglob("*") if p.is_file()),
        "image_shape": read_json(image_local / "0" / "zarr.json")["shape"],
        "gt_nodes": read_json(gt_local / "nodes" / "ids" / "zarr.json")["shape"][0],
        "gt_edges": read_json(gt_local / "edges" / "ids" / "zarr.json")["shape"][0],
    }
    (movie_root / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--root", type=Path, default=Path("experiments/colab_transfer/train_movies"))
    args = ap.parse_args()
    download_movie(args.stem, args.root)


if __name__ == "__main__":
    main()
