from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "artifact_manifest.csv"

SOURCES = {
    ROOT / "artifacts" / "pilkwang_deepcenter": ("kaggle_dataset", "pilkwang/biohub-deepcenter-unet3d-center-prior-v1", "CC0-1.0"),
    ROOT / "artifacts" / "pilkwang_temporal_seed314159": ("kaggle_dataset", "pilkwang/biohub-temporal-unet3d-seed314159-v1", "CC0-1.0"),
    ROOT / "artifacts" / "pilkwang_support50": ("kaggle_dataset", "pilkwang/biohub-tracking-support-pack-50ep-v1", "CC0-1.0"),
    ROOT / "artifacts" / "busyaprime_knob_provenance": ("kaggle_dataset", "busyaprime/biohub-knob-provenance", "CC0-1.0"),
    ROOT / "vendor" / "royerlab_metric": ("github_archive", "royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2", "BSD-3-Clause"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows: list[dict[str, str | int]] = []
    now = datetime.now(timezone.utc).isoformat()
    artifact_id = 1
    for root, (kind, source_ref, license_name) in SOURCES.items():
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(ROOT).as_posix()
            rows.append(
                {
                    "artifact_id": f"ART{artifact_id:05d}",
                    "type": kind,
                    "name": path.name,
                    "source": "Kaggle" if kind == "kaggle_dataset" else "GitHub",
                    "source_ref": source_ref,
                    "local_path": rel,
                    "version": "",
                    "license": license_name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "acquired_at": now,
                    "notes": "",
                }
            )
            artifact_id += 1
    fieldnames = [
        "artifact_id", "type", "name", "source", "source_ref", "local_path",
        "version", "license", "sha256", "size_bytes", "acquired_at", "notes",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"artifacts={len(rows)} manifest={OUT}")


if __name__ == "__main__":
    main()
