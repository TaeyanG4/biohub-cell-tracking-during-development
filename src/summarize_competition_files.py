from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


COMP = "biohub-cell-tracking-during-development"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "research_20260913"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    all_files: list[tuple[str, int]] = []
    token = None
    pages = 0
    while True:
        resp = api.competition_list_files(COMP, page_token=token, page_size=200)
        pages += 1
        for item in getattr(resp, "files", []) or []:
            name = str(getattr(item, "name", ""))
            size = int(getattr(item, "totalBytes", 0) or getattr(item, "total_bytes", 0) or 0)
            all_files.append((name, size))
        token = getattr(resp, "nextPageToken", None) or getattr(resp, "next_page_token", None)
        if not token:
            break

    by_top: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_movie: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})

    for name, size in all_files:
        parts = name.replace("\\", "/").split("/")
        top = parts[0] if parts else ""
        by_top[top]["files"] += 1
        by_top[top]["bytes"] += size

        if top in {"train", "test"} and len(parts) >= 2:
            component = parts[1]
            stem = component
            if stem.endswith(".zarr"):
                stem = stem[:-5]
            elif stem.endswith(".geff"):
                stem = stem[:-5]
            by_movie[(top, stem)]["files"] += 1
            by_movie[(top, stem)]["bytes"] += size

    movie_rows = []
    for (split, stem), stats in sorted(by_movie.items()):
        prefix = stem.split("_", 1)[0]
        movie_rows.append(
            {
                "split": split,
                "stem": stem,
                "prefix": prefix,
                "files": stats["files"],
                "bytes": stats["bytes"],
                "gib": stats["bytes"] / (1024**3),
            }
        )

    prefix_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"movies": 0, "files": 0, "bytes": 0}
    )
    for row in movie_rows:
        key = (row["split"], row["prefix"])
        prefix_stats[key]["movies"] += 1
        prefix_stats[key]["files"] += int(row["files"])
        prefix_stats[key]["bytes"] += int(row["bytes"])

    summary = {
        "competition": COMP,
        "pages": pages,
        "total_files": len(all_files),
        "total_bytes": sum(size for _, size in all_files),
        "total_gib": sum(size for _, size in all_files) / (1024**3),
        "top_level": {
            key: {
                **value,
                "gib": value["bytes"] / (1024**3),
            }
            for key, value in sorted(by_top.items())
        },
        "movie_counts": {
            split: sum(1 for row in movie_rows if row["split"] == split)
            for split in ("train", "test")
        },
        "prefixes": [
            {
                "split": split,
                "prefix": prefix,
                "movies": int(stats["movies"]),
                "files": int(stats["files"]),
                "bytes": int(stats["bytes"]),
                "gib": float(stats["bytes"]) / (1024**3),
            }
            for (split, prefix), stats in sorted(prefix_stats.items())
        ],
    }

    (OUT / "competition_files_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "competition_movies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "stem", "prefix", "files", "bytes", "gib"]
        )
        writer.writeheader()
        writer.writerows(movie_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
