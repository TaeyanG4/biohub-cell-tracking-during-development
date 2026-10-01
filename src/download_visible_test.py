from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


COMP = "biohub-cell-tracking-during-development"
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "visible_test"
OUT.mkdir(parents=True, exist_ok=True)


def run(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode:
        raise RuntimeError(f"command failed {proc.returncode}: {' '.join(args)}\n{proc.stdout[-4000:]}")
    return proc.stdout


def parse_page(text: str) -> tuple[str | None, list[dict]]:
    match = re.search(r"^Next Page Token = (.+)$", text, re.MULTILINE)
    token = match.group(1).strip() if match else None
    start = text.find("[")
    if start < 0:
        raise RuntimeError(f"JSON array not found in output:\n{text[:2000]}")
    rows = json.loads(text[start:])
    return token, rows


files: list[dict] = []
token: str | None = None
for page_index in range(1, 8):
    cmd = [
        "kaggle",
        "competitions",
        "files",
        COMP,
        "--page-size",
        "200",
        "--format",
        "json",
    ]
    if token:
        cmd += ["--page-token", token]
    text = run(cmd)
    token, rows = parse_page(text)
    matching = [
        row
        for row in rows
        if any(row["name"].startswith(f"test/{stem}.zarr/") for stem in STEMS)
    ]
    files.extend(matching)
    counts = {
        stem: sum(row["name"].startswith(f"test/{stem}.zarr/") for row in files)
        for stem in STEMS
    }
    print(f"page={page_index} matched={len(matching)} totals={counts}", flush=True)
    if all(counts[stem] >= 102 for stem in STEMS):
        break
    if token is None:
        break

unique = {row["name"]: row for row in files}
counts = {
    stem: sum(name.startswith(f"test/{stem}.zarr/") for name in unique)
    for stem in STEMS
}
print("final_counts", counts, "total", len(unique), flush=True)
if not all(counts[stem] >= 102 for stem in STEMS):
    raise RuntimeError(f"incomplete visible test listing: {counts}")

manifest = OUT / "manifest.json"
manifest.write_text(json.dumps(list(unique.values()), indent=2) + "\n", encoding="utf-8")

for index, name in enumerate(sorted(unique), start=1):
    rel = Path(name)
    parent = OUT / rel.parent
    parent.mkdir(parents=True, exist_ok=True)
    target = OUT / rel
    if target.is_file() and target.stat().st_size > 0:
        if index % 25 == 0 or index == len(unique):
            print(f"reuse {index}/{len(unique)} {name}", flush=True)
        continue
    # Kaggle's single-file downloader writes the basename to --path, so give it
    # the exact Zarr parent directory to preserve the chunk layout.
    run(
        [
            "kaggle",
            "competitions",
            "download",
            COMP,
            "-f",
            name,
            "-p",
            str(parent),
            "-q",
        ]
    )
    downloaded = parent / rel.name
    if not downloaded.is_file():
        raise FileNotFoundError(f"download did not create {downloaded}")
    if downloaded != target:
        downloaded.replace(target)
    if index % 10 == 0 or index == len(unique):
        print(f"downloaded {index}/{len(unique)} {name}", flush=True)

print("VISIBLE_TEST_READY", OUT, flush=True)
