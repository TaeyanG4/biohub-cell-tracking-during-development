from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "visible_gt"
OUT.mkdir(parents=True, exist_ok=True)
COMP = "biohub-cell-tracking-during-development"
STEMS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


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
        raise RuntimeError(proc.stdout[-4000:])
    return proc.stdout


files: dict[str, dict] = {}
token = None
for page in range(1, 80):
    cmd = [
        "kaggle", "competitions", "files", COMP,
        "--page-size", "200", "--format", "json",
    ]
    if token:
        cmd += ["--page-token", token]
    out = run(cmd)
    m = re.search(r"^Next Page Token = (.+)$", out, re.MULTILINE)
    token = m.group(1).strip() if m else None
    rows = json.loads(out[out.find("["):])
    for row in rows:
        name = row["name"]
        if any(name.startswith(f"train/{stem}.geff/") for stem in STEMS):
            files[name] = row
    counts = {
        stem: sum(name.startswith(f"train/{stem}.geff/") for name in files)
        for stem in STEMS
    }
    if page % 5 == 0 or any(counts.values()):
        print(f"page={page} counts={counts}", flush=True)
    if all(counts[stem] >= 21 for stem in STEMS):
        break
    if token is None:
        break

counts = {
    stem: sum(name.startswith(f"train/{stem}.geff/") for name in files)
    for stem in STEMS
}
print("found", counts, flush=True)
if not all(counts[stem] >= 21 for stem in STEMS):
    raise RuntimeError(f"GT GEFF listing incomplete: {counts}")

for i, name in enumerate(sorted(files), 1):
    rel = Path(name)
    parent = OUT / rel.parent
    parent.mkdir(parents=True, exist_ok=True)
    target = OUT / rel
    if target.is_file():
        continue
    run([
        "kaggle", "competitions", "download", COMP,
        "-f", name, "-p", str(parent), "-q",
    ])
    downloaded = parent / rel.name
    if not downloaded.is_file():
        raise FileNotFoundError(downloaded)
    if downloaded != target:
        downloaded.replace(target)
    if i % 10 == 0:
        print(f"downloaded {i}/{len(files)}", flush=True)

print("VISIBLE_GT_READY", OUT, flush=True)
