from __future__ import annotations

import json
import re
import sys
from pathlib import Path


KEYWORDS = (
    "score",
    "lb",
    "tta",
    "deepcenter",
    "secondary",
    "hoct",
    "focus",
    "division",
    "dense",
    "veto",
    "self-distill",
    "self distill",
    "edge_feature",
    "candidate",
)


def inspect(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n### {path}")
    env = {}
    seen = set()
    for ci, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for m in re.finditer(
            r"os\.environ\[['\"](?P<key>BIOHUB_[A-Z0-9_]+)['\"]\]\s*=\s*['\"](?P<val>[^'\"]*)['\"]",
            src,
        ):
            env[m.group("key")] = m.group("val")
        for line in src.splitlines():
            low = line.lower()
            if any(k in low for k in KEYWORDS):
                clean = line.strip()
                if clean and clean not in seen and len(clean) < 260:
                    seen.add(clean)
                    print(f"cell{ci}: {clean}")
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in text.splitlines():
                low = line.lower()
                if any(k in low for k in KEYWORDS):
                    clean = line.strip()
                    if clean and clean not in seen and len(clean) < 260:
                        seen.add(clean)
                        print(f"out{ci}: {clean}")
    print("ENV:")
    for k, v in sorted(env.items()):
        print(f"  {k}={v}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inspect(Path(arg))
