from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8-sig"))
    code = "\n".join(
        c.get("source", "") if isinstance(c.get("source", ""), str) else "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )
    markdown = "\n".join(
        c.get("source", "") if isinstance(c.get("source", ""), str) else "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "markdown"
    )

    env = []
    for m in re.finditer(r"os\.environ\[['\"]([^'\"]+)['\"]\]\s*=\s*['\"]([^'\"]*)['\"]", code):
        env.append((m.group(1), m.group(2)))

    print("NOTEBOOK", args.notebook)
    print("ENV")
    for key, value in env:
        if key.startswith("BIOHUB_"):
            print(f"{key}={value}")

    print("\nKEY MARKDOWN")
    for line in markdown.splitlines():
        if re.search(r"(?i)public lb|leaderboard|score|ranker|association|ensemble|0\.9[0-9][0-9]", line):
            print(line.strip())


if __name__ == "__main__":
    main()
