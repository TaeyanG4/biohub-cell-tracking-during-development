from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("needle")
    parser.add_argument("--context", type=int, default=8)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nb = json.loads(args.notebook.read_text(encoding="utf-8-sig"))
    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        lines = source.splitlines()
        for line_index, line in enumerate(lines):
            if args.needle in line:
                lo = max(0, line_index - args.context)
                hi = min(len(lines), line_index + args.context + 1)
                print(f"CELL {cell_index} LINES {lo + 1}-{hi}")
                for i in range(lo, hi):
                    print(f"{i + 1:05d}: {lines[i]}")
                print()


if __name__ == "__main__":
    main()
