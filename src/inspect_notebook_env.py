from __future__ import annotations

import json
import sys
from pathlib import Path


KEYS = [
    "BIOHUB_DET_THRESHOLD",
    "BIOHUB_MOTION_RELINK_TIGHT_UM",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD",
    "BIOHUB_EDGE_FEATURE_TTA",
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA",
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT",
    "BIOHUB_DEEPCENTER_TTA",
    "BIOHUB_OUTPUT_LINEFIT_SMOOTH",
    "BIOHUB_HOCT_VETO",
]


def main(path: str) -> None:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in KEYS:
        matches = []
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            for line in src.splitlines():
                if key in line and ("os.environ" in line or "OUTPUT_LINEFIT" in line):
                    matches.append(line.strip())
        print(key, matches[:8] if matches else "<unset>")


if __name__ == "__main__":
    main(sys.argv[1])
