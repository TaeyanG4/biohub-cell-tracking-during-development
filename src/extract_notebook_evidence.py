from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "kaggle_notebooks"
REPORT = ROOT / "reports" / "exploit_evidence.md"

TARGETS = [
    "nusrati_0940",
    "analytica_0941",
    "busyaprime_0942",
    "reyhan_0946",
    "harmonic_fusion",
    "lineage_forge",
    "yusuke_clean_cv",
]

TERMS = re.compile(
    r"-10000|-10001|-1000|-999|sentinel|metric[_ ]?hack|augment_dataset|"
    r"synthetic|gap_inserted|gap_synthetic|hub node|weakly connected|"
    r"row_type|submission|source_id|target_id|no cycle|outdegree|out_degree",
    re.IGNORECASE,
)


def load_cells(directory: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((directory / metadata["code_file"]).read_text(encoding="utf-8"))
    return metadata, notebook.get("cells", [])


def main() -> None:
    out = ["# Exploit-audit evidence excerpts", ""]
    for target in TARGETS:
        directory = NOTEBOOKS / target
        if not directory.exists():
            continue
        metadata, cells = load_cells(directory)
        out += [f"## {target}", "", f"Kaggle ref: `{metadata.get('id','')}`", ""]
        hits = 0
        large_negative_code_hits = 0
        for idx, cell in enumerate(cells):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            for line_no, line in enumerate(source.splitlines(), 1):
                if not TERMS.search(line):
                    continue
                hits += 1
                stripped = line.strip()
                if cell.get("cell_type") == "code" and re.search(r"-(10000|10001|1000|999)\b", stripped):
                    large_negative_code_hits += 1
                # Keep report readable while preserving the highest-value evidence.
                if hits <= 90:
                    clipped = stripped.replace("`", "'")[:320]
                    out.append(f"- cell {idx} [{cell.get('cell_type')}] line {line_no}: `{clipped}`")
        out += ["", f"Executable large-negative sentinel literal hits: **{large_negative_code_hits}**", ""]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
