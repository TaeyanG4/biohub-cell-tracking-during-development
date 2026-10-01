from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle_notebooks"

TARGETS = {
    "nusrati_0940": NB / "nusrati_0940",
    "analytica_0941": NB / "analytica_0941",
    "busyaprime_0942": NB / "busyaprime_0942",
}

PATTERN = re.compile(
    r'os\.environ\[\s*["\'](?P<name>BIOHUB_[A-Z0-9_]+)["\']\s*\]\s*=\s*["\'](?P<value>[^"\']*)["\']'
)


def load_env(directory: Path) -> tuple[str, dict[str, str]]:
    metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    payload = json.loads((directory / metadata["code_file"]).read_text(encoding="utf-8"))
    code_parts: list[str] = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        code_parts.append(source)
    code = "\n".join(code_parts)
    env: dict[str, str] = {}
    for m in PATTERN.finditer(code):
        env[m.group("name")] = m.group("value")
    return metadata["id"], env


def diff(a: dict[str, str], b: dict[str, str]) -> list[tuple[str, str, str]]:
    keys = sorted(set(a) | set(b))
    return [(key, a.get(key, "<unset>"), b.get(key, "<unset>")) for key in keys if a.get(key) != b.get(key)]


def main() -> None:
    configs = {name: load_env(path) for name, path in TARGETS.items()}
    report = [
        "# One-knob configuration diff",
        "",
        "The effective environment assignments are extracted from the pulled notebook source. This complements the notebook author's provenance table; it does not assume titles are scores.",
        "",
    ]
    pairs = [("nusrati_0940", "analytica_0941"), ("analytica_0941", "busyaprime_0942")]
    for left, right in pairs:
        left_ref, left_env = configs[left]
        right_ref, right_env = configs[right]
        changes = diff(left_env, right_env)
        report += [
            f"## {left} -> {right}",
            "",
            f"- left: `{left_ref}`",
            f"- right: `{right_ref}`",
            f"- changed effective BIOHUB variables: **{len(changes)}**",
            "",
            "| Variable | Left | Right |",
            "|---|---:|---:|",
        ]
        for key, old, new in changes:
            report.append(f"| `{key}` | `{old}` | `{new}` |")
        report.append("")

    # The 0.942 notebook documents the actual controlled leaderboard sweep.
    report += [
        "## Controlled one-knob result documented in busyaprime_0942",
        "",
        "`BIOHUB_DET_THRESHOLD`: published 0.941 value `0.965` -> `0.96`.",
        "",
        "Documented leaderboard sweep: `0.94 -> 0.938`, `0.95 -> 0.940`, `0.96 -> 0.942`, `0.965 -> 0.941`.",
        "",
        "Independent second route: `BIOHUB_SECONDARY_EDGE_WEIGHT` `0.15 -> 0.25` also reached 0.942 with detection threshold held at the published value.",
        "",
        "Robustness warning: the notebook states its held-out proxy moved in the opposite direction for the 0.965 -> 0.96 change, so the leaderboard optimum is not yet a validated CV optimum.",
    ]
    out = ROOT / "reports" / "one_knob_diff.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
