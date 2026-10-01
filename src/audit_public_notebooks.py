from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "kaggle_notebooks"
REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "experiments"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def notebook_text(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    code: list[str] = []
    markdown: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if cell.get("cell_type") == "code":
            code.append(source)
        elif cell.get("cell_type") == "markdown":
            markdown.append(source)
    return "\n\n".join(code), "\n\n".join(markdown)


ENV_PATTERNS = [
    re.compile(r'os\.environ\[\s*["\'](?P<name>BIOHUB_[A-Z0-9_]+)["\']\s*\]\s*=\s*["\'](?P<value>[^"\']*)["\']'),
    re.compile(r'os\.environ\.setdefault\(\s*["\'](?P<name>BIOHUB_[A-Z0-9_]+)["\']\s*,\s*["\'](?P<value>[^"\']*)["\']'),
]


def env_assignments(code: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pattern in ENV_PATTERNS:
        for match in pattern.finditer(code):
            result[match.group("name")] = match.group("value")
    return result


def _strip_comments(code: str) -> str:
    # A lightweight line-level pass is intentional: we want to preserve strings
    # and executable literals, but not classify a comment that *describes* a past
    # exploit as an exploit itself.
    cleaned: list[str] = []
    for line in code.splitlines():
        in_single = in_double = False
        out: list[str] = []
        for i, ch in enumerate(line):
            if ch == "'" and not in_double and (i == 0 or line[i - 1] != "\\"):
                in_single = not in_single
            elif ch == '"' and not in_single and (i == 0 or line[i - 1] != "\\"):
                in_double = not in_double
            if ch == "#" and not in_single and not in_double:
                break
            out.append(ch)
        cleaned.append("".join(out))
    return "\n".join(cleaned)


def classify_audit(title: str, code: str, markdown: str) -> tuple[str, list[str]]:
    executable = _strip_comments(code).lower()
    all_text = f"{title}\n{markdown}\n{code}".lower()
    reasons: list[str] = []
    status = "PASS_HEURISTIC"

    explicit_hack_title = "metric hack" in title.lower() or "metric_hack" in title.lower()
    # The competition format legitimately uses -1 in unused edge/node columns.
    # Only large negative coordinate/time literals in executable code are treated
    # as sentinel evidence. A condition such as `if t < 0` is a *guard*, not a hack.
    sentinel_coord = bool(re.search(r'-(?:10000|10001|9999|1000|999)\b', executable))
    hub_sentinel = "hub" in executable and sentinel_coord

    if explicit_hack_title:
        reasons.append("title_explicit_metric_hack")
    if sentinel_coord:
        reasons.append("sentinel_or_large_negative_coordinate_pattern")
    if hub_sentinel:
        reasons.append("hub_plus_sentinel_pattern")
    if explicit_hack_title or sentinel_coord or hub_sentinel:
        status = "RED_FLAG"

    # Legitimate pipelines may add gap-recovery nodes. That is not automatically
    # an exploit, but it needs image/time/bounds verification before promotion.
    if "gap_synthetic" in all_text or "synthetic midpoint" in all_text or "gap_inserted_synthetic" in all_text:
        reasons.append("synthetic_gap_nodes_require_bounds_and_image_evidence_review")
        if status == "PASS_HEURISTIC":
            status = "REVIEW"

    if "max(0, int(round" in code and "min(" not in code:
        reasons.append("serializer_has_lower_clamp_pattern_check_upper_bounds")
        if status == "PASS_HEURISTIC":
            status = "REVIEW"

    if "source.t < target.t" in all_text or "target_t" in all_text or "consecutive" in all_text:
        reasons.append("temporal_edge_guard_present")
    if "no metric hack" in all_text or '"metric_hack_used": false' in all_text:
        reasons.append("self_declares_no_metric_hack_not_treated_as_proof")
    return status, reasons


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for directory in sorted(p for p in NOTEBOOK_ROOT.iterdir() if p.is_dir()):
        metadata_path = directory / "kernel-metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        code_file = directory / metadata["code_file"]
        code, markdown = notebook_text(code_file)
        env = env_assignments(code)
        audit_status, audit_reasons = classify_audit(metadata.get("title", ""), code, markdown)
        rows.append(
            {
                "notebook_id": directory.name,
                "title": metadata.get("title", ""),
                "author": metadata.get("id", "").split("/", 1)[0],
                "kaggle_ref": metadata.get("id", ""),
                "public_lb": "",
                "parent_ref": "",
                "input_datasets": ";".join(metadata.get("dataset_sources", [])),
                "code_sha256": sha256_file(code_file),
                "pulled_at": str(code_file.stat().st_mtime_ns),
                "exploit_audit": audit_status,
                "status": "PULLED",
                "notes": ";".join(audit_reasons),
                "env_count": str(len(env)),
                "det_threshold": env.get("BIOHUB_DET_THRESHOLD", ""),
                "secondary_edge_weight": env.get("BIOHUB_SECONDARY_EDGE_WEIGHT", ""),
                "bidirectional_edge_weight": env.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", ""),
            }
        )

    csv_path = REPORTS / "notebook_audit.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_fields = [
        "notebook_id", "title", "author", "kaggle_ref", "public_lb", "parent_ref",
        "input_datasets", "code_sha256", "pulled_at", "exploit_audit", "status", "notes",
    ]
    with (EXPERIMENTS / "notebook_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in manifest_fields})

    md = [
        "# Public notebook audit and lineage inventory",
        "",
        "Heuristic audit only. RED_FLAG means the source contains an explicit exploit/sentinel pattern; REVIEW means manual physical-validity audit is still required.",
        "",
        "| Notebook | Author | DET | Secondary edge | Bidir | Audit | Inputs |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        inputs = row["input_datasets"].replace(";", "<br>") or "-"
        md.append(
            f"| {row['title']} | {row['author']} | {row['det_threshold'] or '-'} | "
            f"{row['secondary_edge_weight'] or '-'} | {row['bidirectional_edge_weight'] or '-'} | "
            f"{row['exploit_audit']} | {inputs} |"
        )
    (REPORTS / "notebook_lineage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"audited={len(rows)} csv={csv_path}")


if __name__ == "__main__":
    main()
