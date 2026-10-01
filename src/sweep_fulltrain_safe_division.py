from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_fulltrain_safe_division_gt import simulate


CONFIGS = {
    "base": {},
    "diverge_1p5": {"diverge_um": 1.5},
    "diverge_1p0": {"diverge_um": 1.0},
    "diverge_0p5": {"diverge_um": 0.5},
    "diverge_0": {"diverge_um": 0.0},
    "no_divergence": {"require_divergence": False},
    "sym_0p8": {"symmetry_tau": 0.8},
    "sym_1p0": {"symmetry_tau": 1.0},
    "sym_off": {"symmetry_tau": 0.0},
    "parent10": {"parent_max": 10.0},
    "parent11": {"parent_max": 11.0},
    "existing12": {"existing_max": 12.0},
    "mutual_off": {"require_mutual_nn": False},
    "combo_conservative": {
        "parent_max": 10.0,
        "symmetry_tau": 0.8,
        "diverge_um": 1.5,
    },
    "combo_medium": {
        "parent_max": 10.0,
        "symmetry_tau": 1.0,
        "diverge_um": 1.0,
    },
    "combo_wide": {
        "parent_max": 11.0,
        "existing_max": 12.0,
        "symmetry_tau": 1.0,
        "diverge_um": 0.5,
    },
}


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for mode in ("nearest", "farther"):
        for group in ("44b6", "6bba"):
            rr = [r for r in rows if r["existing_mode"] == mode and r["group"] == group]
            tp = sum(r["tp"] for r in rr)
            fp = sum(r["fp"] for r in rr)
            fn = sum(r["fn"] for r in rr)
            key = f"{mode}:{group}"
            out[key] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 1.0,
                "recall": tp / (tp + fn) if tp + fn else 1.0,
                "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            }
    vals = list(out.values())
    out["robust"] = {
        "min_jaccard": min(v["jaccard"] for v in vals),
        "mean_jaccard": sum(v["jaccard"] for v in vals) / len(vals),
        "min_precision": min(v["precision"] for v in vals),
        "min_recall": min(v["recall"] for v in vals),
        "tp": sum(v["tp"] for v in vals),
        "fp": sum(v["fp"] for v in vals),
        "fn": sum(v["fn"] for v in vals),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.gt_root.glob("*.geff"))
    result = {}
    for ci, (name, cfg) in enumerate(CONFIGS.items(), 1):
        rows = []
        print(f"CONFIG {ci}/{len(CONFIGS)} {name} {cfg}", flush=True)
        for i, path in enumerate(paths, 1):
            for mode in ("nearest", "farther"):
                rows.append(simulate(path, existing_mode=mode, **cfg))
            if i % 50 == 0 or i == len(paths):
                print(" ", name, "processed", i, "of", len(paths), flush=True)
        agg = aggregate(rows)
        result[name] = {"config": cfg, "aggregate": agg}
        print("RESULT", name, json.dumps(agg["robust"], sort_keys=True), flush=True)

    ranked = sorted(
        result.items(),
        key=lambda kv: (
            kv[1]["aggregate"]["robust"]["min_jaccard"],
            kv[1]["aggregate"]["robust"]["mean_jaccard"],
            kv[1]["aggregate"]["robust"]["min_precision"],
        ),
        reverse=True,
    )
    report = {
        "ranking": [
            {
                "name": name,
                "config": payload["config"],
                **payload["aggregate"]["robust"],
            }
            for name, payload in ranked
        ],
        "results": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TOP")
    print(json.dumps(report["ranking"][:8], indent=2), flush=True)


if __name__ == "__main__":
    main()
