#!/usr/bin/env python3
"""C055: Lineage Forge V12 guarded endpoint readmission replacing C023's readmission.

Finite, resumable, CPU/DeepCenter-only replay study on the existing C023 FP32
caches (experiments/candidates/c032_temporal_context/e2e/control_fp32_*).  No
inference, training, threshold sweep, Kaggle write or notebook edit.

    python src/c055_guarded_readmit.py prepare
    python src/c055_guarded_readmit.py run           # the finite queue (hidden process)
    python src/c055_guarded_readmit.py analyse       # re-run the analysis only

Jobs (status.json): check_cache, control_replay_{heldout12,confirm10} (pinned
harness CLI, as configured), study_{heldout12,confirm10} (control / readmit_off /
v12_guarded in process, exact control parity, graphs, survival), analyse_22;
then, only if the pre-registered gate passes, control_replay_extension75,
study_extension75 and analyse_97.

Pre-registered extension gate (v12_guarded versus control on all 22 movies):
official total delta > 0 AND adjusted-edge delta > 0 AND total delta >= 0 on
both embryo subsets AND at least one readmitted node survives to a final graph.
Small local gains remain fit-domain post-processing evidence (HANDOFF section 4
protocol), never a submission decision by themselves.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_last_days_local import Queue, read_rows, read_stems, extension_stems, stamp  # noqa: E402
import eval_pp_variants_local as harness  # noqa: E402
from c055_guarded_readmit_stage import install_guarded_readmit, assign_node_peaks, NODE_PEAK_IDENTITY_UM  # noqa: E402

DEST = ROOT / "experiments/candidates/c055_guarded_readmit"
BASE = ROOT / "experiments/candidates/c023_x138_head_stabilize_restore/biohub-c023-x138-head-stabilize-restore.ipynb"
CONTROL = ROOT / "experiments/candidates/c032_temporal_context/e2e"
OLD = ROOT / "state/last_days_local_20260926/replay"
C046_GRAPHS = ROOT / "experiments/candidates/c046_fixed_appearance_extension/graphs"
GT_DIR = ROOT / "data/train"
SPLITS = ROOT / "experiments/candidates/c012_v1284_head"
PREDICT_SCRIPT = CONTROL / "control_fp32_heldout12/_work/tracking_repo/scripts/predict_unet_transformer.py"
VARIANTS = {"control": {}, "readmit_off": {"READMIT_RADIUS_UM": 0.0},
            "v12_guarded": {"READMIT_RADIUS_UM": 0.0, "GUARDED_READMIT": 1}}
SPLIT_ORDER = ["heldout12", "confirm10", "extension75"]
COMPARE_INT = ["edge_tp", "edge_fp", "edge_fn", "t_pred", "div_tp", "div_fp", "div_fn", "weight", "missed_gt_nodes",
               "spurious_pred_nodes", "edges_recovered", "edges_fragmented", "edges_lost_to_detection",
               "wrong_association_edges", "nodes", "edges", "safe_divisions_added", "readmitted_nodes",
               "gapfill_added_nodes", "leaf_pruned_nodes", "flow_frames"]
COMPARE_FLOAT = ["edge_jaccard", "t_true", "adjusted_edge_jaccard", "div_jaccard"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_stems():
    held = read_stems(SPLITS / "heldout_stems.txt")
    conf = read_stems(SPLITS / "confirm_stems.txt")
    ext, _ = extension_stems(held + conf)
    return {"heldout12": held, "confirm10": conf, "extension75": ext}


def run_dir(split):
    return CONTROL / f"control_fp32_{split}"


def pred_path(split, stem):
    p = next((run_dir(split) / "predictions").rglob(f"{stem}.geff"), None)
    if p is None:
        raise FileNotFoundError(f"{split}/{stem}: no ILP graph")
    return p


# ----------------------------------------------------------------------------- prepare
def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "plan.json").exists(), "Registered source is immutable; use a new --out"
    stems = split_stems()
    for split, lst in stems.items():
        (out / f"{split}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    save_json(out / "variants.json", VARIANTS)
    save_json(out / "control_variants.json", {"as_configured": {}})
    pinned = {}
    files = [Path(__file__), ROOT / "src/c055_guarded_readmit_stage.py", ROOT / "src/eval_pp_variants_local.py",
             ROOT / "src/run_last_days_local.py", BASE, PREDICT_SCRIPT,
             ROOT / "state/public_lineage_review_20260929/latest_cell_05.txt",
             ROOT / "state/public_lineage_review_20260929/source_manifest.json",
             harness.DEEPCENTER_CHECKPOINT, harness.DEEPCENTER_MANIFEST]
    files += [OLD / f"control_fp32_{s}.csv" for s in SPLIT_ORDER]
    for split, lst in stems.items():
        for stem in lst:
            files.append(run_dir(split) / "edge_cache" / f"{stem}.npz")
            files += [p for p in pred_path(split, stem).rglob("*") if p.is_file()]
            files += [p for p in (GT_DIR / f"{stem}.geff").rglob("*") if p.is_file()]
            files.append(C046_GRAPHS / split / f"{stem}_off.npz")
    files += [p for p in out.iterdir() if p.is_file()]
    for p in files:
        assert p.exists(), p
        pinned[str(p.relative_to(ROOT))] = sha(p)
    save_json(out / "plan.json", dict(
        created=stamp(), splits={k: v for k, v in stems.items()}, variants=VARIANTS, hashes=pinned,
        total_jobs=9, jobs_if_gate_fails=6, max_start_job_hours=4, estimated_minutes=95,
        estimate_basis="C054 replays: 178s/12 + 124s/10 + 972s/75 per variant; 3 in-process variants + 1 CLI control per split + cache check",
        gate="v12_guarded vs control on all22: total>0 and edge>0 and both-embryo total>=0 and readmit_nodes_final>0",
        policy="Fixed V12 constants; C023 readmission replaced (READMIT_RADIUS_UM=0 in the new arms); no fitting, sweep, "
               "inference, Kaggle write or notebook edit. Unannotated detections are never treated as negatives."))
    (out / "README.md").write_text(README, encoding="utf-8")
    print("Prepared C055:", len(pinned), "pinned inputs", flush=True)


README = """# C055 guarded endpoint readmission (Lineage Forge V12 port) on C023

Study of ONE fixed component: V12's motion-predicted endpoint readmission
(score >= 0.94, step <= 3 um, residual <= 1.5 um, separation >= 1.5 um,
ambiguity margin 0.5, bridge preference, budget min(100, 0.2% of nodes))
replacing C023's radius readmission (0.965 / 4 um).  Detector, V1284 head,
ILP (division weight 1.2), stabilized relink, restore and every other stage are
C023's own, replayed by the pinned `src/eval_pp_variants_local.py` on the
existing C023 FP32 caches.  Stage: `src/c055_guarded_readmit_stage.py`;
driver: `src/c055_guarded_readmit.py`.

Arms: `control` (C023 as configured), `readmit_off` (C023 readmission off,
diagnostic only), `v12_guarded` (readmission off + V12 stage).  Each split first
reproduces the stored 2026-09-26 C023 control rows with the harness CLI, then the
in-process loop must equal the CLI rows and the C046 `off` final graphs exactly.

Coordinate semantics: raw peaks belonging to existing (head-refined) nodes are
excluded by identity before V12's separation test (documented in the stage).
Outputs: `replay/`, `study/`, `graphs/`, `cache_semantics.json`, `parity_*.json`,
`analysis.json`, `official_summary.csv`, `decision.json`, `REVIEW.md`.
No Kaggle operation is performed by this queue.
"""


# ----------------------------------------------------------------------------- cache semantics
def check_cache(out):
    """Verification step 1: C023 low-detection dump == V12 candidate semantics."""
    script = PREDICT_SCRIPT.read_text(encoding="utf-8")
    assert "is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)" in script
    assert "_lc[:, 1:] *= np.array(downsample, dtype=np.float32)" in script
    assert "torch.sigmoid(_lg[_lz[:, 0], _lz[:, 1], _lz[:, 2]])" in script
    assert '"downsample": [1, 4, 4]' in script
    scale = np.array([1.625, 0.40625, 0.40625])
    rows = []
    for split, stems in split_stems().items():
        for stem in stems:
            with np.load(run_dir(split) / "edge_cache" / f"{stem}.npz") as z:
                coords = z["coords"].astype(np.float64)
                low = z["low_coords"]
                score = z["low_score"].astype(np.float64)
                assert low.dtype == np.int16 and z["low_score"].dtype == np.float32
            low = low.astype(np.float64)
            keys = {tuple(int(v) for v in r) for r in low.astype(np.int64)}
            hi = low[score > 0.965]
            worst, unassigned, conflicts = 0.0, 0, 0
            for t in np.unique(coords[:, 0]):
                a = coords[coords[:, 0] == t][:, 1:] * scale
                b = hi[hi[:, 0] == t][:, 1:] * scale
                if len(b) == 0:
                    unassigned += len(a)
                    continue
                d, _ = cKDTree_query(b, a)
                worst = max(worst, float(d.max()))
                _, missing, conf = assign_node_peaks(a, b, NODE_PEAK_IDENTITY_UM)
                unassigned += missing
                conflicts += conf
            rows.append(dict(split=split, stem=stem, production_nodes=len(coords), peaks_gt_0965=len(hi),
                             peaks_ge_094=int((score >= 0.94).sum()), extra_094_0965=int(((score >= 0.94) & (score <= 0.965)).sum()),
                             dump_rows=len(low), unique_keys=len(keys), score_min=float(score.min()), score_max=float(score.max()),
                             yx_multiple_of_4=bool(((low[:, 2] % 4 == 0) & (low[:, 3] % 4 == 0)).all()),
                             refined_to_peak_max_um=worst, nodes_without_own_peak=unassigned, nearest_peak_conflicts_resolved=conflicts))
    df = pd.DataFrame(rows)
    df.to_csv(out / "cache_semantics.csv", index=False)
    ok = bool((df.production_nodes == df.peaks_gt_0965).all() and (df.dump_rows == df.unique_keys).all()
              and df.yx_multiple_of_4.all() and (df.nodes_without_own_peak == 0).all()
              and (df.score_min >= 0.3).all() and (df.score_max <= 1.0).all())
    save_json(out / "cache_semantics.json", dict(
        status="passed" if ok else "FAILED", movies=len(df), predict_script_sha256=sha(PREDICT_SCRIPT),
        local_max_mask_threshold_independent=True, score_is_sigmoid_at_peak=True, coords_scaled_by_downsample=[1, 4, 4],
        production_nodes_equal_peaks_gt_0965=bool((df.production_nodes == df.peaks_gt_0965).all()),
        dedup_noop=bool((df.dump_rows == df.unique_keys).all()),
        refined_to_peak_max_um=float(df.refined_to_peak_max_um.max()),
        nodes_without_own_peak_total=int(df.nodes_without_own_peak.sum()),
        nearest_peak_conflicts_resolved_total=int(df.nearest_peak_conflicts_resolved.sum()),
        extra_candidates_094_0965_total=int(df.extra_094_0965.sum()),
        note="Identity exclusion of existing nodes' raw peaks (<=2.5 um, unambiguous) replaces V12's implicit zero-distance rejection."))
    print(json.dumps(read(out / "cache_semantics.json"), indent=1))
    assert ok, "cache semantics check failed"


def cKDTree_query(points, queries):
    from scipy.spatial import cKDTree
    return cKDTree(points).query(queries, k=1)


# ----------------------------------------------------------------------------- control parity + study
def control_command(out, split, stems):
    return [sys.executable, "-u", ROOT / "src/eval_pp_variants_local.py", "--notebook", BASE,
            "--variants", out / "control_variants.json", "--stems", ",".join(stems),
            "--pred-root", run_dir(split) / "predictions", "--lowdet-dir", run_dir(split) / "edge_cache",
            "--round-coords", "--out", out / "replay" / f"control_{split}.csv"]


def compare_rows(a, b, label):
    """Exact row equality on the official / count columns (floats within 1e-12)."""
    a = {r["stem"]: r for r in a}
    b = {r["stem"]: r for r in b}
    assert set(a) == set(b), (label, "stem sets differ")
    for stem in a:
        for key in COMPARE_INT:
            if key in a[stem] and key in b[stem]:
                assert int(float(a[stem][key])) == int(float(b[stem][key])), (label, stem, key, a[stem][key], b[stem][key])
        for key in COMPARE_FLOAT:
            assert abs(float(a[stem][key]) - float(b[stem][key])) <= 1e-12, (label, stem, key)
    return len(a)


def study(out, split):
    stems = read_stems(out / f"{split}.txt")
    control_csv = out / "replay" / f"control_{split}.csv"
    stored = [r for r in read_rows(OLD / f"control_fp32_{split}.csv") if r["config"] == "as_configured"]
    cli = [r for r in read_rows(control_csv) if r["config"] == "as_configured"]
    n_stored = compare_rows(cli, stored, "cli_vs_stored_2026-09-26")
    lowdet = run_dir(split) / "edge_cache"
    graphs = out / "graphs" / split
    log_path = out / "study" / f"{split}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = harness.build_namespace(BASE.resolve(), {}, lowdet)
    ns["TEST_DIR"] = GT_DIR.resolve()
    frames, heatmaps = harness.install_frame_caches(ns)
    install_guarded_readmit(ns, lowdet, graphs)
    rows = []
    t_start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for stem in stems:
            raw_nodes, raw_edges = harness.load_raw_graph(ns, pred_path(split, stem))
            gt_path = GT_DIR / f"{stem}.geff"
            gt_nodes, gt_edges = ns["graph_to_plain"](ns["graph_from_geff"](gt_path))
            t_true = ns["read_estimated_true_node_count"](gt_path)
            for label, overrides in VARIANTS.items():
                saved = harness.apply_overrides(ns, overrides)
                ns["_C055_LABEL"] = label
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(log):
                        print(f"===== {stem} | {label} | {overrides}")
                        nodes, edges, stats = ns["filter_output_graph"](
                            copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                            deepcenter_bundle=ns.get("DEEPCENTER_VETO_DETECTOR"))
                finally:
                    for key, value in saved.items():
                        ns[key] = value
                pred_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
                pred_nodes = ns["nodes_by_id_to_plain"](nodes)
                pred_nodes = {nid: (t, *(max(0, int(round(v))) for v in zyx)) for nid, (t, *zyx) in pred_nodes.items()}
                row = ns["score_sample"](pred_nodes, pred_edges, gt_nodes, gt_edges, t_true)
                row.update(stem=stem, config=label, nodes=len(nodes), edges=len(edges),
                           safe_divisions_added=int(stats.get("safe_divisions_added", 0)),
                           readmitted_nodes=int(stats.get("readmitted_nodes", 0)),
                           gapfill_added_nodes=int(stats.get("gapfill_added_nodes", 0)),
                           leaf_pruned_nodes=int(stats.get("leaf_prune_nodes", 0)),
                           flow_frames=int(stats.get("motion_relink_flow_frames", 0)),
                           seconds=round(time.time() - t0, 1))
                record = ns["_C055_RECORDS"].get((stem, label), {})
                row.update({k: v for k, v in record.items() if k not in ("stem", "label")})
                rows.append(row)
                print(f"{stem:15s} {label:12s} adj={row['adjusted_edge_jaccard']:.6f} edge tp/fp/fn={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                      f"div={row['div_tp']}/{row['div_fp']}/{row['div_fn']} nodes={row['nodes']} edges={row['edges']} "
                      f"readmit={row.get('readmit_nodes', 0)}/{row.get('readmit_nodes_final', 0)} [{row['seconds']:.0f}s]", flush=True)
            frames.clear()
            heatmaps.clear()
    df = pd.DataFrame(rows)
    (out / "study").mkdir(exist_ok=True)
    df.to_csv(out / "study" / f"{split}.csv", index=False)
    # In-process control must equal the CLI control rows exactly.
    n_cli = compare_rows(df[df.config == "control"].to_dict("records"), cli, "inprocess_vs_cli")
    # Control final graphs versus C046's stored C023 'off' graphs (independent prior artifact; recorded, reviewed).
    graph_exact, graph_diff = 0, []
    for stem in stems:
        mine = np.load(graphs / f"{stem}_control.npz")
        ref = np.load(C046_GRAPHS / split / f"{stem}_off.npz")
        if np.array_equal(mine["ids"], ref["ids"]) and np.array_equal(mine["txyz"], ref["txyz"]) and np.array_equal(mine["edges"], ref["edges"]):
            graph_exact += 1
        else:
            graph_diff.append(stem)
    save_json(out / f"parity_{split}.json", dict(status="passed", split=split, movies=len(stems), cli_vs_stored_rows=n_stored,
                                                 inprocess_vs_cli_rows=n_cli, control_final_graphs_equal_c046_off=graph_exact,
                                                 control_final_graphs_differ_from_c046_off=graph_diff,
                                                 minutes=round((time.time() - t_start) / 60, 1)))
    print(json.dumps(read(out / f"parity_{split}.json")))


# ----------------------------------------------------------------------------- analysis
def analyse(out):
    plan = read(out / "plan.json")
    frames_ = [pd.read_csv(out / "study" / f"{s}.csv") for s in SPLIT_ORDER if (out / "study" / f"{s}.csv").exists()]
    data = pd.concat(frames_, ignore_index=True)
    with (out / "analysis.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = harness.build_namespace(BASE.resolve(), {}, None)
    agg = ns["aggregate_official"]
    stems_all = sorted(set(data.stem))
    splits = plan["splits"]
    groups = {"all22": set(splits["heldout12"]) | set(splits["confirm10"]), "heldout12": set(splits["heldout12"]),
              "confirm10": set(splits["confirm10"])}
    if len(stems_all) > 22:
        groups.update(all97=set(stems_all), extension75=set(splits["extension75"]))
    for g in ["44b6", "6bba"]:
        groups[f"{g}_22"] = {s for s in groups["all22"] if s.startswith(g)}
        if "all97" in groups:
            groups[f"{g}_97"] = {s for s in groups["all97"] if s.startswith(g)}
    counts = ["edge_tp", "edge_fp", "edge_fn", "div_tp", "div_fp", "div_fn"]
    survival = ["readmit_candidates", "readmit_proposals", "readmit_nodes", "readmit_bridges", "readmit_edges",
                "readmit_nodes_final", "readmit_edges_final", "readmit_incident_final"]
    rows = []
    for group, stems in groups.items():
        stems = {s for s in stems if s in set(stems_all)}
        if not stems:
            continue
        sub = {label: data[(data.config == label) & data.stem.isin(stems)] for label in VARIANTS}
        base = agg(sub["control"].to_dict("records"))
        for label in VARIANTS:
            cur = agg(sub[label].to_dict("records"))
            row = dict(group=group, arm=label, movies=len(stems), score=cur["proxy_score"], adjusted_edge=cur["adjusted_edge_jaccard"],
                       delta_total=cur["proxy_score"] - base["proxy_score"], delta_edge=cur["adjusted_edge_jaccard"] - base["adjusted_edge_jaccard"])
            a = sub[label].set_index("stem").loc[sorted(stems)]
            b = sub["control"].set_index("stem").loc[sorted(stems)]
            for k in counts:
                row[k] = int(a[k].sum())
                row["d_" + k] = int(a[k].sum() - b[k].sum())
            per = a["adjusted_edge_jaccard"] - b["adjusted_edge_jaccard"]
            row["movie_wins"] = int((per > 1e-12).sum())
            row["movie_losses"] = int((per < -1e-12).sum())
            for k in survival:
                row[k] = int(a[k].fillna(0).sum()) if k in a else 0
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "official_summary.csv", index=False)
    v = {r["group"]: r for r in rows if r["arm"] == "v12_guarded"}
    gate = bool(v["all22"]["delta_total"] > 0 and v["all22"]["delta_edge"] > 0
                and v["44b6_22"]["delta_total"] >= 0 and v["6bba_22"]["delta_total"] >= 0
                and v["all22"]["readmit_nodes_final"] > 0)
    result = dict(status="complete_review_required", created=stamp(), movies=len(stems_all), rows=rows, gate_passed_22=gate,
                  gate=plan["gate"], limitation="Fit-domain post-processing replay on public-detector training movies "
                  "(HANDOFF section 4): gains below ~0.005 are unproven for the LB; unannotated detections are not negatives.")
    save_json(out / "analysis.json", result)
    save_json(out / "decision.json", dict(gate_passed_22=gate, extension75_run=len(stems_all) > 22, decided=stamp(),
                                          rule=plan["gate"], v12_all22=v["all22"], v12_all97=v.get("all97")))
    print(summary.to_string())
    print("GATE(22):", gate)


# ----------------------------------------------------------------------------- queue
def run(out):
    assert not (out / "status.json").exists(), "No blind restart; review status first"
    plan = read(out / "plan.json")
    os.environ["PYTHONUTF8"] = "1"
    q = Queue(out, plan["max_start_job_hours"])
    try:
        q.state.update(total_jobs=plan["total_jobs"], plan_sha256=sha(out / "plan.json"))
        q.save()
        for name, digest in plan["hashes"].items():
            assert sha(ROOT / name) == digest, ("input drift", name)
        me = lambda verb, *extra: [sys.executable, "-u", Path(__file__), verb, "--out", out, *extra]
        q.run("check_cache", me("check_cache"))
        for split in ["heldout12", "confirm10"]:
            q.run(f"control_replay_{split}", control_command(out, split, plan["splits"][split]))
            q.run(f"study_{split}", me("study", "--split", split))
        q.run("analyse_22", me("analyse"))
        gate = read(out / "decision.json")["gate_passed_22"]
        q.state["gate_passed_22"] = gate
        q.save()
        if gate:
            split = "extension75"
            q.run(f"control_replay_{split}", control_command(out, split, plan["splits"][split]))
            q.run(f"study_{split}", me("study", "--split", split))
            q.run("analyse_97", me("analyse"))
        for name, digest in plan["hashes"].items():
            assert sha(ROOT / name) == digest, ("input drift after run", name)
        files = [p for folder in ["replay", "study", "graphs"] for p in (out / folder).rglob("*") if p.is_file()]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in [".csv", ".json"]
                  and p.name not in ["plan.json", "status.json", "launch.json", "artifact_hashes.json"]]
        save_json(out / "artifact_hashes.json", {str(p.relative_to(out)): sha(p) for p in files})
        q.close("complete_review_required" if gate else "complete_22_gate_failed_review_required")
    except BaseException as exc:
        q.close("failed", exc)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "check_cache", "study", "analyse", "run"])
    p.add_argument("--out", type=Path, default=DEST)
    p.add_argument("--split", choices=SPLIT_ORDER)
    a = p.parse_args()
    out = a.out.resolve()
    out.relative_to(ROOT)
    if a.command == "study":
        study(out, a.split)
    else:
        globals()[a.command](out)
