#!/usr/bin/env python3
"""C056: complementary-detector guarded endpoint readmission on C023 (follow-up of C055).

Same replay protocol, harness, caches, control parity and V12 guard as C055; the
only change is the candidate source: retained Hengck StrongUNet peaks
(`src/cache_strongunet_gpu_peaks.py` model, fixed published threshold p >= 0.5)
instead of the primary detector's 0.94-0.965 range.  Peaks are cached once per
movie on the GPU (resumable); everything else is CPU replay.  No fitting, no
threshold sweep, no Kaggle write, no notebook edit.

    python src/c056_complementary_readmit.py prepare
    python src/c056_complementary_readmit.py run          # finite queue, hidden process
    python src/c056_complementary_readmit.py analyse

Pre-registered extension gate (all 22 movies): strongunet_guarded beats BOTH the
C023 control and the readmit_off diagnostic on total and adjusted-edge score,
both embryo subsets are >= 0 versus control, and readmitted nodes survive.  A
gain that only reproduces the readmit-off effect (C055, C015) does not pass.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import c055_guarded_readmit as c055  # noqa: E402
from c055_guarded_readmit import (BASE, GT_DIR, OLD, C046_GRAPHS, SPLIT_ORDER, compare_rows, harness,  # noqa: E402
                                  pred_path, read, read_rows, read_stems, run_dir, save_json, sha, split_stems)
from c056_complementary_readmit_stage import STRONGUNET_MIN_P, install_complementary_readmit  # noqa: E402
from run_last_days_local import Queue, stamp  # noqa: E402

DEST = ROOT / "experiments/candidates/c056_complementary_readmit"
C055_DIR = c055.DEST
PEAK_TOOL = ROOT / "src/cache_strongunet_gpu_peaks.py"
MODEL_DIR = ROOT / "artifacts/hengck_point_detector"
MISSED_22 = ROOT / "state/c055_diagnostics_20260929/missed_gt_nodes_22.csv"
VARIANTS = {"control": {}, "readmit_off": {"READMIT_RADIUS_UM": 0.0},
            "strongunet_guarded": {"READMIT_RADIUS_UM": 0.0, "COMPLEMENTARY_READMIT": 1}}


def peaks_dir(out):
    return out / "peaks"


# ----------------------------------------------------------------------------- prepare
def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "plan.json").exists(), "Registered source is immutable; use a new --out"
    assert read(C055_DIR / "cache_semantics.json")["status"] == "passed"
    stems = split_stems()
    for split, lst in stems.items():
        (out / f"{split}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    save_json(out / "variants.json", VARIANTS)
    files = [Path(__file__), ROOT / "src/c056_complementary_readmit_stage.py", ROOT / "src/c055_guarded_readmit.py",
             ROOT / "src/c055_guarded_readmit_stage.py", ROOT / "src/eval_pp_variants_local.py", ROOT / "src/run_last_days_local.py",
             PEAK_TOOL, MODEL_DIR / "00000030.pth", MODEL_DIR / "model_v5.py", BASE, c055.PREDICT_SCRIPT,
             harness.DEEPCENTER_CHECKPOINT, harness.DEEPCENTER_MANIFEST, MISSED_22,
             C055_DIR / "cache_semantics.json", C055_DIR / "plan.json"]
    files += [OLD / f"control_fp32_{s}.csv" for s in SPLIT_ORDER]
    files += [C055_DIR / "replay" / f"control_{s}.csv" for s in ["heldout12", "confirm10"]]
    files += [C055_DIR / "study" / f"{s}.csv" for s in ["heldout12", "confirm10"]]
    for split, lst in stems.items():
        for stem in lst:
            files.append(run_dir(split) / "edge_cache" / f"{stem}.npz")
            files += [p for p in pred_path(split, stem).rglob("*") if p.is_file()]
            files += [p for p in (GT_DIR / f"{stem}.geff").rglob("*") if p.is_file()]
            files += [p for p in (GT_DIR / f"{stem}.zarr").rglob("*") if p.is_file()]
            files.append(C046_GRAPHS / split / f"{stem}_off.npz")
    files += [p for p in out.iterdir() if p.is_file()]
    pinned = {}
    for p in files:
        assert p.exists(), p
        pinned[str(p.relative_to(ROOT))] = sha(p)
    save_json(out / "plan.json", dict(
        created=stamp(), splits=stems, variants=VARIANTS, hashes=pinned, total_jobs=12, jobs_if_gate_fails=8,
        max_start_job_hours=8, estimated_minutes=240,
        estimate_basis="StrongUNet peak caching (GPU, per movie) for 22 then 75 movies plus C055-like CPU replays (3 arms)",
        gate="strongunet_guarded vs control AND vs readmit_off on all22: total>0 and edge>0; both embryos >=0 vs control; readmit_nodes_final>0",
        strongunet_threshold=STRONGUNET_MIN_P,
        policy="Fixed V12 constants and fixed published StrongUNet threshold; C023 readmission off in the new arms; no fitting, sweep, "
               "Kaggle write or notebook edit. Unannotated detections are never treated as negatives."))
    (out / "README.md").write_text(README, encoding="utf-8")
    print("Prepared C056:", len(pinned), "pinned inputs", flush=True)


README = """# C056 complementary-detector guarded readmission on C023

Follow-up of C055 (`experiments/candidates/c055_guarded_readmit/REVIEW.md`).
C055 showed that V12's 0.94-0.965 primary-detector range holds almost no
annotated missed cells (6 of 213 on 22 movies), while 101 missed cells have no
primary peak at all and the 2026-09-28 priority review found StrongUNet peaks
at 16 of 21 such cells on the visible movies.  This study feeds the unchanged
V12 guard (step <= 3 um, residual <= 1.5 um, separation >= 1.5 um, margin 0.5,
bridges, budget min(100, 0.2%)) with StrongUNet peaks at the fixed published
threshold p >= 0.5, replacing C023's own readmission (off in the new arms).

Arms: `control`, `readmit_off` (must equal C055's rows exactly),
`strongunet_guarded`.  Peaks: `peaks/<stem>_peaks.parquet` cached with the
existing `src/cache_strongunet_gpu_peaks.py` model on `data/train` movies.
Diagnostic ceiling: `diagnose.json` (missed GT nodes with a StrongUNet peak).
Gate and outputs: see the driver docstring; REVIEW.md after completion.
No Kaggle operation is performed by this queue.
"""


# ----------------------------------------------------------------------------- peak cache (GPU)
def cache_peaks(out, split):
    import torch
    import cache_strongunet_gpu_peaks as tool
    torch.set_num_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    stems = read_stems(out / f"{split}.txt")
    folder = peaks_dir(out)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (folder / f"{s}_peaks.parquet").exists()]
    print(f"{split}: {len(stems)} movies, {len(todo)} to cache", flush=True)
    if not todo:
        return
    device = torch.device("cuda:0")
    Model = tool.load_model_class()
    model = Model(in_channels=1, channels=(64, 128, 256), node_channels=1, gradient_checkpointing=False).to(device)
    ckpt = torch.load(tool.CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("device", torch.cuda.get_device_name(0), "checkpoint_epoch", ckpt.get("epoch"), "min_threshold", tool.MIN_THRESHOLD, flush=True)
    rows = []
    for stem in todo:
        t0 = time.time()
        pred = tool.infer_sample(model, GT_DIR / f"{stem}.zarr", device)
        tmp = folder / f"{stem}_peaks.tmp.parquet"
        pred.to_parquet(tmp, index=False)
        tmp.replace(folder / f"{stem}_peaks.parquet")
        rows.append(dict(dataset=stem, frames=int(pred.t.nunique()), peaks=int(len(pred)), peaks_p05=int((pred.p >= 0.5).sum()),
                         seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    summary = folder / f"summary_{split}.csv"
    pd.concat([pd.read_csv(summary), pd.DataFrame(rows)]).to_csv(summary, index=False) if summary.exists() else pd.DataFrame(rows).to_csv(summary, index=False)


# ----------------------------------------------------------------------------- ceiling diagnostic
def diagnose(out):
    """For the 213 GT nodes the C023 control misses (C055 diagnostic), the nearest StrongUNet peak."""
    from scipy.spatial import cKDTree
    scale = np.array([1.625, 0.40625, 0.40625])
    missed = pd.read_csv(MISSED_22)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ns = harness.build_namespace(BASE.resolve(), {}, None)
    rows = []
    for (split, stem), sub in missed.groupby(["split", "stem"]):
        gt_nodes, _ = ns["graph_to_plain"](ns["graph_from_geff"](GT_DIR / f"{stem}.geff"))
        pk = pd.read_parquet(peaks_dir(out) / f"{stem}_peaks.parquet")
        trees = {}
        for t, part in pk[pk.p >= STRONGUNET_MIN_P].groupby("t"):
            trees[int(t)] = (cKDTree(part[["z", "y", "x"]].to_numpy(float) * scale), part.p.to_numpy())
        trees_all = {}
        for t, part in pk.groupby("t"):
            trees_all[int(t)] = (cKDTree(part[["z", "y", "x"]].to_numpy(float) * scale), part.p.to_numpy())
        for _, m in sub.iterrows():
            t, z, y, x = gt_nodes[int(m.gt_id)]
            q = np.array([z, y, x]) * scale
            d05, i05 = trees[int(t)][0].query(q, k=1) if int(t) in trees else (np.inf, -1)
            dall, iall = trees_all[int(t)][0].query(q, k=1) if int(t) in trees_all else (np.inf, -1)
            rows.append(dict(split=split, stem=stem, embryo=stem[:4], gt_id=int(m.gt_id), t=int(t), gt_edges=int(m.gt_edges),
                             primary_peak_um=float(m.nearest_peak_um), primary_peak_score=float(m.nearest_peak_score),
                             strongunet_p05_um=float(d05), strongunet_p05_score=float(trees[int(t)][1][i05]) if i05 >= 0 else np.nan,
                             strongunet_any_um=float(dall), strongunet_any_score=float(trees_all[int(t)][1][iall]) if iall >= 0 else np.nan))
    df = pd.DataFrame(rows)
    df["primary_bucket"] = np.where(df.primary_peak_um > 3, "no_primary_peak", np.where(df.primary_peak_score > 0.965, "primary_dropped", "primary_low"))
    df["strongunet_p05_within3um"] = df.strongunet_p05_um <= 3.0
    df.to_csv(out / "diagnose.csv", index=False)
    summ = df.groupby(["embryo", "primary_bucket", "strongunet_p05_within3um"]).agg(n=("gt_id", "size"), gt_edges=("gt_edges", "sum")).reset_index()
    result = dict(status="done", missed_gt_nodes=len(df), strongunet_p05_within3um=int(df.strongunet_p05_within3um.sum()),
                  by_primary_bucket=df.groupby("primary_bucket").strongunet_p05_within3um.agg(["size", "sum"]).rename(columns={"size": "n", "sum": "with_strongunet_p05"}).to_dict("index"),
                  table=summ.to_dict("records"),
                  note="Ceiling only: a nearby peak is not a recovered edge; the guard, the budget and later stages decide. Unmatched peaks are unknown, never negatives.")
    save_json(out / "diagnose.json", result)
    print(json.dumps(result, indent=1, default=str))


# ----------------------------------------------------------------------------- study (reuse C055 protocol)
def study(out, split):
    stems = read_stems(out / f"{split}.txt")
    stored = [r for r in read_rows(OLD / f"control_fp32_{split}.csv") if r["config"] == "as_configured"]
    cli_csv = C055_DIR / "replay" / f"control_{split}.csv"
    cli = [r for r in read_rows(cli_csv) if r["config"] == "as_configured"]
    n_stored = compare_rows(cli, stored, "c055_cli_vs_stored")
    lowdet = run_dir(split) / "edge_cache"
    graphs = out / "graphs" / split
    log_path = out / "study" / f"{split}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        ns = harness.build_namespace(BASE.resolve(), {}, lowdet)
    ns["TEST_DIR"] = GT_DIR.resolve()
    frames, heatmaps = harness.install_frame_caches(ns)
    install_complementary_readmit(ns, peaks_dir(out), graphs)
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
                ns["_C056_LABEL"] = label
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
                record = ns["_C056_RECORDS"].get((stem, label), {})
                row.update({k: v for k, v in record.items() if k not in ("stem", "label")})
                rows.append(row)
                print(f"{stem:15s} {label:18s} adj={row['adjusted_edge_jaccard']:.6f} edge tp/fp/fn={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                      f"div={row['div_tp']}/{row['div_fp']}/{row['div_fn']} nodes={row['nodes']} readmit={row.get('readmit_nodes', 0)}/{row.get('readmit_nodes_final', 0)} [{row['seconds']:.0f}s]", flush=True)
            frames.clear()
            heatmaps.clear()
    df = pd.DataFrame(rows)
    (out / "study").mkdir(exist_ok=True)
    df.to_csv(out / "study" / f"{split}.csv", index=False)
    n_cli = compare_rows(df[df.config == "control"].to_dict("records"), cli, "inprocess_vs_c055_cli")
    n_off = 0
    prior = C055_DIR / "study" / f"{split}.csv"
    if prior.exists():
        n_off = compare_rows(df[df.config == "readmit_off"].to_dict("records"),
                             [r for r in read_rows(prior) if r["config"] == "readmit_off"], "readmit_off_vs_c055")
    graph_exact, graph_diff = 0, []
    for stem in stems:
        mine = np.load(graphs / f"{stem}_control.npz")
        ref = np.load(C046_GRAPHS / split / f"{stem}_off.npz")
        if np.array_equal(mine["ids"], ref["ids"]) and np.array_equal(mine["txyz"], ref["txyz"]) and np.array_equal(mine["edges"], ref["edges"]):
            graph_exact += 1
        else:
            graph_diff.append(stem)
    save_json(out / f"parity_{split}.json", dict(status="passed", split=split, movies=len(stems), c055_cli_vs_stored_rows=n_stored,
                                                 inprocess_vs_c055_cli_rows=n_cli, readmit_off_vs_c055_rows=n_off,
                                                 control_final_graphs_equal_c046_off=graph_exact, control_final_graphs_differ_from_c046_off=graph_diff,
                                                 minutes=round((time.time() - t_start) / 60, 1)))
    print(json.dumps(read(out / f"parity_{split}.json")))


# ----------------------------------------------------------------------------- analysis (C055's, rebound)
def analyse(out):
    source = inspect.getsource(c055.analyse)
    for before, after in [('"v12_guarded"', '"strongunet_guarded"')]:
        assert before in source, before
        source = source.replace(before, after)
    scope = dict(c055.__dict__, VARIANTS=VARIANTS)
    exec(compile(source, str(Path(__file__)) + "::reuse_analyse", "exec"), scope)
    scope["analyse"](out)
    plan = read(out / "plan.json")
    rows = read(out / "analysis.json")["rows"]
    by = {(r["group"], r["arm"]): r for r in rows}
    s, o = by[("all22", "strongunet_guarded")], by[("all22", "readmit_off")]
    gate = bool(s["delta_total"] > 0 and s["delta_edge"] > 0 and s["score"] > o["score"] and s["adjusted_edge"] > o["adjusted_edge"]
                and by[("44b6_22", "strongunet_guarded")]["delta_total"] >= 0 and by[("6bba_22", "strongunet_guarded")]["delta_total"] >= 0
                and s["readmit_nodes_final"] > 0)
    dec = read(out / "decision.json")
    dec.update(gate_passed_22=gate, rule=plan["gate"], vs_readmit_off_total=s["score"] - o["score"], vs_readmit_off_edge=s["adjusted_edge"] - o["adjusted_edge"])
    save_json(out / "decision.json", dec)
    res = read(out / "analysis.json")
    res["gate_passed_22"] = gate
    res["gate"] = plan["gate"]
    save_json(out / "analysis.json", res)
    print("GATE(22, vs control and vs readmit_off):", gate)


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
        for split in ["heldout12", "confirm10"]:
            q.run(f"cache_peaks_{split}", me("cache_peaks", "--split", split))
        q.run("diagnose", me("diagnose"))
        for split in ["heldout12", "confirm10"]:
            q.run(f"study_{split}", me("study", "--split", split))
        q.run("analyse_22", me("analyse"))
        gate = read(out / "decision.json")["gate_passed_22"]
        q.state["gate_passed_22"] = gate
        q.save()
        if gate:
            split = "extension75"
            assert (C055_DIR / "replay" / f"control_{split}.csv").exists(), "C055 extension control rows required"
            q.run(f"cache_peaks_{split}", me("cache_peaks", "--split", split))
            q.run(f"study_{split}", me("study", "--split", split))
            q.run("analyse_97", me("analyse"))
        for name, digest in plan["hashes"].items():
            assert sha(ROOT / name) == digest, ("input drift after run", name)
        files = [p for folder in ["peaks", "study", "graphs"] for p in (out / folder).rglob("*") if p.is_file()]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in [".csv", ".json"]
                  and p.name not in ["plan.json", "status.json", "launch.json", "artifact_hashes.json"]]
        save_json(out / "artifact_hashes.json", {str(p.relative_to(out)): sha(p) for p in files})
        q.close("complete_review_required" if gate else "complete_22_gate_failed_review_required")
    except BaseException as exc:
        q.close("failed", exc)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "cache_peaks", "diagnose", "study", "analyse", "run"])
    p.add_argument("--out", type=Path, default=DEST)
    p.add_argument("--split", choices=SPLIT_ORDER)
    a = p.parse_args()
    out = a.out.resolve()
    out.relative_to(ROOT)
    if a.command in ("cache_peaks", "study"):
        globals()[a.command](out, a.split)
    else:
        globals()[a.command](out)
