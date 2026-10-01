"""C063 fixed source-only component fit and original-pair residual diagnostic.

No graph is edited or scored here. Known-pair capture is a supervised component
probe, never a GT-based deployment mask. A later all-node integration needs its
own unchanged controls and runtime acceptance decision. The caller owns queues.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

import c063_frozen_spatial_model as model_code

ROOT = Path(__file__).resolve().parents[1]
C058 = ROOT / "experiments/candidates/c058_raw_localizer"
EMBRYOS = ("44b6", "6bba")
VOX = np.array([1.625, .40625, .40625])
KEYS = ["row", "node_id", "gt_id", "t"]
RESIDUAL = ["dz_um", "dy_um", "dx_um"]
RECIPE = dict(model_code.RECIPE, folds="source44b6->opposite6bba; source6bba->opposite44b6",
    evaluation="both source and opposite on all16931 original known pairs; excluded shift0",
    residual_sign="GT-minus-original-anchor; prediction error is shift-minus-target",
    graph_intervention=False, no_gt_deployment_mask=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pin(paths):
    return {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify(hashes):
    for path, expected in hashes.items():
        assert sha(ROOT / path) == expected, ("C063 input changed", path)


def policy(device):
    device = torch.device(device)
    torch.set_num_threads(4)
    torch.manual_seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        assert torch.cuda.is_available(), "Requested CUDA runtime unavailable"
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def expected_stems():
    stems = sorted(read(C058 / "plan.json")["test_movies"])
    assert len(stems) == 22 and {s[:4] for s in stems} == set(EMBRYOS)
    return stems


class CaptureEvidence:
    """Read-only hash and parity chain for full passive or proven prefix data.

    Prefix movies receive a representation-equivalence proof, never invented
    per-movie ILP or official-score flags. Every nested declared input is hashed.
    """

    def __init__(self):
        import c063_frozen_spatial_capture as capture
        self.capture = capture
        self.hashes, self.summaries, self.proof_roots = {}, {}, set()
        self.primary = ROOT / "artifacts/pilkwang_support50/weights/unet_transformer/split_0/edge_predictor_best.pth"
        self.production = capture.SOURCE / "scripts/predict_unet_transformer.py"
        self.extractor = ROOT / "src/c063_frozen_spatial_extract.py"

    def track(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT))
        if key not in self.hashes:
            self.hashes[key] = sha(path)
        if expected is not None:
            assert self.hashes[key] == expected, ("C063 evidence hash mismatch", key)
        return self.hashes[key]

    def json(self, path):
        self.track(path)
        return read(path)

    def manifest(self, path):
        records = self.json(path)
        assert isinstance(records, dict) and records
        normalized = {}
        for key, expected in records.items():
            assert isinstance(expected, str) and len(expected) == 64
            absolute = (ROOT / key).resolve()
            relative = str(absolute.relative_to(ROOT))
            self.track(absolute, expected)
            normalized[relative] = expected
        return normalized

    def required(self, records, paths):
        for path in paths:
            key = str(Path(path).resolve().relative_to(ROOT))
            assert key in records, ("Missing required evidence input", key)
            self.track(path, records[key])

    def common(self, folder, summary):
        assert summary["movie"] == folder.name and summary["contract"] == self.capture.CONTRACT
        assert summary["runtime"]["dtype"] == "float32" and not summary["runtime"]["tf32"]
        assert not summary["runtime"]["flash_sdp"] and not summary["runtime"]["mem_efficient_sdp"]
        assert summary["runtime"]["math_sdp"] and summary["no_graph_intervention"]
        for filename, key in [("features.npy", "feature_sha256"), ("targets.npz", "targets_sha256"),
                              ("labels.csv", "labels_sha256"), ("all_original_pairs.csv", "all_pairs_sha256"),
                              ("frame_provenance.json", "frame_provenance_sha256")]:
            self.track(folder / filename, summary[key])
        records = self.manifest(folder / "input_hashes.json")
        assert len(records) == summary["input_files"]
        self.required(records, [Path(self.capture.__file__), self.capture.BASE,
            self.primary, self.primary.parent / "config.json", self.production,
            self.capture.SOURCE / "scripts/v1284_coordinate_refinement.py",
            C058 / "baseline" / f"{folder.name}.npz", C058 / "labels" / f"{folder.name}.csv"])
        return records

    def full(self, folder):
        folder = Path(folder).resolve()
        key = str(folder)
        if key in self.summaries:
            return self.summaries[key]
        summary = self.json(folder / "summary.json")
        assert summary == self.json(folder / "receipt.json")
        assert summary["status"] == "passed" and summary["no_fit"]
        for flag in ["original_vs_passive_ilp_exact", "actual_c023_reference_ilp_exact",
                     "actual_original_final_zero_graph_exact", "actual_official_zero_exact",
                     "original_graph_exact", "passive_field_unchanged"]:
            assert summary[flag], (folder.name, flag)
        records = self.common(folder, summary)
        self.required(records, [self.capture.HEAD])
        primary_rows = [p for p in summary["model_proofs"] if (ROOT / p["path"]).resolve() == self.primary]
        assert len(primary_rows) == 2 and {p["arm"] for p in primary_rows} == {"original", "passive"}
        assert all(p["unchanged"] for p in summary["model_proofs"])
        assert len({p["state_sha256"] for p in primary_rows}) == 1
        # Bind the actual original-writer/official result rather than copying flags.
        zero_path = ROOT / summary["original_baseline_proof"]
        zero = self.json(zero_path)
        old = self.json(C058 / "baseline" / f"{folder.name}.json")
        assert zero["status"] == "passed" and zero["official"] == old["official"]
        assert zero["exact_c055_ids_txyz_edges"] and zero["zero_rounding_exact"]
        self.track(zero_path.with_suffix(".npz"), zero["saved_sha256"])
        with np.load(zero_path.with_suffix(".npz")) as a, np.load(C058 / "baseline" / f"{folder.name}.npz") as b:
            for name in ["ids", "txyz", "float_txyz", "edges", "gap_synthetic"]:
                assert np.array_equal(a[name], b[name]), (folder.name, "full zero graph", name)
        self.summaries[key] = summary
        return summary

    def prefix_summary(self, folder):
        import c063_frozen_spatial_extract as extract
        folder = Path(folder).resolve()
        summary = self.json(folder / "summary.json")
        assert summary["status"] in ("passed", "passed_proven_exact_prefix")
        for key in ["primary_state_unchanged", "full_feature_field_unchanged",
                    "no_secondary_or_association_or_ilp", "no_training", "no_graph_intervention"]:
            assert summary[key], (folder.name, key)
        records = self.common(folder, summary)
        self.required(records, [self.extractor])
        self.track(self.extractor, summary["extractor_sha256"])
        self.track(Path(self.capture.__file__), summary["capture_sha256"])
        module, current_ast = extract.prefix_ast()
        assert summary["ast"] == current_ast == self.json(folder / "prefix_ast.json")
        self.track(folder / "generated_prefix.py")
        assert (folder / "generated_prefix.py").read_text(encoding="utf-8") == ast.unparse(module) + "\n"
        assert current_ast["setup_ast_exact"] and current_ast["primary_ast_exact"]
        self.track(self.production, current_ast["original_source_sha256"])
        plan = self.json(folder / "data_plan.json")
        assert plan == summary["plan"] == extract.movie_plan(folder.name)[2]
        frames = self.json(folder / "frame_provenance.json")
        assert len(frames) == len(plan["selected_times"])
        assert sorted(row["t"] for row in frames) == plan["selected_times"]
        selected = pd.read_csv(folder / "labels.csv")
        for row in frames:
            t = row["t"]
            assert row["encoder_window"] == [max(t - 1, 0), max(t, 1)]
            assert row["window_index"] == (0 if t == 0 else 1)
            assert row["full_feature_shape"] == [1, 32, *plan["feature_grid"]]
            assert row["selected_anchors"] == int((selected.t == t).sum())
        return summary, records

    def prove_pair(self, proof_root):
        proof_root = Path(proof_root).resolve()
        if str(proof_root) in self.proof_roots:
            return
        primary_states = set()
        for stem in self.capture.STEMS:
            folder = proof_root / stem
            proof = self.json(folder / "proof.json")
            assert proof["status"] == "passed" and proof["movie"] == stem
            assert proof["all_requested_fields_exact"] and proof["entire_fp32_cubes_exact"]
            assert proof["labels_targets_exclusions_exact"] and proof["no_graph_intervention"]
            summary, records = self.prefix_summary(folder)
            assert summary["status"] == "passed_proven_exact_prefix"
            for path, expected in proof["shared_dependency_hashes"].items():
                self.track(ROOT / path, expected)
                key = str((ROOT / path).resolve().relative_to(ROOT))
                assert records[key] == expected
            for path, key in [(self.production, "source_sha256"), (self.extractor, "extractor_sha256"),
                              (Path(self.capture.__file__), "capture_sha256"), (self.primary, "primary_checkpoint_sha256")]:
                self.track(path, proof[key])
            assert proof["generated_ast_sha256"] == summary["ast"]["generated_ast_sha256"]
            reference = (ROOT / proof["original_full_pipeline_official_zero_reference"]).resolve()
            reference.relative_to(ROOT)
            assert reference.name == stem
            full = self.full(reference)
            self.track(reference / "summary.json", proof["reference_summary_sha256"])
            self.required(records, [reference / name for name in ["summary.json", "receipt.json", "input_hashes.json",
                "frame_provenance.json", "features.npy", "targets.npz", "labels.csv", "all_original_pairs.csv"]])
            assert summary["feature_sha256"] == full["feature_sha256"]
            assert summary["feature_shape"] == full["feature_shape"] == proof["cube_shape"]
            for key in ["labels_sha256", "all_pairs_sha256"]:
                assert summary[key] == full[key]
            with np.load(folder / "targets.npz") as a, np.load(reference / "targets.npz") as b:
                assert set(a.files) == set(b.files)
                assert all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a.files)
            frames = self.json(folder / "frame_provenance.json")
            originals = {row["t"]: row for row in self.json(reference / "frame_provenance.json")}
            assert proof["full_feature_fields"] == len(frames)
            for row in frames:
                for key in ["encoder_window", "window_index", "full_feature_shape", "full_feature_sha256", "selected_anchors"]:
                    assert row[key] == originals[row["t"]][key], (stem, row["t"], key)
            recorded_states = {p["state_sha256"] for p in full["model_proofs"] if (ROOT / p["path"]).resolve() == self.primary}
            assert recorded_states == {summary["primary_state_sha256"]}
            primary_states.update(recorded_states)
        assert len(primary_states) == 1
        self.proof_roots.add(str(proof_root))

    def movie(self, folder):
        folder = Path(folder).resolve()
        if (folder / "receipt.json").exists():
            return self.full(folder)
        summary, records = self.prefix_summary(folder)
        if summary["status"] == "passed_proven_exact_prefix":
            proof_root = folder.parent
        else:
            proof_paths = [ROOT / key for key in records if Path(key).name == "proof.json"]
            assert len(proof_paths) == 2 and {p.parent.name for p in proof_paths} == set(self.capture.STEMS)
            roots = {p.parent.parent for p in proof_paths}
            assert len(roots) == 1
            proof_root = roots.pop()
        self.prove_pair(proof_root)
        for stem in self.capture.STEMS:
            proof_summary = self.json(proof_root / stem / "summary.json")
            assert proof_summary["primary_state_sha256"] == summary["primary_state_sha256"]
            assert proof_summary["ast"] == summary["ast"] and proof_summary["contract"] == summary["contract"]
        return summary


def dependencies(capture_dir, *, partial=False, verify_chain=True):
    """Concrete inputs including the verified nested passive/prefix proof chain."""
    capture_dir = Path(capture_dir)
    expected = expected_stems()
    stems = sorted(p.name for p in capture_dir.iterdir() if p.is_dir() and (p / "summary.json").exists())
    assert stems and set(stems) <= set(expected)
    if not partial:
        assert stems == expected, "Need every original22 capture before component fit"
    paths = [Path(__file__), Path(model_code.__file__), C058 / "plan.json"]
    for stem in stems:
        paths.extend(capture_dir / stem / name for name in
                     ["summary.json", "input_hashes.json", "frame_provenance.json", "all_original_pairs.csv", "labels.csv", "targets.npz", "features.npy"])
        if (capture_dir / stem / "receipt.json").exists():
            paths.append(capture_dir / stem / "receipt.json")
        paths.extend([C058 / "labels" / f"{stem}.csv", C058 / "baseline" / f"{stem}.npz",
                      ROOT / "data/train" / f"{stem}.zarr/0/zarr.json"])
    assert all(p.is_file() for p in paths)
    if verify_chain:
        evidence = CaptureEvidence()
        for path in paths:
            evidence.track(path)
        for stem in stems:
            evidence.movie(capture_dir / stem)
        paths = [ROOT / key for key in evidence.hashes]
    return sorted(set(paths))


def load_data(capture_dir, *, partial=False):
    """Verify capture identity, geometry and artifacts without rematching GT."""
    capture_dir = Path(capture_dir)
    paths = dependencies(capture_dir, partial=partial, verify_chain=False)
    evidence = CaptureEvidence()
    for path in paths:
        evidence.track(path)
    hashes = evidence.hashes
    data = {}
    for summary_path in sorted(capture_dir.glob("*/summary.json")):
        stem, folder = summary_path.parent.name, summary_path.parent
        proof = evidence.movie(folder)
        for filename, key in [("features.npy", "feature_sha256"), ("labels.csv", "labels_sha256"),
                              ("all_original_pairs.csv", "all_pairs_sha256"), ("targets.npz", "targets_sha256")]:
            assert hashes[str((folder / filename).resolve().relative_to(ROOT))] == proof[key], (stem, filename)
        original = pd.read_csv(C058 / "labels" / f"{stem}.csv")
        all_pairs = pd.read_csv(folder / "all_original_pairs.csv")
        labels = pd.read_csv(folder / "labels.csv")
        assert len(original) == len(all_pairs) == proof["original_known_pairs"]
        assert np.array_equal(original[KEYS].to_numpy(), all_pairs[KEYS].to_numpy())
        assert np.array_equal(original[RESIDUAL].to_numpy(), all_pairs[RESIDUAL].to_numpy())
        assert original.node_id.is_unique and original.gt_id.is_unique
        with np.load(C058 / "baseline" / f"{stem}.npz") as stored:
            graph = {key: stored[key].copy() for key in ("ids", "txyz", "gap_synthetic")}
        rows = all_pairs.row.to_numpy(np.int64)
        assert np.array_equal(graph["ids"][rows], all_pairs.node_id.to_numpy(np.int64))
        assert np.array_equal(graph["txyz"][rows, 0], all_pairs.t.to_numpy(np.int64))
        assert np.array_equal(graph["txyz"][rows, 1:], all_pairs[["center_z", "center_y", "center_x"]].to_numpy())
        shape = read(ROOT / "data/train" / f"{stem}.zarr/0/zarr.json")["shape"]
        grid = graph["txyz"][rows, 1:] / np.array([1., 4., 4.])
        grid_shape = (np.array(shape[1:]) + [0, 3, 3]) // [1, 4, 4]
        supported = ((grid - 6 >= 0) & (grid + 6 <= grid_shape - 1)).all(axis=1)
        eligible = ~graph["gap_synthetic"][rows].astype(bool) & supported
        assert np.array_equal(eligible, all_pairs.eligible.to_numpy(bool)), stem
        selected = all_pairs[eligible]
        assert np.array_equal(labels[KEYS].to_numpy(), selected[KEYS].to_numpy())
        assert np.array_equal(labels.cube_row.to_numpy(), np.arange(len(labels)))
        assert (all_pairs.loc[~eligible, "cube_row"] == -1).all()
        cubes = np.load(folder / "features.npy", mmap_mode="r")
        assert cubes.dtype == np.float32 and cubes.shape == (len(labels), 32, 13, 13, 13)
        with np.load(folder / "targets.npz") as target:
            targets = target["targets_um"].copy()
            for field, column in [("node_ids", "node_id"), ("gt_ids", "gt_id"), ("times", "t"),
                                  ("baseline_rows", "row"), ("cube_rows", "cube_row")]:
                assert np.array_equal(target[field], labels[column].to_numpy(np.int64)), (stem, field)
            assert np.array_equal(targets, labels[RESIDUAL].to_numpy(np.float32))
            assert np.array_equal(target["targets_grid"], (labels[RESIDUAL].to_numpy(np.float64) / 1.625).astype(np.float32))
        assert targets.dtype == np.float32 and np.isfinite(targets).all()
        assert (np.linalg.norm(targets, axis=1) <= 7 + 1e-5).all()
        assert len(labels) == proof["eligible_pairs"] > 0
        data[stem] = dict(cubes=cubes, targets=targets, labels=labels, all_pairs=all_pairs,
                          graph=graph, shape=shape)
    if not partial:
        assert sorted(data) == expected_stems()
        assert sum(len(d["all_pairs"]) for d in data.values()) == 16931
    return data, hashes


def sample_batch(data, rng, device):
    """Existing source-movie-uniform then pair-uniform-with-replacement sampler."""
    stem = sorted(data)[int(rng.integers(len(data)))]
    item = data[stem]
    indices = rng.integers(len(item["cubes"]), size=RECIPE["batch_size"])
    cubes = torch.from_numpy(np.asarray(item["cubes"][indices], dtype=np.float32)).to(device)
    targets = torch.from_numpy(item["targets"][indices].copy()).to(device)
    return cubes, targets, stem, indices


@torch.inference_mode()
def decoded_batches(model, cubes, device):
    count = len(cubes)
    shifts, raw, uniform, projected = np.zeros((count, 3), np.float32), np.zeros((count, 3), np.float32), np.zeros(count, bool), np.zeros(count, bool)
    for begin in range(0, count, RECIPE["batch_size"]):
        end = min(begin + RECIPE["batch_size"], count)
        batch = torch.from_numpy(np.array(cubes[begin:end], dtype=np.float32, copy=True)).to(device)
        result = model.predict(batch)
        shifts[begin:end] = result["shift_um"].cpu().numpy()
        raw[begin:end] = result["raw_mean_um"].cpu().numpy()
        uniform[begin:end] = result["uniform"].cpu().numpy()
        projected[begin:end] = result["projected"].cpu().numpy()
    assert np.isfinite(shifts).all() and (np.linalg.norm(shifts, axis=1) <= 7 + 1e-5).all()
    return dict(shift=shifts, raw=raw, uniform=uniform, projected=projected)


def zero_proof(data, device):
    model = model_code.FrozenSpatialLocalizer().to(device).eval()
    result = []
    for stem, item in sorted(data.items()):
        decoded = decoded_batches(model, item["cubes"], device)
        assert np.count_nonzero(decoded["shift"]) == 0 and decoded["uniform"].all()
        all_shift = np.zeros((len(item["all_pairs"]), 3), np.float32)
        all_shift[item["all_pairs"].eligible.to_numpy(bool)] = decoded["shift"]
        targets = item["all_pairs"][RESIDUAL].to_numpy()
        assert np.array_equal(targets - all_shift, targets)
        result.append(dict(stem=stem, original_pairs=len(targets), captured_pairs=len(decoded["shift"]),
                           all_original_residuals_exact=True, original_pair_ids_unchanged=True))
    return dict(status="passed", movies=result, original_pair_count=sum(r["original_pairs"] for r in result),
                exact_zero_shift=True, no_official_graph_claim=True)


def train_step(model, optimizer, scheduler, cubes, targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model_code.conditional_loss(model(cubes), targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def fit(capture_dir, out, source, device="cuda"):
    assert source in EMBRYOS
    out = Path(out)
    checkpoint = out / "models" / f"{source}.pt"
    receipt_path = checkpoint.with_suffix(".json")
    assert not checkpoint.exists() and not receipt_path.exists(), "Refuse repeat of fixed fit"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    device = policy(device)
    all_data, hashes = load_data(capture_dir)
    data = {s: d for s, d in all_data.items() if s.startswith(source + "_")}
    assert data and not any(s[:4] != source for s in data)
    zero = zero_proof(data, device)
    model = model_code.FrozenSpatialLocalizer().to(device)
    optimizer, scheduler = model_code.make_optimizer(model)
    rng = np.random.default_rng(RECIPE["seed"])
    counts = {s: np.zeros(len(d["cubes"]), np.int64) for s, d in data.items()}
    movie_steps = {s: 0 for s in data}
    history, sequence = [], hashlib.sha256()
    synchronize(device)
    start = time.perf_counter()
    for step in range(1, RECIPE["steps"] + 1):
        cubes, targets, stem, indices = sample_batch(data, rng, device)
        assert stem.startswith(source + "_")
        np.add.at(counts[stem], indices, 1)
        movie_steps[stem] += 1
        sequence.update(stem.encode() + b"\0" + indices.astype("<i8").tobytes())
        loss = train_step(model, optimizer, scheduler, cubes, targets)
        if step == 1 or step % 100 == 0:
            row = dict(step=step, loss=loss, seconds=time.perf_counter() - start)
            history.append(row)
            print(json.dumps(dict(source=source, **row)), flush=True)
    synchronize(device)
    fit_seconds = time.perf_counter() - start
    assert sum(movie_steps.values()) == 1200 and sum(int(v.sum()) for v in counts.values()) == 38400
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = dict(state_dict=state, recipe=RECIPE, train_embryo=source, training_stems=sorted(data),
                   steps=1200, model_source_sha256=sha(model_code.__file__), fit_source_sha256=sha(__file__),
                   input_hashes=hashes, sampled_examples=38400, sample_sequence_sha256=sequence.hexdigest())
    torch.save(payload, checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["recipe"] == RECIPE and saved["training_stems"] == sorted(data)
    assert all(torch.equal(state[k], saved["state_dict"][k]) for k in state)
    reload = model_code.FrozenSpatialLocalizer().to(device).eval()
    reload.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    first = data[sorted(data)[0]]["cubes"][:8]
    x = torch.from_numpy(np.array(first, copy=True)).to(device)
    with torch.inference_mode():
        before, after = model(x), reload(x)
        assert torch.equal(before, after), "Checkpoint reload logit drift"
        a, b = model_code.decode_logits(before), model_code.decode_logits(after)
        assert all(torch.equal(a[k], b[k]) for k in a), "Checkpoint reload decoder drift"
    counter_rows = []
    for stem, item in sorted(data.items()):
        rows = item["labels"][["row", "node_id", "gt_id", "t", "cube_row"]].copy()
        rows["stem"], rows["source_embryo"], rows["sample_draws"] = stem, source, counts[stem]
        counter_rows.append(rows)
    counter_path = checkpoint.parent / f"{source}_sample_counts.csv"
    pd.concat(counter_rows, ignore_index=True).to_csv(counter_path, index=False)
    verify(hashes)
    receipt = dict(status="passed", source=source, opposite=EMBRYOS[1] if source == EMBRYOS[0] else EMBRYOS[0],
        recipe=RECIPE, steps=1200, sampled_examples=38400, source_only_samples=True,
        training_stems=sorted(data), movie_step_counts=movie_steps,
        movie_sample_counts={s: int(v.sum()) for s, v in counts.items()},
        unique_sampled_pairs={s: int((v > 0).sum()) for s, v in counts.items()},
        sample_sequence_sha256=sequence.hexdigest(), history=history, fit_seconds=fit_seconds,
        checkpoint_sha256=sha(checkpoint), sample_counts_sha256=sha(counter_path),
        exact_reload_state=True, exact_reload_logits=True, exact_reload_decoder=True,
        zero_proof=zero, input_hashes=hashes, final_checkpoint_only=True,
        frozen_backbone_limitation="Backbone saw both embryos; only new head uses whole-embryo holdout",
        no_graph_score_claim=True)
    save(receipt_path, receipt)
    print(json.dumps(dict(status="passed", source=source, fit_seconds=fit_seconds, checkpoint=str(checkpoint))), flush=True)
    return receipt


def load_model(out, source, data, device):
    path = Path(out) / "models" / f"{source}.pt"
    receipt = read(path.with_suffix(".json"))
    assert receipt["status"] == "passed" and receipt["checkpoint_sha256"] == sha(path)
    assert receipt["exact_reload_state"] and receipt["exact_reload_logits"] and receipt["exact_reload_decoder"]
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["recipe"] == RECIPE and saved["train_embryo"] == source and saved["steps"] == 1200
    assert saved["training_stems"] == sorted(s for s in data if s.startswith(source + "_"))
    assert saved["model_source_sha256"] == sha(model_code.__file__) and saved["fit_source_sha256"] == sha(__file__)
    verify(saved["input_hashes"])
    model = model_code.FrozenSpatialLocalizer().to(device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, path


def ownership_diagnostics(all_pairs, graph, shape, shifts):
    """Prediction-only nearest-original-node check; diagnostic only, no guard."""
    conflict, before_conflict = np.zeros(len(all_pairs), bool), np.zeros(len(all_pairs), bool)
    outside = np.zeros(len(all_pairs), bool)
    for t, selected in all_pairs.groupby("t", sort=True):
        positions = selected.index.to_numpy()
        rows = selected.row.to_numpy(np.int64)
        frame_rows = np.flatnonzero(graph["txyz"][:, 0] == t)
        origin = graph["txyz"][rows, 1:] * VOX
        points = graph["txyz"][frame_rows, 1:] * VOX
        proposed = origin + shifts[positions]
        outside[positions] = ((proposed < 0) | (proposed > (np.array(shape[1:]) - 1) * VOX)).any(axis=1)
        if len(frame_rows) < 2:
            continue
        tree = cKDTree(points)
        for queries, flags in [(origin, before_conflict), (proposed, conflict)]:
            distance, nearest = tree.query(queries, k=2)
            is_other = frame_rows[nearest] != rows[:, None]
            nearest_other = np.where(is_other, distance, np.inf).min(axis=1)
            own_distance = np.linalg.norm(queries - origin, axis=1)
            flags[positions] = nearest_other <= own_distance + 1e-9
    return conflict, before_conflict, outside


def pair_diagnostics(stem, item, decoded, source):
    """Keep every original pair; excluded positions receive exact zero shift."""
    rows = item["all_pairs"].copy().reset_index(drop=True)
    mask = rows.eligible.to_numpy(bool)
    shifts = np.zeros((len(rows), 3), np.float32)
    raw = np.zeros_like(shifts)
    shifts[mask], raw[mask] = decoded["shift"], decoded["raw"]
    uniform, projected = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    uniform[mask], projected[mask] = decoded["uniform"], decoded["projected"]
    assert np.count_nonzero(shifts[~mask]) == 0
    target = rows[RESIDUAL].to_numpy(np.float64)
    after = target - shifts
    assert np.array_equal(after[~mask], target[~mask])
    for axis, name in enumerate("zyx"):
        rows[f"shift_{name}_um"] = shifts[:, axis]
        rows[f"raw_mean_{name}_um"] = raw[:, axis]
        rows[f"before_signed_error_{name}_um"] = -target[:, axis]
        rows[f"after_signed_error_{name}_um"] = -after[:, axis]
    rows["before_3d_um"], rows["after_3d_um"] = np.linalg.norm(target, axis=1), np.linalg.norm(after, axis=1)
    rows["before_absz_um"], rows["after_absz_um"] = np.abs(target[:, 0]), np.abs(after[:, 0])
    assert np.allclose(rows.before_3d_um, rows.residual_um, rtol=0, atol=1e-9)
    rows["uniform"], rows["projected"] = uniform, projected
    conflict, original_conflict, outside = ownership_diagnostics(rows, item["graph"], item["shape"], shifts)
    rows["ownership_conflict"], rows["original_ownership_conflict"] = conflict, original_conflict
    rows["new_ownership_conflict"], rows["outside_image"] = conflict & ~original_conflict, outside
    rows["source_embryo"], rows["embryo"], rows["stem"] = source, stem[:4], stem
    rows["domain"] = "source_fit" if source == stem[:4] else "opposite_embryo"
    return rows


def predict(capture_dir, out, source, device="cuda"):
    assert source in EMBRYOS
    out = Path(out)
    destination = out / "evaluation" / f"pairs_{source}.csv"
    assert not destination.exists() and not destination.with_suffix(".json").exists(), "Refuse repeat prediction"
    device = policy(device)
    data, hashes = load_data(capture_dir)
    model, checkpoint = load_model(out, source, data, device)
    start, results = time.perf_counter(), []
    for stem, item in sorted(data.items()):
        decoded = decoded_batches(model, item["cubes"], device)
        results.append(pair_diagnostics(stem, item, decoded, source))
    combined = pd.concat(results, ignore_index=True)
    assert len(combined) == 16931 and not combined.duplicated(["stem", "node_id"]).any()
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    verify(hashes)
    proof = dict(status="passed", source=source, checkpoint_sha256=sha(checkpoint),
        prediction_sha256=sha(destination), all_original_pairs=16931, movies=22,
        eligible=int(combined.eligible.sum()), excluded=int((~combined.eligible).sum()),
        original_ids_and_targets_exact=True, excluded_exact_zero=True, no_rematching=True,
        ownership_is_diagnostic_only=True, no_gt_deployment_mask=True,
        seconds=time.perf_counter()-start, input_hashes=hashes, no_graph_score_claim=True)
    save(destination.with_suffix(".json"), proof)
    print(json.dumps({k: v for k, v in proof.items() if k != "input_hashes"}), flush=True)
    return proof


def strata(frame):
    return dict(all_original=np.ones(len(frame), bool), good_le2_5=frame.before_3d_um.to_numpy() <= 2.5,
                tail3d_gt3_5=frame.before_3d_um.to_numpy() > 3.5,
                axialtail_gt3_5=frame.before_absz_um.to_numpy() > 3.5,
                eligible=frame.eligible.to_numpy(bool), excluded_unchanged=~frame.eligible.to_numpy(bool))


def summarize(frame):
    if len(frame) == 0:
        return dict(n=0)
    result = dict(n=len(frame), eligible=int(frame.eligible.sum()),
                  ownership_conflicts=int(frame.ownership_conflict.sum()),
                  new_ownership_conflicts=int(frame.new_ownership_conflict.sum()),
                  outside_image=int(frame.outside_image.sum()), uniform=int(frame.uniform.sum()),
                  projected=int(frame.projected.sum()))
    for metric in ("3d", "absz"):
        before, after = frame[f"before_{metric}_um"], frame[f"after_{metric}_um"]
        result.update({f"before_{metric}_mean_um": float(before.mean()),
                       f"after_{metric}_mean_um": float(after.mean()),
                       f"delta_{metric}_mean_um": float((after-before).mean()),
                       f"before_{metric}_median_um": float(before.median()),
                       f"after_{metric}_median_um": float(after.median()),
                       f"improved_{metric}": int((after < before).sum()),
                       f"worsened_{metric}": int((after > before).sum())})
    for axis in "zyx":
        for phase in ("before", "after"):
            result[f"{phase}_signed_bias_{axis}_um"] = float(frame[f"{phase}_signed_error_{axis}_um"].mean())
    return result


def analyse(capture_dir, out):
    out = Path(out)
    destination = out / "analysis"
    assert not destination.exists(), "Refuse overwrite of fixed component decision"
    data, hashes = load_data(capture_dir)
    frames, prediction_paths = [], []
    for source in EMBRYOS:
        path = out / "evaluation" / f"pairs_{source}.csv"
        proof = read(path.with_suffix(".json"))
        assert proof["status"] == "passed" and proof["prediction_sha256"] == sha(path)
        checkpoint = out / "models" / f"{source}.pt"
        fit_proof = read(checkpoint.with_suffix(".json"))
        assert proof["checkpoint_sha256"] == fit_proof["checkpoint_sha256"] == sha(checkpoint)
        assert fit_proof["source"] == source and fit_proof["source_only_samples"]
        assert fit_proof["training_stems"] == sorted(s for s in data if s.startswith(source + "_"))
        assert fit_proof["steps"] == 1200 and fit_proof["sampled_examples"] == 38400
        assert fit_proof["exact_reload_state"] and fit_proof["exact_reload_logits"] and fit_proof["exact_reload_decoder"]
        frame = pd.read_csv(path)
        assert len(frame) == 16931 and set(frame.source_embryo) == {source}
        assert sorted(frame.stem.unique()) == expected_stems()
        for stem, group in frame.groupby("stem", sort=True):
            original = data[stem]["all_pairs"]
            assert np.array_equal(group[KEYS].to_numpy(), original[KEYS].to_numpy())
            assert np.array_equal(group[RESIDUAL].to_numpy(), original[RESIDUAL].to_numpy())
            assert np.array_equal(group.eligible.to_numpy(bool), original.eligible.to_numpy(bool))
            assert set(group.embryo) == {stem[:4]}
            assert set(group.domain) == {"source_fit" if source == stem[:4] else "opposite_embryo"}
            shifts = group[["shift_z_um", "shift_y_um", "shift_x_um"]].to_numpy()
            target = group[RESIDUAL].to_numpy()
            assert np.isfinite(shifts).all() and (np.linalg.norm(shifts, axis=1) <= 7 + 1e-5).all()
            for phase, residual in [("before", target), ("after", target-shifts)]:
                assert np.allclose(group[f"{phase}_3d_um"], np.linalg.norm(residual, axis=1), rtol=0, atol=1e-9)
                assert np.allclose(group[f"{phase}_absz_um"], np.abs(residual[:, 0]), rtol=0, atol=1e-9)
                for axis, name in enumerate("zyx"):
                    assert np.allclose(group[f"{phase}_signed_error_{name}_um"], -residual[:, axis], rtol=0, atol=1e-9)
            excluded = ~group.eligible
            assert np.count_nonzero(group.loc[excluded, ["shift_z_um", "shift_y_um", "shift_x_um"]].to_numpy()) == 0
            assert np.array_equal(group.loc[excluded, "before_3d_um"], group.loc[excluded, "after_3d_um"])
        frames.append(frame)
        prediction_paths.extend([path, path.with_suffix(".json"), checkpoint, checkpoint.with_suffix(".json")])
    pairs = pd.concat(frames, ignore_index=True)
    assert not pairs.duplicated(["source_embryo", "stem", "node_id"]).any()
    summaries, movie_summaries = [], []
    for (source, embryo), group in pairs.groupby(["source_embryo", "embryo"], sort=True):
        for name, mask in strata(group).items():
            summaries.append(dict(source=source, embryo=embryo, opposite=source != embryo,
                                   stratum=name, **summarize(group[mask])))
        for stem, movie in group.groupby("stem", sort=True):
            for name, mask in strata(movie).items():
                movie_summaries.append(dict(source=source, embryo=embryo, opposite=source != embryo,
                                           stem=stem, stratum=name, **summarize(movie[mask])))
    summary = pd.DataFrame(summaries)
    movies = pd.DataFrame(movie_summaries)
    gates = []
    for embryo in EMBRYOS:
        part = summary[(summary.embryo == embryo) & summary.opposite].set_index("stratum")
        conditions = {}
        for stratum in ("all_original", "tail3d_gt3_5", "axialtail_gt3_5"):
            row = part.loc[stratum]
            conditions[stratum] = bool(row.n > 0 and row.delta_3d_mean_um < 0 and row.delta_absz_mean_um < 0)
        good = part.loc["good_le2_5"]
        conditions["good_not_worse"] = bool(good.n > 0 and good.delta_3d_mean_um <= 1e-12 and good.delta_absz_mean_um <= 1e-12)
        wins = {}
        for stratum in ("tail3d_gt3_5", "axialtail_gt3_5"):
            chosen = movies[(movies.embryo == embryo) & movies.opposite & (movies.stratum == stratum)]
            winners = chosen[(chosen.n > 0) & (chosen.delta_3d_mean_um < 0) & (chosen.delta_absz_mean_um < 0)]
            wins[stratum] = sorted(winners.stem.tolist())
            conditions[f"{stratum}_more_than_one_movie_win"] = len(winners) > 1
        gates.append(dict(embryo=embryo, conditions=conditions, tail_movie_wins=wins,
                           passed=all(conditions.values())))
    passed = all(gate["passed"] for gate in gates)
    decision = dict(status="review_required" if passed else "hold", component_gate_passed=passed,
        gates=gates, paired_original_count_per_source=16931, original_movies=22,
        original_ids_targets_exact=True, excluded_zero_exact=True, no_rematching=True,
        criterion="Both opposite directions:3D and absz overall/tails improve;good means not worse;both tail strata win>1 movie each",
        next_step="Independent original-pair/ownership review before any all-node integration" if passed else "Close this fixed component recipe; no decoder/strength/epoch sweep",
        source_fit_is_not_validation=True, no_graph_score_claim=True, no_leaderboard_gain_claim=True,
        frozen_backbone_saw_both_embryos=True, no_gt_deployment_mask=True,
        input_hashes=dict(hashes, **pin(prediction_paths)))
    destination.mkdir(parents=True)
    pairs.to_csv(destination / "all_original_paired_diagnostics.csv", index=False)
    summary.to_csv(destination / "stratum_summary.csv", index=False)
    movies.to_csv(destination / "movie_stratum_summary.csv", index=False)
    verify(decision["input_hashes"])
    save(destination / "decision.json", decision)
    print(json.dumps({k: v for k, v in decision.items() if k != "input_hashes"}), flush=True)
    return decision


def benchmark(capture_dir, out, source="44b6", device="cuda"):
    """Only an explicit caller runs30 disposable steps; never save a checkpoint."""
    assert source in EMBRYOS
    destination = Path(out) / "benchmark.json"
    assert not destination.exists(), "Refuse repeated timing experiment"
    device = policy(device)
    data, hashes = load_data(capture_dir, partial=True)
    zero = zero_proof(data, device)
    source_data = {s: d for s, d in data.items() if s.startswith(source + "_")}
    assert source_data
    model = model_code.FrozenSpatialLocalizer().to(device)
    optimizer, scheduler = model_code.make_optimizer(model)
    rng = np.random.default_rng(RECIPE["seed"])
    counts = {s: 0 for s in source_data}
    synchronize(device)
    start = time.perf_counter()
    for _ in range(30):
        cubes, targets, stem, indices = sample_batch(source_data, rng, device)
        train_step(model, optimizer, scheduler, cubes, targets)
        counts[stem] += len(indices)
    synchronize(device)
    elapsed = time.perf_counter() - start
    assert sum(counts.values()) == 960
    verify(hashes)
    result = dict(status="passed", source=source, source_only=True, actual_steps=30,
                  actual_samples=960, seconds=elapsed, estimated1200step_seconds=elapsed*40,
                  sample_counts=counts, checkpoint_saved=False, disposable_timing_only=True,
                  no_efficacy_output=True, zero_proof=zero, input_hashes=hashes,
                  timing_limitation="30steps on passed capture subset; full dataset IO may differ")
    save(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_hashes"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=["fit", "predict", "analyse", "benchmark"])
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=EMBRYOS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    capture_dir, out = args.capture_dir.resolve(), args.out.resolve()
    capture_dir.relative_to(ROOT)
    out.relative_to(ROOT)
    if args.verb == "analyse":
        analyse(capture_dir, out)
    else:
        if args.source is None:
            parser.error("--source is required for fit, predict and benchmark")
        globals()[args.verb](capture_dir, out, args.source, args.device)


if __name__ == "__main__":
    main()
