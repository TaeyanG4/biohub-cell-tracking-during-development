"""Passive C023 dense-feature capture at original fixed official matched anchors.

capture --movie STEM --output-dir DIR writes DIR/STEM. No training or queue.
The original and instrumented full predictors must have exact cache/graph
parity, followed by the existing C058 original replay/official zero proof.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tmp/c023_output/tracking_repo'
C058 = ROOT / 'experiments/candidates/c058_raw_localizer'
BASE = ROOT / 'experiments/candidates/c023_x138_head_stabilize_restore/biohub-c023-x138-head-stabilize-restore.ipynb'
HEAD = ROOT / 'artifacts/anvithpothula_v1284_head_s075/v1284_head.pt'
CUBE = (13, 13, 13)
RADIUS = 6
CHANNELS = 32
SPACING_UM = 1.625
DOWNSAMPLE = np.array([1., 4., 4.])
VOX = np.array([1.625, .40625, .40625])
STEMS = ('44b6_12dfb391', '6bba_05db0fb1')
CONTRACT = dict(version=1, channels=CHANNELS, cube_zyx=list(CUBE), dtype='float32',
    input_offsets_zyx=[-6, 6], output_offsets_zyx=[-5, 5], spacing_um=SPACING_UM,
    native_to_feature='(z,y/4,x/4), origin zero, no center rounding',
    temporal='first seen: t0 -> window[0,1] index0; t>=1 -> window[t-1,t] index1',
    representation='actual primary production eight-XY-view averaged UNet output',
    lookup='unchanged C023 v1284_coordinate_refinement.index_features trilinear',
    normalization='actual C023 complete-frame quantiles; lower clamp0, no upper clamp',
    anchors='all original C058 official-known pairs; no rematching',
    support='nonsynthetic and entire fractional-centered13cube inside actual feature field',
    excluded='retained unchanged in all_original_pairs.csv with cube_row=-1',
    purpose='conditional component diagnostic; GT is not a deployable eligibility mask',
    graph_modified=False, no_training=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()), encoding='utf-8')


def verify(manifest):
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, ('C063 input drift', name)


def split_for(movie):
    result = read(C058 / 'baseline' / f'{movie}.json')['split']
    assert result in ['heldout12', 'confirm10']
    return result


def _dependencies_one(movie):
    """Return concrete immutable inputs so the parent can pin before launch."""
    import c055_guarded_readmit as pp
    from v1284_capture_local import PRIMARY_WEIGHTS, SECONDARY_WEIGHTS
    split = split_for(movie)
    paths = {Path(__file__), BASE, HEAD, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS,
             PRIMARY_WEIGHTS.parent / 'config.json', SECONDARY_WEIGHTS.parent / 'config.json',
             C058 / 'plan.json', C058 / 'output_hashes.json',
             C058 / 'baseline' / f'{movie}.json', C058 / 'baseline' / f'{movie}.npz',
             C058 / 'labels' / f'{movie}.csv', pp.DEST / 'graphs' / split / f'{movie}_control.npz',
             pp.DEST / 'study' / f'{split}.csv', pp.run_dir(split) / 'edge_cache' / f'{movie}.npz'}
    for name in ['v1284_capture_local.py', 'c037_transformer_runtime.py',
                 'c058_localizer_baseline.py', 'c055_guarded_readmit.py',
                 'c055_guarded_readmit_stage.py', 'eval_pp_variants_local.py',
                 'evaluate_local.py', 'run_last_days_local.py',
                 'build_temporal_context_candidate.py', 'build_structured_candidate.py']:
        paths.add(ROOT / 'src' / name)
    for part in ['scripts', 'src']:
        paths.update(p for p in (SOURCE / part).rglob('*.py') if '__pycache__' not in p.parts)
    for part in [ROOT / 'data/train' / f'{movie}.zarr', ROOT / 'data/train' / f'{movie}.geff',
                 pp.pred_path(split, movie)]:
        paths.update(p for p in part.rglob('*') if p.is_file())
    inherited = read(C058 / 'plan.json')['hashes']
    paths.update(ROOT / p for p in inherited if any(s in p.lower() for s in ['vendor', 'deepcenter']))
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    return sorted(paths)


def dependencies(stems=STEMS):
    """Both benchmark movies by default; accepts a single movie or iterable."""
    if isinstance(stems, str):
        stems = [stems]
    return sorted({p for stem in stems for p in _dependencies_one(stem)})


def selection(movie):
    """Identity-preserving selection; labels never influence cube support."""
    labels = pd.read_csv(C058 / 'labels' / f'{movie}.csv')
    with np.load(C058 / 'baseline' / f'{movie}.npz') as z:
        ids, points, synthetic = z['ids'].copy(), z['txyz'].copy(), z['gap_synthetic'].copy()
    assert not labels.node_id.duplicated().any() and not labels.gt_id.duplicated().any()
    index = labels.row.to_numpy(np.int64)
    assert np.array_equal(ids[index], labels.node_id.to_numpy(np.int64))
    assert np.array_equal(points[index, 0], labels.t.to_numpy(np.int64))
    assert np.array_equal(synthetic[index], labels.gap_synthetic.to_numpy(bool))
    native = points[index, 1:].astype(np.float64)
    grid = native / DOWNSAMPLE
    shape = read(ROOT / 'data/train' / f'{movie}.zarr/0/zarr.json')['shape']
    spatial = (np.array(shape[1:]) + DOWNSAMPLE.astype(int) - 1) // DOWNSAMPLE.astype(int)
    supported = ((grid - RADIUS >= 0) & (grid + RADIUS <= spatial - 1)).all(axis=1)
    eligible = ~synthetic[index] & supported
    result = labels.copy().rename(columns={'eligible': 'historical_c058_raw_eligible'})
    for k, name in enumerate(['z', 'y', 'x']):
        result['center_' + name] = native[:, k].astype(np.int64)
        result['feature_center_' + name] = grid[:, k]
        result['target_' + name + '_grid'] = result['d' + name + '_um'] / SPACING_UM
    result['feature_support'] = supported
    result['eligible'] = eligible
    result['cube_row'] = -1
    result.loc[eligible, 'cube_row'] = np.arange(int(eligible.sum()))
    result['excluded_reason'] = np.where(synthetic[index], 'synthetic', np.where(supported, '', 'feature_boundary'))
    result['stem'] = movie
    target = result[['dz_um', 'dy_um', 'dx_um']].to_numpy()
    assert np.allclose(np.linalg.norm(target, axis=1), result.residual_um, rtol=0, atol=1e-9)
    assert (np.linalg.norm(target, axis=1) <= 7. + 1e-8).all()
    assert (np.abs(target / SPACING_UM) <= 5).all()
    assert int(eligible.sum()) > 0
    return result, shape, spatial


def policy():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    assert torch.cuda.is_available(), 'C063 live parity requires the established CUDA runtime'
    return dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), dtype='float32',
        tf32=False, flash_sdp=False, mem_efficient_sdp=False, math_sdp=True)


def _array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode()); h.update(str(array.dtype).encode()); h.update(array.tobytes())
    return h.hexdigest()


class PassiveCapture:
    def __init__(self, original, index_features, selected, destination, frames):
        self.original, self.index_features = original, index_features
        self.selected = selected.reset_index(drop=True)
        self.output = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
            shape=(len(selected), CHANNELS, *CUBE))
        grid = np.stack(np.meshgrid(*[np.arange(-RADIUS, RADIUS + 1)] * 3, indexing='ij'), axis=-1)
        self.offsets = grid.reshape(-1, 3).astype(np.float32)
        self.seen, self.written, self.receipts = set(), set(), []
        self.frames = frames

    def __call__(self, movie, t, arr, feature):
        # The only returned value is the original candidate-mode head result.
        original_arr = arr.copy()
        result = self.original(movie, t, arr, feature)
        assert np.array_equal(arr, original_arr), 'Original C023 head changed its input in place'
        assert int(t) not in self.seen and feature.dtype == torch.float32
        self.seen.add(int(t))
        field_before = _array_digest(feature.detach().cpu().numpy())
        rows = self.selected[self.selected.t == int(t)]
        # Read only eight anchors at a time to bound temporary gather memory.
        for begin in range(0, len(rows), 8):
            block = rows.iloc[begin:begin + 8]
            centers = block[['feature_center_z', 'feature_center_y', 'feature_center_x']].to_numpy(np.float32)
            query = centers[:, None, :] + self.offsets[None, :, :]
            assert ((query >= 0) & (query <= np.array(feature.shape[-3:]) - 1)).all()
            tensor = torch.from_numpy(query.reshape(1, -1, 3)).to(feature.device)
            mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=feature.device)
            values = self.index_features(None, feature, tensor, mask)[0]
            cubes = values.reshape(len(block), *CUBE, CHANNELS).permute(0, 4, 1, 2, 3).cpu().numpy()
            assert np.isfinite(cubes).all()
            center_query = torch.from_numpy(centers[None]).to(feature.device)
            center_mask = torch.ones(center_query.shape[:2], dtype=torch.bool, device=feature.device)
            exact_center = self.index_features(None, feature, center_query, center_mask)[0].cpu().numpy()
            assert np.array_equal(cubes[:, :, RADIUS, RADIUS, RADIUS], exact_center)
            locations = block.cube_row.to_numpy(np.int64)
            self.output[locations] = cubes
            self.written.update(locations.tolist())
        assert _array_digest(feature.detach().cpu().numpy()) == field_before, 'Passive field mutation'
        self.receipts.append(dict(t=int(t), encoder_window=[max(int(t) - 1, 0), max(int(t), 1)],
            window_index=0 if t == 0 else 1, full_feature_shape=list(feature.shape),
            full_feature_sha256=field_before, selected_anchors=len(rows),
            incoming_detections=len(arr), returned_coordinates_sha256=_array_digest(result)))
        return result

    def close(self):
        assert self.seen == set(range(self.frames)), 'Not all production first-seen frames were observed'
        assert self.written == set(range(len(self.selected))), 'Missing selected feature cubes'
        self.output.flush()
        del self.output


def compare_npz(left, right):
    with np.load(left) as a, np.load(right) as b:
        assert set(a.files) == set(b.files), (left, right, 'fields')
        for name in a.files:
            assert a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name]), (left, right, name)
    return dict(left=str(left.relative_to(ROOT)), right=str(right.relative_to(ROOT)), exact=True)


def _semantic_graph(ns, path):
    import eval_pp_variants_local as harness
    nodes, edges = harness.load_raw_graph(ns, path)
    return sorted((i, tuple(n[k] for k in ['t', 'z', 'y', 'x'])) for i, n in nodes.items()), sorted(
        (e['source_id'], e['target_id'], e['edge_prob']) for e in edges)


def capture(movie, output_dir):
    """Two live full predictors, passive cube capture and exact official zero replay."""
    import c055_guarded_readmit as pp
    import c058_localizer_baseline as frozen
    import eval_pp_variants_local as harness
    from c037_transformer_runtime import state_digest
    from v1284_capture_local import PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, notebook_env, prepare_repo
    out = Path(output_dir).resolve() / movie
    out.relative_to(ROOT)
    assert not out.exists(), 'Refuse to overwrite a registered capture or partial run'
    out.mkdir(parents=True)
    start = time.perf_counter()
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in dependencies(movie)}
    save(out / 'input_hashes.json', manifest)
    all_pairs, shape, spatial = selection(movie)
    selected = all_pairs[all_pairs.eligible].copy().reset_index(drop=True)
    all_pairs.to_csv(out / 'all_original_pairs.csv', index=False)
    selected.to_csv(out / 'labels.csv', index=False)
    np.savez_compressed(out / 'targets.npz', node_ids=selected.node_id.to_numpy(np.int64),
        gt_ids=selected.gt_id.to_numpy(np.int64), times=selected.t.to_numpy(np.int64),
        baseline_rows=selected.row.to_numpy(np.int64), cube_rows=selected.cube_row.to_numpy(np.int64),
        centers_native=selected[['center_z', 'center_y', 'center_x']].to_numpy(np.int64),
        centers_feature=selected[['feature_center_z', 'feature_center_y', 'feature_center_x']].to_numpy(np.float32),
        targets_um=selected[['dz_um', 'dy_um', 'dx_um']].to_numpy(np.float32),
        targets_grid=selected[['target_z_grid', 'target_y_grid', 'target_x_grid']].to_numpy(np.float32))
    runtime = policy()
    env = notebook_env(BASE)
    env.pop('BIOHUB_DIAGNOSTIC_ARM', None)
    os.environ.pop('BIOHUB_DIAGNOSTIC_ARM', None)
    os.environ.update(env)
    os.environ.update(V1284_MODE='candidate', V1284_HEAD=str(HEAD), BIOHUB_CACHE_DIR='',
        BIOHUB_LOCAL_WORKING=str(out), BIOHUB_SECONDARY_WEIGHTS=str(SECONDARY_WEIGHTS))
    repo = prepare_repo(SOURCE, out / '_work', BASE)
    sys.path[:0] = [str(repo / 'scripts'), str(repo / 'src')]
    assert 'predict_unet_transformer' not in sys.modules, 'Use one movie capture per fresh process'
    pu = importlib.import_module('predict_unet_transformer')
    dataspec = importlib.import_module('dataspec')
    pu.UNetNodeTransformer._index_features = pu._v1284_index
    cfg = pu.PredictConfig(det_threshold=float(env['BIOHUB_DET_THRESHOLD']), use_ilp=True,
        ilp_edge_weight=float(env.get('BIOHUB_ILP_EDGE_WEIGHT', '-1')),
        ilp_appearance_weight=float(env['BIOHUB_ILP_APPEARANCE_WEIGHT']),
        ilp_disappearance_weight=float(env['BIOHUB_ILP_DISAPPEARANCE_WEIGHT']),
        ilp_division_weight=float(env['BIOHUB_ILP_DIVISION_WEIGHT']))
    cfg.threshold = float(env['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'])
    original = pu._v1284_refine
    captured = PassiveCapture(original, pu._v1284_index, selected, out / 'features.npy', shape[0])
    models, model_proofs, phase_seconds, graphs = [], [], {}, {}
    original_loader = pu.load_model

    def observed_load(weights, device):
        model, window, down = original_loader(weights, device)
        assert window == 2 and tuple(down) == (1, 4, 4)
        assert not model.training
        models.append((model, str(weights.relative_to(ROOT)), state_digest(model.state_dict())))
        return model, window, down

    pu.load_model = observed_load
    for arm, hook in [('original', original), ('passive', captured)]:
        arm_out = out / arm
        arm_out.mkdir()
        dataspec.PREDICTIONS_PATH = arm_out / 'predictions'
        pu._CACHE_DIR = str(arm_out / 'edge_cache')
        pu._LOWDET.clear(); pu._CACHE_EDGES.clear()
        pu._v1284_refine = hook
        os.environ['BIOHUB_CACHE_DIR'] = pu._CACHE_DIR
        mark = time.perf_counter()
        with (arm_out / 'predict.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            pu.predict(ROOT / 'data/train', 0, out / 'unused_splits.json', PRIMARY_WEIGHTS,
                cfg, method='c063_passive_zero', debug_video=ROOT / 'data/train' / movie,
                unet_batch_size=4, evaluate=False)
        for model, path, before in models:
            assert state_digest(model.state_dict()) == before, ('Frozen model drift', path)
            model_proofs.append(dict(arm=arm, path=path, state_sha256=before, unchanged=True))
        models.clear()
        graphs[arm] = next((arm_out / 'predictions').rglob(f'{movie}.geff'))
        phase_seconds[arm] = time.perf_counter() - mark
    captured.close()
    pu._v1284_refine = original
    pu.load_model = original_loader
    save(out / 'frame_provenance.json', captured.receipts)
    compare = [compare_npz(out / 'original/edge_cache' / f'{movie}.npz', out / 'passive/edge_cache' / f'{movie}.npz')]
    split = split_for(movie)
    compare.append(compare_npz(out / 'passive/edge_cache' / f'{movie}.npz', pp.run_dir(split) / 'edge_cache' / f'{movie}.npz'))
    ns = harness.build_namespace(BASE, {}, out / 'passive/edge_cache')
    ns['TEST_DIR'] = ROOT / 'data/train'
    assert _semantic_graph(ns, graphs['original']) == _semantic_graph(ns, graphs['passive']), 'Passive ILP graph drift'
    assert _semantic_graph(ns, graphs['passive']) == _semantic_graph(ns, pp.pred_path(split, movie)), 'Cached C023 ILP graph drift'
    # Reuse the actual C058 passive baseline/official verifier against this
    # newly inferred graph and cache, without changing that pinned helper.
    original_pred_path, original_run_dir = pp.pred_path, pp.run_dir
    try:
        pp.pred_path = lambda requested_split, requested_movie: graphs['passive'] if requested_movie == movie and requested_split == split else original_pred_path(requested_split, requested_movie)
        pp.run_dir = lambda requested_split: out / 'passive' if requested_split == split else original_run_dir(requested_split)
        graph_path = frozen.capture(movie, split, out / 'zero_replay', ns=ns)
    finally:
        pp.pred_path, pp.run_dir = original_pred_path, original_run_dir
    with np.load(graph_path) as actual, np.load(C058 / 'baseline' / f'{movie}.npz') as expected:
        for name in ['ids', 'txyz', 'float_txyz', 'edges', 'gap_synthetic']:
            assert np.array_equal(actual[name], expected[name]), ('Final zero drift', name)
    proof = read(graph_path.with_suffix('.json'))
    old_proof = read(C058 / 'baseline' / f'{movie}.json')
    assert json.dumps(proof['official'], sort_keys=True) == json.dumps(old_proof['official'], sort_keys=True), 'Actual official zero score drift'
    verify(manifest)
    receipt = dict(status='passed', movie=movie, contract=CONTRACT, runtime=runtime,
        original_known_pairs=len(all_pairs), eligible_pairs=len(selected), excluded_pairs=len(all_pairs) - len(selected),
        tails_gt3_5_all=int((all_pairs.residual_um > 3.5).sum()),
        tails_gt3_5_eligible=int((selected.residual_um > 3.5).sum()),
        feature_shape=[len(selected), CHANNELS, *CUBE], feature_sha256=sha(out / 'features.npy'),
        labels_sha256=sha(out / 'labels.csv'), all_pairs_sha256=sha(out / 'all_original_pairs.csv'),
        targets_sha256=sha(out / 'targets.npz'), frame_provenance_sha256=sha(out / 'frame_provenance.json'),
        input_files=len(manifest), cache_comparisons=compare, model_proofs=model_proofs,
        original_vs_passive_ilp_exact=True, actual_c023_reference_ilp_exact=True,
        actual_original_final_zero_graph_exact=True, actual_official_zero_exact=True,
        original_graph_exact=True, passive_field_unchanged=True,
        original_baseline_proof=str(graph_path.with_suffix('.json').relative_to(ROOT)),
        phase_seconds=phase_seconds, seconds=time.perf_counter() - start,
        no_fit=True, no_graph_intervention=True, no_score_gain_claim=True)
    save(out / 'receipt.json', receipt)
    save(out / 'summary.json', receipt)
    print(json.dumps(receipt), flush=True)


def controls(out):
    """CPU-only geometry and actual production interpolation tests; no encode."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'controls.json'
    assert not destination.exists(), 'Refuse to overwrite controls'
    module = SOURCE / 'scripts/v1284_coordinate_refinement.py'
    tree = ast.parse(module.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'index_features')
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(module) + ':index_features', 'exec'), scope)
    zz, yy, xx = torch.meshgrid(torch.arange(17), torch.arange(19), torch.arange(21), indexing='ij')
    feature = torch.stack([.25 * zz + .5 * yy + .125 * xx + c for c in range(CHANNELS)])[None].float()
    centers = np.array([[8., 9.25, 10.75], [7., 8., 9.]], np.float32)
    selected = pd.DataFrame(dict(t=[0, 0], feature_center_z=centers[:, 0],
        feature_center_y=centers[:, 1], feature_center_x=centers[:, 2], cube_row=[0, 1]))
    source = np.array([[0, 1, 2, 3]], np.int16)
    original = lambda _movie, _t, arr, _field: arr.astype(np.float32) + np.array([[0, .125, .25, .5]], np.float32)
    path = out / 'control_features.npy'
    observer = PassiveCapture(original, scope['index_features'], selected, path, 1)
    returned = observer(Path(STEMS[0]), 0, source, feature)
    assert np.array_equal(returned, original(None, None, source, None))
    observer.close()
    with path.open('rb') as stream:
        cubes = np.load(stream)
    offsets = np.stack(np.meshgrid(*[np.arange(-6, 7)] * 3, indexing='ij'), axis=0)
    expected = (centers[:, :, None, None, None] + offsets[None]).astype(np.float32)
    expected = (.25 * expected[:, 0] + .5 * expected[:, 1] + .125 * expected[:, 2])[:, None] + np.arange(CHANNELS)[None, :, None, None, None]
    assert np.array_equal(cubes, expected), 'Cube axis/order or interpolation error'
    inventory = []
    for movie in STEMS:
        selected, shape, grid = selection(movie)
        inventory.append(dict(movie=movie, original_known_pairs=len(selected),
            eligible_pairs=int(selected.eligible.sum()), excluded_pairs=int((~selected.eligible).sum()),
            tails_eligible=int(((selected.residual_um > 3.5) & selected.eligible).sum()),
            feature_grid=grid.astype(int).tolist(), input_raw_shape=shape))
    proof = dict(status='passed', contract=CONTRACT, inventory=inventory,
        actual_production_interpolation=True, integer_and_fractional_centers_exact=True,
        xyz_axis_order_exact=True, passive_return_exact=True, passive_field_unchanged=True,
        no_model_inference=True, no_training=True, no_graph_intervention=True,
        interpolation_source_sha256=sha(module), control_features_sha256=sha(path))
    save(destination, proof)
    print(json.dumps(proof), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['capture', 'controls'])
    parser.add_argument('--movie')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.verb == 'capture':
        assert args.movie, '--movie required for capture'
        capture(args.movie, args.output_dir)
    else:
        controls(args.output_dir)


if __name__ == '__main__':
    main()
