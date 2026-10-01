"""Exact AST-selected C023 primary prefix; shortened only after live parity proof.

No replacement encoder, normalization, TTA or lookup is implemented. The exact
production statements stop before secondary inference; only requested original
first-seen windows execute. One movie per fresh process. No queue or training.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

import c063_frozen_spatial_capture as cap

ROOT = cap.ROOT
SOURCE = cap.SOURCE / 'scripts/predict_unet_transformer.py'


def prefix_ast():
    """Retain exact AST statements; fail if the reviewed source structure moves."""
    source = SOURCE.read_text(encoding='utf-8')
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'predict_video')
    loops = [(i, n) for i, n in enumerate(original.body) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'ws']
    assert len(loops) == 1
    loop_index, original_loop = loops[0]
    stops = [i for i, n in enumerate(original_loop.body) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == 'secondary_unet_out']
    assert len(stops) == 1 and isinstance(original_loop.body[stops[0]].value, ast.Constant)
    stop = stops[0]
    # The complete primary8TTA block remains a single unchanged If statement.
    primary = original_loop.body[:stop]
    assert isinstance(primary[-1], ast.If) and ast.unparse(primary[-1].test) == 'cfg.det_tta'
    assert 'unet_out = _unet_acc / _nv' in ast.unparse(primary[-1])
    assert 'secondary_model' not in ast.unparse(ast.Module(body=primary, type_ignores=[]))
    selected = copy.deepcopy(original)
    selected.name = '_c063_primary_prefix'
    selected.returns = None
    selected.body = copy.deepcopy(original.body[:loop_index])
    # Keep production window construction then narrow only to requested windows.
    selected.body += ast.parse("assert W == 2\nassert set(_c063_windows).issubset(window_starts)\nwindow_starts = list(_c063_windows)").body
    loop = copy.deepcopy(original_loop)
    loop.body = copy.deepcopy(primary) + ast.parse(
        "_c063_sink(ws, frame_indices, unet_out)\ndel imgs, unet_out, det_logits").body
    selected.body.append(loop)
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    # Prove the setup and encode/TTA statements themselves were not rewritten.
    assert ast.dump(ast.Module(body=selected.body[:loop_index], type_ignores=[])) == ast.dump(ast.Module(body=original.body[:loop_index], type_ignores=[]))
    assert ast.dump(ast.Module(body=loop.body[:stop], type_ignores=[])) == ast.dump(ast.Module(body=primary, type_ignores=[]))
    proof = dict(original_source_sha256=cap.sha(SOURCE),
        retained_setup_statements=loop_index, retained_primary_statements=stop,
        setup_ast_exact=True, primary_ast_exact=True,
        generated_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        stop_before='secondary_unet_out = None', added='requested-window subset; passive field callback; temporary cleanup')
    return module, proof


def movie_plan(movie):
    pairs, shape, spatial = cap.selection(movie)
    selected = pairs[pairs.eligible].copy().reset_index(drop=True)
    times = sorted(selected.t.astype(int).unique().tolist())
    windows = sorted({max(t - 1, 0) for t in times})
    frames = sorted({t for ws in windows for t in (ws, ws + 1)})
    assert windows and min(frames) >= 0 and max(frames) < shape[0]
    folder = ROOT / 'data/train' / f'{movie}.zarr'
    metadata = cap.read(folder / '0/zarr.json')
    assert metadata['chunk_grid']['configuration']['chunk_shape'] == [1, *shape[1:]]
    assert metadata['chunk_key_encoding'] == {'name': 'default', 'configuration': {'separator': '/'}}
    chunks = [folder / '0/c' / str(t) / '0/0/0' for t in frames]
    assert all(p.is_file() for p in chunks)
    return pairs, selected, dict(movie=movie, selected_times=times, first_seen_windows=windows,
        encoder_source_frames=frames, image_chunks=[str(p.relative_to(ROOT)) for p in chunks],
        raw_shape=shape, feature_grid=spatial.astype(int).tolist(),
        eligible_pairs=len(selected), original_pairs=len(pairs), excluded_pairs=len(pairs) - len(selected))


def dependencies(stems=cap.STEMS, reference_dir=None, proof_dir=None):
    """Exact image-frame footprint, including preceding context and frame1 for t0."""
    from v1284_capture_local import PRIMARY_WEIGHTS
    stems = [stems] if isinstance(stems, str) else list(stems)
    paths = {Path(__file__), Path(cap.__file__), cap.BASE, cap.C058 / 'plan.json',
             cap.C058 / 'output_hashes.json', PRIMARY_WEIGHTS, PRIMARY_WEIGHTS.parent / 'config.json',
             ROOT / 'src/v1284_capture_local.py', ROOT / 'src/c037_transformer_runtime.py'}
    for part in ['scripts', 'src']:
        paths.update(p for p in (cap.SOURCE / part).rglob('*.py') if '__pycache__' not in p.parts)
    for movie in stems:
        _, _, plan = movie_plan(movie)
        paths.update(ROOT / p for p in plan['image_chunks'])
        paths.update([ROOT / 'data/train' / f'{movie}.zarr/zarr.json',
            ROOT / 'data/train' / f'{movie}.zarr/0/zarr.json',
            cap.C058 / 'baseline' / f'{movie}.npz', cap.C058 / 'labels' / f'{movie}.csv'])
        if reference_dir is not None:
            paths.update(Path(reference_dir) / movie / name for name in [
                'summary.json', 'receipt.json', 'frame_provenance.json', 'features.npy',
                'targets.npz', 'labels.csv', 'all_original_pairs.csv', 'input_hashes.json'])
    if proof_dir is not None:
        paths.update(Path(proof_dir) / stem / 'proof.json' for stem in cap.STEMS)
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    return sorted(paths)


class SparseSink(cap.PassiveCapture):
    def __init__(self, index, selected, destination, plan):
        super().__init__(lambda _p, _t, arr, _f: arr, index, selected, destination, len(plan['selected_times']))
        self.plan = plan
        self.requested = set(plan['selected_times'])

    def accept_window(self, ws, indices, feature):
        assert indices == [ws, ws + 1] and feature.shape[:3] == (1, 2, cap.CHANNELS)
        for index, t in enumerate(indices):
            if t not in self.requested or ws != max(t - 1, 0):
                continue
            expected_index = 0 if t == 0 else 1
            assert index == expected_index
            self(Path(self.plan['movie']), int(t), np.empty((0, 4), np.float32), feature[:, index])

    def close(self):
        assert self.seen == self.requested
        assert self.written == set(range(len(self.selected)))
        self.output.flush()
        del self.output


def _write_targets(out, selected):
    # Identical array names, dtypes and row order to the passive adapter.
    np.savez_compressed(out / 'targets.npz', node_ids=selected.node_id.to_numpy(np.int64),
        gt_ids=selected.gt_id.to_numpy(np.int64), times=selected.t.to_numpy(np.int64),
        baseline_rows=selected.row.to_numpy(np.int64), cube_rows=selected.cube_row.to_numpy(np.int64),
        centers_native=selected[['center_z', 'center_y', 'center_x']].to_numpy(np.int64),
        centers_feature=selected[['feature_center_z', 'feature_center_y', 'feature_center_x']].to_numpy(np.float32),
        targets_um=selected[['dz_um', 'dy_um', 'dx_um']].to_numpy(np.float32),
        targets_grid=selected[['target_z_grid', 'target_y_grid', 'target_x_grid']].to_numpy(np.float32))


def _run(movie, output_dir, reference_dir=None, proof_dir=None):
    from v1284_capture_local import PRIMARY_WEIGHTS, notebook_env
    from c037_transformer_runtime import state_digest
    out = Path(output_dir).resolve() / movie
    out.relative_to(ROOT)
    assert not out.exists(), 'No overwrite of prefix extraction or partial run'
    out.mkdir(parents=True)
    start = time.perf_counter()
    hashes = {str(p.relative_to(ROOT)): cap.sha(p) for p in dependencies(movie, reference_dir, proof_dir)}
    cap.save(out / 'input_hashes.json', hashes)
    pairs, selected, plan = movie_plan(movie)
    cap.save(out / 'data_plan.json', plan)
    pairs.to_csv(out / 'all_original_pairs.csv', index=False)
    selected.to_csv(out / 'labels.csv', index=False)
    _write_targets(out, selected)
    runtime = cap.policy()
    env = notebook_env(cap.BASE)
    env.pop('BIOHUB_DIAGNOSTIC_ARM', None)
    os.environ.pop('BIOHUB_DIAGNOSTIC_ARM', None)
    os.environ.update(env)
    os.environ.update(V1284_MODE='candidate', V1284_HEAD=str(cap.HEAD),
        BIOHUB_CACHE_DIR='', BIOHUB_LOCAL_WORKING=str(out))
    assert os.environ['BIOHUB_EDGE_FEATURE_TTA'] == '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(cap.SOURCE / 'scripts'), str(cap.SOURCE / 'src')]
    assert 'c063_original_predictor' not in sys.modules, 'Run one movie per fresh process'
    spec = importlib.util.spec_from_file_location('c063_original_predictor', SOURCE)
    pu = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pu
    spec.loader.exec_module(pu)
    module, ast_proof = prefix_ast()
    cap.save(out / 'prefix_ast.json', ast_proof)
    (out / 'generated_prefix.py').write_text(ast.unparse(module) + '\n', encoding='utf-8')
    sink = SparseSink(pu._v1284_index, selected, out / 'features.npy', plan)
    pu.__dict__.update(_c063_windows=plan['first_seen_windows'], _c063_sink=sink.accept_window)
    exec(compile(module, str(SOURCE) + ':C063_exact_primary_prefix', 'exec'), pu.__dict__)
    model, window, down = pu.load_model(PRIMARY_WEIGHTS, torch.device('cuda'))
    assert window == 2 and tuple(down) == (1, 4, 4) and not model.training
    before = state_digest(model.state_dict())
    cfg = pu.PredictConfig(det_threshold=float(env['BIOHUB_DET_THRESHOLD']))
    assert cfg.det_tta
    mark = time.perf_counter()
    with (out / 'prefix.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        pu._c063_primary_prefix(model, ROOT / 'data/train' / movie, torch.device('cuda'),
            cfg, window_size=window, downsample=down)
    compute_seconds = time.perf_counter() - mark
    sink.close()
    assert state_digest(model.state_dict()) == before
    cap.save(out / 'frame_provenance.json', sink.receipts)
    cap.verify(hashes)
    summary = dict(status='passed_extraction_requires_proof' if reference_dir else 'passed',
        movie=movie, contract=cap.CONTRACT, plan=plan, ast=ast_proof, runtime=runtime,
        primary_state_sha256=before, primary_state_unchanged=True,
        full_feature_field_unchanged=True, no_secondary_or_association_or_ilp=True,
        original_known_pairs=len(pairs), eligible_pairs=len(selected), excluded_pairs=len(pairs) - len(selected),
        feature_shape=[len(selected), cap.CHANNELS, *cap.CUBE],
        feature_sha256=cap.sha(out / 'features.npy'), targets_sha256=cap.sha(out / 'targets.npz'),
        labels_sha256=cap.sha(out / 'labels.csv'), all_pairs_sha256=cap.sha(out / 'all_original_pairs.csv'),
        frame_provenance_sha256=cap.sha(out / 'frame_provenance.json'), input_files=len(hashes),
        compute_seconds=compute_seconds, seconds=time.perf_counter() - start,
        extractor_sha256=cap.sha(Path(__file__)), capture_sha256=cap.sha(Path(cap.__file__)),
        no_training=True, no_graph_intervention=True, no_score_gain_claim=True)
    cap.save(out / 'summary.json', summary)
    return out, summary


def prove(movie, output_dir, reference_dir):
    reference = Path(reference_dir).resolve() / movie
    expected = cap.read(reference / 'summary.json')
    assert expected['status'] == 'passed' and expected['actual_original_final_zero_graph_exact']
    assert expected['actual_official_zero_exact'] and expected['original_vs_passive_ilp_exact']
    assert expected['feature_sha256'] == cap.sha(reference / 'features.npy')
    assert expected['frame_provenance_sha256'] == cap.sha(reference / 'frame_provenance.json')
    out, summary = _run(movie, output_dir, reference_dir=Path(reference_dir).resolve())
    times = summary['plan']['selected_times']
    original_frames = {r['t']: r for r in cap.read(reference / 'frame_provenance.json')}
    frames = cap.read(out / 'frame_provenance.json')
    assert len(frames) == len(times)
    for row in frames:
        actual = original_frames[row['t']]
        for key in ['encoder_window', 'window_index', 'full_feature_shape', 'full_feature_sha256', 'selected_anchors']:
            assert row[key] == actual[key], ('Prefix field mismatch', movie, row['t'], key)
    # NPY containers and every numeric cube are exactly equal, not tolerance based.
    assert cap.sha(out / 'features.npy') == expected['feature_sha256']
    a, b = np.load(out / 'features.npy', mmap_mode='r'), np.load(reference / 'features.npy', mmap_mode='r')
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    for begin in range(0, len(a), 16):
        assert np.array_equal(a[begin:begin + 16], b[begin:begin + 16])
    cap.compare_npz(out / 'targets.npz', reference / 'targets.npz')
    for name in ['labels.csv', 'all_original_pairs.csv']:
        assert cap.sha(out / name) == cap.sha(reference / name), ('Identity manifest drift', name)
    cap.verify(cap.read(out / 'input_hashes.json'))
    proof = dict(status='passed', movie=movie, all_requested_fields_exact=True,
        full_feature_fields=len(frames), entire_fp32_cubes_exact=True,
        cube_shape=list(a.shape), labels_targets_exclusions_exact=True,
        original_full_pipeline_official_zero_reference=str(reference.relative_to(ROOT)),
        reference_summary_sha256=cap.sha(reference / 'summary.json'),
        source_sha256=cap.sha(SOURCE), extractor_sha256=cap.sha(Path(__file__)),
        capture_sha256=cap.sha(Path(cap.__file__)), generated_ast_sha256=summary['ast']['generated_ast_sha256'],
        shared_dependency_hashes={str(p.relative_to(ROOT)): cap.sha(p) for p in dependencies([])},
        primary_checkpoint_sha256=cap.sha(ROOT / 'artifacts/pilkwang_support50/weights/unet_transformer/split_0/edge_predictor_best.pth'),
        compute_seconds=summary['compute_seconds'], seconds=summary['seconds'], no_graph_intervention=True)
    cap.save(out / 'proof.json', proof)
    summary['status'] = 'passed_proven_exact_prefix'
    cap.save(out / 'summary.json', summary)
    print(proof, flush=True)


def extract(movie, output_dir, proof_dir):
    _, ast_proof = prefix_ast()
    for stem in cap.STEMS:
        proof = cap.read(Path(proof_dir) / stem / 'proof.json')
        assert proof['status'] == 'passed' and proof['all_requested_fields_exact'] and proof['entire_fp32_cubes_exact']
        assert proof['extractor_sha256'] == cap.sha(Path(__file__))
        assert proof['capture_sha256'] == cap.sha(Path(cap.__file__))
        assert proof['source_sha256'] == cap.sha(SOURCE)
        assert proof['generated_ast_sha256'] == ast_proof['generated_ast_sha256']
        cap.verify(proof['shared_dependency_hashes'])
    _, summary = _run(movie, output_dir, proof_dir=Path(proof_dir).resolve())
    print(summary, flush=True)


def controls(out):
    """AST and image-chunk metadata only. Does not load model or GPU tensors."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'controls.json').exists()
    module, proof = prefix_ast()
    compile(module, str(SOURCE) + ':control', 'exec')
    old = cap.read(cap.C058 / 'plan.json')['test_movies']
    rows = [movie_plan(movie)[2] for movie in old]
    assert len(rows) == 22
    result = dict(status='passed', ast=proof, movies=rows,
        total_pairs=sum(r['original_pairs'] for r in rows), eligible_pairs=sum(r['eligible_pairs'] for r in rows),
        requested_windows=sum(len(r['first_seen_windows']) for r in rows),
        image_chunks=sum(len(r['image_chunks']) for r in rows),
        dependencies=len(dependencies(old)), no_gpu_execution=True, no_model_loaded=True)
    cap.save(out / 'controls.json', result)
    (out / 'generated_prefix.py').write_text(ast.unparse(module) + '\n', encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'movies'}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['controls', 'prove', 'extract'])
    parser.add_argument('--movie')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--proof-dir', type=Path)
    args = parser.parse_args()
    if args.verb == 'controls':
        controls(args.output_dir)
    elif args.verb == 'prove':
        assert args.movie and args.reference_dir
        prove(args.movie, args.output_dir, args.reference_dir)
    else:
        assert args.movie and args.proof_dir
        extract(args.movie, args.output_dir, args.proof_dir)


if __name__ == '__main__':
    main()
