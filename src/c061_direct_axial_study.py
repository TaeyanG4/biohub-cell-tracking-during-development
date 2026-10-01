"""Finite direct-GEFF axial component study; no graph mutation or Kaggle writes.

Reuse the existing frame reader, cropper, hash helpers and Queue. This driver
computes fixed-pair coordinate diagnostics, not a new competition metric.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from c055_guarded_readmit import read, save_json, sha
from run_last_days_local import Queue, stamp
import c058_raw_localizer_study as old
import c061_axial_model as model_code
import c061_axial_data as data_code

DEST = ROOT / 'experiments/candidates/c061_direct_axial'
EMBRYOS = ('44b6', '6bba')
VOX = np.array([1.625, .40625, .40625])


def policy():
    old.numeric_policy()
    torch.manual_seed(6101)
    np.random.seed(6101)


def source_data(out, embryo):
    """Read only this biological source's registered crop files."""
    result = {}
    expected = sorted(data_code.load_manifest(out, 'direct').query('embryo == @embryo').stem.unique())
    paths = sorted((out / 'direct').glob(embryo + '*.npy'))
    if out.name != 'benchmark_data':
        assert [p.stem for p in paths] == expected, ('Missing source crops', embryo)
    for path in paths:
        crops = np.load(path, mmap_mode='r')
        if len(crops):
            assert tuple(crops.shape[1:]) == (21, 49, 49), path
            result[path.stem] = crops
    assert result and all(s.startswith(embryo + '_') for s in result)
    return result


def sample(data, rng):
    stem = sorted(data)[int(rng.integers(len(data)))]
    ix = rng.integers(len(data[stem]), size=36)
    expanded = torch.from_numpy(np.asarray(data[stem][ix], dtype=np.float32)[:, None]).cuda()
    shifts = torch.as_tensor(np.tile(np.arange(-4, 5), 4)[rng.permutation(36)], device='cuda')
    crops, target = model_code.shifted_batch(expanded, shifts)
    crops, _ = model_code.reflect_xy_batch(crops)
    assert np.array_equal(np.bincount((target.long() + 4).cpu().numpy(), minlength=9), np.full(9, 4))
    return crops, target, stem, ix


def train(out, embryo):
    policy()
    data = source_data(out, embryo)
    model = model_code.AxialLocalizer().cuda()
    opt, scheduler = model_code.make_optimizer(model)
    rng = np.random.default_rng(6101)
    counts = {s: 0 for s in data}
    classes = np.zeros(9, dtype=np.int64)
    history = []
    start = time.perf_counter()
    for step in range(1, 1201):
        x, y, stem, ix = sample(data, rng)
        assert stem.startswith(embryo + '_')
        counts[stem] += len(x)
        classes += np.bincount((y.long() + 4).cpu().numpy(), minlength=9)
        opt.zero_grad(set_to_none=True)
        loss = model_code.loss(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            item = dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter() - start)
            history.append(item)
            print(json.dumps(item), flush=True)
    assert np.array_equal(classes, np.full(9, 4800)) and sum(counts.values()) == 43200
    folder = out / 'models'
    folder.mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.cpu().state_dict(), recipe=model_code.RECIPE,
                    train_embryo=embryo, training_stems=sorted(data), steps=1200), folder / f'{embryo}.pt')
    save_json(folder / f'{embryo}.json', dict(status='passed', train_embryo=embryo,
        training_stems=sorted(data), recipe=model_code.RECIPE, steps=1200,
        class_counts=classes.tolist(), sample_counts=counts, history=history,
        seconds=time.perf_counter() - start))


def load_model(out, source):
    saved = torch.load(out / 'models' / f'{source}.pt', map_location='cpu', weights_only=True)
    assert saved['recipe'] == model_code.RECIPE and saved['train_embryo'] == source
    assert saved['steps'] == 1200 and saved['training_stems'] == sorted(source_data(out, source))
    model = model_code.AxialLocalizer().cuda().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    return model


def predict(model, crops):
    with torch.inference_mode():
        decoded = model_code.decode(model(crops))
    return decoded['z_vox'].cpu().numpy().astype(np.int64), decoded['tied'].cpu().numpy().astype(bool)


def synthetic(out, source):
    policy()
    model = load_model(out, source)
    rows = []
    for embryo in EMBRYOS:
        for stem, expanded in source_data(out, embryo).items():
            for begin in range(0, len(expanded), 36):
                ix = np.arange(begin, min(begin + 36, len(expanded)))
                full = torch.from_numpy(np.asarray(expanded[ix], np.float32)[:, None]).cuda()
                for shift in range(-4, 5):
                    shifts = torch.full((len(ix),), shift, device='cuda', dtype=torch.int64)
                    crops, target = model_code.shifted_batch(full, shifts)
                    dz, tied = predict(model, crops)
                    for j, i in enumerate(ix):
                        rows.append(dict(source_embryo=source, embryo=embryo,
                            domain='own_source_fit' if source == embryo else 'opposite_embryo',
                            stem=stem, sample_index=int(i), shift_vox=shift, target_z_vox=-shift,
                            predicted_z_vox=int(dz[j]), tied=bool(tied[j]),
                            before_z_um=abs(shift) * VOX[0], after_z_um=abs(shift + dz[j]) * VOX[0],
                            signed_error_z_um=(shift + dz[j]) * VOX[0]))
    folder = out / 'evaluation'
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f'synthetic_{source}.csv', index=False)
    print(source, 'synthetic rows', len(rows), flush=True)


def real(out, source):
    policy()
    model = load_model(out, source)
    rows = []
    paths = sorted((out / 'real').glob('*.npy'))
    expected_manifest = data_code.load_manifest(out, 'real')
    assert [p.stem for p in paths] == sorted(expected_manifest.stem.unique())
    for path in paths:
        stem = path.stem
        crops = np.load(path, mmap_mode='r')
        labels = pd.read_csv(path.with_suffix('.csv'))
        assert len(labels) == len(crops)
        expected = expected_manifest[expected_manifest.stem == stem]
        assert np.array_equal(labels[['node_id', 'gt_id', 'row']].to_numpy(),
                              expected[['node_id', 'gt_id', 'row']].to_numpy())
        graph = old.load_baseline(old.DEST, stem)
        meta = old.metadata(stem)
        proposal = np.zeros(len(labels), np.int64)
        ties = np.zeros(len(labels), bool)
        for begin in range(0, len(crops), 36):
            ix = np.arange(begin, min(begin + 36, len(crops)))
            x = torch.from_numpy(np.asarray(crops[ix], np.float32)[:, None]).cuda()
            proposal[ix], ties[ix] = predict(model, x)
        labels = labels.copy()
        labels['predicted_z_vox'] = proposal
        labels['tied'] = ties
        labels['ownership_rejected'] = False
        labels['outside_image'] = False
        for t, selection in labels.groupby('t'):
            part = selection.index.to_numpy()
            original_rows = labels.loc[part, 'row'].to_numpy(np.int64)
            all_rows = np.flatnonzero(graph['txyz'][:, 0] == t)
            points = graph['txyz'][original_rows, 1:].copy()
            points[:, 0] += proposal[part]
            outside = ((points < 0) | (points >= np.array(meta[1]['shape'][1:]))).any(1)
            tree = cKDTree(graph['txyz'][all_rows, 1:] * VOX)
            dist, near = tree.query(points * VOX, k=min(2, len(all_rows)))
            owned = ((all_rows[near[:, 0]] == original_rows) & (dist[:, 1] > dist[:, 0] + 1e-9)
                     if len(all_rows) > 1 else np.ones(len(part), bool))
            labels.loc[part, 'ownership_rejected'] = ~owned
            labels.loc[part, 'outside_image'] = outside
        dz = labels.dz_um.to_numpy() - proposal * VOX[0]
        before = np.linalg.norm(labels[['dz_um', 'dy_um', 'dx_um']].to_numpy(), axis=1)
        assert np.allclose(before, labels.residual_um, rtol=0, atol=1e-9)
        labels['before_um'] = before
        labels['after_um'] = np.sqrt(dz ** 2 + labels.dy_um.to_numpy() ** 2 + labels.dx_um.to_numpy() ** 2)
        labels['before_z_um'] = labels.dz_um.abs()
        labels['after_z_um'] = np.abs(dz)
        labels['signed_error_z_um'] = -dz
        labels['source_embryo'] = source
        labels['stem'] = stem
        labels['embryo'] = stem[:4]
        labels['domain'] = 'own_source_fit' if stem.startswith(source) else 'opposite_embryo'
        rows.append(labels)
    folder = out / 'evaluation'
    folder.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(folder / f'real_{source}.csv', index=False)
    print(source, 'real pairs', sum(map(len, rows)), flush=True)


def analyse(out):
    syn = pd.concat([pd.read_csv(out / 'evaluation' / f'synthetic_{e}.csv') for e in EMBRYOS], ignore_index=True)
    real_df = pd.concat([pd.read_csv(out / 'evaluation' / f'real_{e}.csv') for e in EMBRYOS], ignore_index=True)
    real_df['eligible'] = True
    unchanged = []
    for (source, stem), group in real_df.groupby(['source_embryo', 'stem']):
        original = pd.read_csv(old.DEST / 'labels' / f'{stem}.csv')
        missing = original[~original.node_id.isin(group.node_id)].copy()
        missing['source_embryo'] = source
        missing['stem'] = stem
        missing['embryo'] = stem[:4]
        missing['domain'] = 'own_source_fit' if stem.startswith(source) else 'opposite_embryo'
        missing['eligible'] = False
        missing['before_um'] = missing.residual_um
        missing['after_um'] = missing.residual_um
        missing['before_z_um'] = missing.dz_um.abs()
        missing['after_z_um'] = missing.dz_um.abs()
        missing['signed_error_z_um'] = -missing.dz_um
        missing['predicted_z_vox'] = 0
        for flag in ['tied', 'ownership_rejected', 'outside_image']:
            missing[flag] = False
        unchanged.append(missing)
    real_df = pd.concat([real_df, *unchanged], ignore_index=True)
    assert not real_df.duplicated(['source_embryo', 'stem', 'node_id']).any()
    continuity = pd.read_csv(ROOT / 'state/localization_diagnosis_20260929/continuity.csv')
    continuity = continuity.rename(columns={'p': 'node_id'})
    assert not continuity.duplicated(['stem', 'node_id']).any()
    real_df = real_df.merge(continuity[['stem', 'node_id', 'all_known_links_correct', 'any_known_link_wrong']],
                          on=['stem', 'node_id'], how='left', validate='many_to_one')
    syn_rows = []
    for (source, embryo, shift), group in syn.groupby(['source_embryo', 'embryo', 'shift_vox']):
        syn_rows.append(dict(source=source, embryo=embryo, opposite=source != embryo,
            shift_vox=int(shift), n=len(group), before_z_um=float(group.before_z_um.mean()),
            after_z_um=float(group.after_z_um.mean()), median_z_um=float(group.after_z_um.median()),
            signed_error_z_um=float(group.signed_error_z_um.mean()),
            exact_fraction=float((group.after_z_um == 0).mean()), ties=int(group.tied.sum())))
    assert len(syn_rows) == 36
    real_rows = []
    for (source, embryo), group in real_df.groupby(['source_embryo', 'embryo']):
        masks = dict(all=group.eligible, tail=group.eligible & (group.before_um > 3.5),
            axial_tail=group.eligible & (group.before_z_um > 3.5), good=group.eligible & (group.before_um <= 2.5),
            all_original=np.ones(len(group), bool), ineligible_unchanged=~group.eligible,
            persistent_tail=group.eligible & (group.before_um > 3.5) & group.all_known_links_correct.fillna(False),
            broken_tail=group.eligible & (group.before_um > 3.5) & group.any_known_link_wrong.fillna(False))
        for name, mask in masks.items():
            d = group[mask]
            real_rows.append(dict(source=source, embryo=embryo, opposite=source != embryo, group=name,
                n=len(d), before_um=float(d.before_um.mean()), after_um=float(d.after_um.mean()),
                before_z_um=float(d.before_z_um.mean()), after_z_um=float(d.after_z_um.mean()),
                signed_error_z_um=float(d.signed_error_z_um.mean()),
                changed=int((d.predicted_z_vox != 0).sum()), ties=int(d.tied.sum()),
                ownership_rejected=int(d.ownership_rejected.sum()), outside_image=int(d.outside_image.sum())))
    syn_pass = all(r['n'] > 0 and (r['after_z_um'] < r['before_z_um'] if r['shift_vox'] else r['after_z_um'] <= VOX[0])
                   for r in syn_rows if r['opposite'])
    check = [r for r in real_rows if r['opposite'] and r['group'] in ['all', 'tail', 'axial_tail', 'good']]
    assert len(check) == 8
    real_pass = all(r['n'] > 0 and
        ((r['after_um'] <= r['before_um'] + 1e-9 and r['after_z_um'] <= r['before_z_um'] + 1e-9)
         if r['group'] == 'good' else
         (r['after_um'] < r['before_um'] and r['after_z_um'] < r['before_z_um'])) for r in check)
    pd.DataFrame(syn_rows).to_csv(out / 'synthetic_summary.csv', index=False)
    pd.DataFrame(real_rows).to_csv(out / 'real_summary.csv', index=False)
    real_df.to_csv(out / 'paired_real_diagnostics.csv', index=False)
    real_df.groupby(['source_embryo', 'stem', 'domain', 'eligible']).agg(n=('node_id', 'size'),
        before_um=('before_um', 'mean'), after_um=('after_um', 'mean'),
        before_z_um=('before_z_um', 'mean'), after_z_um=('after_z_um', 'mean')).reset_index().to_csv(out / 'real_movie_summary.csv', index=False)
    save_json(out / 'decision.json', dict(status='complete_review_required',
        synthetic_gate_passed=bool(syn_pass), real_anchor_gate_passed=bool(real_pass),
        component_gate_passed=bool(syn_pass and real_pass), synthetic=syn_rows, real=real_rows,
        official_score_computed=False, graphs_modified=False, no_gain_claim=True,
        next='Review evidence before any separately registered all-node or pre-association study; failure closes this fixed recipe.'))
    print(json.dumps(dict(synthetic_gate=bool(syn_pass), real_anchor_gate=bool(real_pass))), flush=True)


def benchmark(out):
    policy()
    data = source_data(out / 'benchmark_data', '44b6')
    model = model_code.AxialLocalizer().cuda()
    opt, scheduler = model_code.make_optimizer(model)
    rng = np.random.default_rng(6101)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        x, y, _, _ = sample(data, rng)
        opt.zero_grad(set_to_none=True)
        loss = model_code.loss(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    start = time.perf_counter()
    for _ in range(30):
        predict(model, x)
    torch.cuda.synchronize()
    inference = time.perf_counter() - start
    save_json(out / 'benchmark.json', dict(status='passed', model_recipe=model_code.RECIPE,
        training30seconds=elapsed, inference1080seconds=inference,
        checkpoint_saved=False, source_only=True, batch_size=36))


def check_hashes(plan):
    for name, digest in plan['hashes'].items():
        assert sha(ROOT / name) == digest, ('Input drift', name)


def prepare(out):
    assert not (out / 'plan.json').exists(), 'Registered recipe is immutable'
    assert read(out / 'preflight.json')['status'] == 'passed'
    bench = read(out / 'benchmark.json')
    assert bench['status'] == 'passed' and bench['model_recipe'] == model_code.RECIPE
    for name, digest in read(out / 'data_input_hashes.json').items():
        assert sha(ROOT / name) == digest, ('Data-selection input drift', name)
    input_paths = set(data_code.input_paths(out))
    input_paths.update(ROOT / 'src' / name for name in [
        'c061_direct_axial_study.py', 'c061_axial_data.py', 'c061_axial_model.py',
        'c058_raw_localizer_study.py', 'c058_raw_localizer_model.py', 'c055_guarded_readmit.py',
        'frame_motion_audit.py', 'local_registration_probe.py', 'run_last_days_local.py'])
    input_paths.update([out / 'README.md', out / 'preflight.json', out / 'benchmark.json', old.DEST / 'output_hashes.json',
        ROOT / 'state/localization_diagnosis_20260929/continuity.csv'])
    input_paths.update(ROOT / name for name in read(out / 'preflight.json')['proof_hashes'])
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(input_paths)}
    selection = read(out / 'data_plan.json')
    control = read(ROOT / 'state/c061_preflight/real_controls.json')
    extraction_seconds = 0.
    planned_frames = {}
    for kind in ['direct', 'real']:
        frame_count = len(data_code.load_manifest(out, kind)[['stem', 't']].drop_duplicates())
        measured = [r for r in control['rows'] if r['kind'] == kind]
        rate = sum(r['seconds'] for r in measured) / sum(r['frames'] for r in measured)
        planned_frames[kind] = frame_count
        extraction_seconds += frame_count * rate
    fit_seconds = 2 * 1200 * bench['training30seconds'] / 30
    inference_examples = 2 * (sum(r['direct_points'] for r in selection['counts']) * 9 +
                              sum(r['real_points'] for r in selection['counts']))
    inference_seconds = inference_examples * bench['inference1080seconds'] / 1080
    estimate = max(12, int(np.ceil((1.5 * (extraction_seconds + fit_seconds + inference_seconds)) / 60 + 5)))
    save_json(out / 'plan.json', dict(created=stamp(), recipe=model_code.RECIPE,
        hashes=hashes, total_jobs=8, max_start_job_hours=3, estimated_minutes=estimate,
        estimated_components_seconds=dict(extraction=extraction_seconds, training=fit_seconds,
            gpu_inference=inference_seconds, io_variation_multiplier=1.5, overhead_allowance_seconds=300),
        planned_frames=planned_frames,
        selection=selection, no_graph_modification=True, no_kaggle_writes=True,
        stage='direct axial component falsification, not a score candidate'))
    print('Prepared', len(hashes), 'inputs; estimated minutes', estimate, flush=True)


def extract(out):
    for kind in ['direct', 'real']:
        for stem in sorted(data_code.load_manifest(out, kind).stem.unique()):
            receipt = data_code.extract_one(out, stem, kind)
            print(json.dumps({k: receipt[k] for k in ['stem', 'kind', 'points', 'frames_read', 'seconds']}), flush=True)
    audit_extracted(out)


def audit_extracted(out):
    rows = []
    for kind in ['direct', 'real']:
        manifest = data_code.load_manifest(out, kind)
        for stem, selected in manifest.groupby('stem', sort=True):
            folder = out / kind
            receipt = read(folder / f'{stem}.json')
            assert sha(folder / f'{stem}.npy') == receipt['crop_sha256']
            assert sha(folder / f'{stem}.npz') == receipt['target_sha256']
            with np.load(folder / f'{stem}.npz') as z:
                assert np.array_equal(z['gt_ids'], selected.gt_id.to_numpy(np.int64))
                assert np.array_equal(z['center_zyx'], selected[['center_z', 'center_y', 'center_x']].to_numpy(np.int64))
                assert np.array_equal(z['true_zyx'], selected[['true_z', 'true_y', 'true_x']].to_numpy(np.int64))
                if kind == 'direct':
                    assert np.array_equal(z['center_zyx'], z['true_zyx'])
                    assert np.array_equal(z['target_z_vox'], -np.arange(-4, 5))
                else:
                    assert np.array_equal(z['node_ids'], selected.node_id.to_numpy(np.int64))
                    assert np.array_equal(z['targets_um'], selected[['dz_um', 'dy_um', 'dx_um']].to_numpy(np.float32))
            rows.append(dict(kind=kind, stem=stem, points=len(selected), status='passed'))
    save_json(out / 'extraction_audit.json', dict(status='passed', rows=rows))


def run(out):
    assert not (out / 'status.json').exists(), 'No blind resume'
    plan = read(out / 'plan.json')
    q = Queue(out, plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'], plan_sha256=sha(out / 'plan.json'))
        q.save()
        check_hashes(plan)
        q.run('extract', [sys.executable, '-u', Path(__file__), 'extract', '--out', out])
        for source in EMBRYOS:
            q.run('train_' + source, [sys.executable, '-u', Path(__file__), 'train', '--out', out, '--group', source])
        for source in EMBRYOS:
            for verb in ['synthetic', 'real']:
                q.run(verb + '_' + source, [sys.executable, '-u', Path(__file__), verb, '--out', out, '--group', source])
        q.run('analyse', [sys.executable, '-u', Path(__file__), 'analyse', '--out', out])
        check_hashes(plan)
        output_hashes = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()
            and p.name not in ['queue.lock', 'status.json', 'launcher.stdout.log', 'launcher.stderr.log', 'output_hashes.json']}
        save_json(out / 'output_hashes.json', output_hashes)
        q.close('complete_review_required')
    except Exception as exc:
        q.close('failed', exc)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('verb', choices=['extract', 'benchmark', 'prepare', 'run', 'train', 'synthetic', 'real', 'analyse'])
    p.add_argument('--out', type=Path, default=DEST)
    p.add_argument('--group', choices=EMBRYOS)
    a = p.parse_args()
    if a.verb in ['train', 'synthetic', 'real']:
        globals()[a.verb](a.out, a.group)
    else:
        globals()[a.verb](a.out)


if __name__ == '__main__':
    main()
