#!/usr/bin/env python3
"""C036: candidate-independent local 3D image registration on C023 candidates.

No training, graph mutation, new metric implementation, or Kaggle operation.
Reuse the existing frame-motion audit's reader/global phase correlation and the
C035 original-coordinate caches whose C023/C034 controls have already passed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter, shift
import skimage
from skimage.feature import match_template
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from frame_motion_audit import VOX, phase_shift, read_frame
from reid_augmented_local import evaluation_plan
from reid_probe_local import sha, save_json, stamp

DEST = ROOT/'experiments/candidates/c036_local_registration'
C035 = ROOT/'experiments/candidates/c035_augmented_reid'
RECIPE = dict(template_small=[9, 33, 33], template_large=[13, 49, 49],
              search_radius=[3, 12, 12], global_xy_stride=2,
              min_ncc=.65, min_peak_gap=.03, peak_exclusion_um=2.,
              max_multiscale_um=1.5, max_cycle_um=1.,
              max_candidate_um=2.5, min_candidate_margin_um=.75,
              min_improvement_um=1., max_c023_cost_increase=1.,
              min_net_each_embryo=5, min_positive_movies=2,
              fixes_per_harm_min=2, version=1)


def bounded_crop(frame, center, shape):
    center = np.rint(center).astype(int)
    size = np.array(shape, int)
    lower = center - size//2
    upper = lower + size
    if np.any(lower < 0) or np.any(upper > frame.shape):
        return None, center
    return frame[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))], center


def register_once(first, second, point, prior, shape):
    """Valid-mode zero-mean NCC; return transport of point, preserving its offset.

    The search region depends only on the source point and global image shift,
    never on candidate coordinates, labels or C023's chosen successor.
    """
    radius = np.array(RECIPE['search_radius'])
    template, ca = bounded_crop(first, point, shape)
    search, cb = bounded_crop(second, np.asarray(point)+prior, np.array(shape)+2*radius)
    if template is None or search is None:
        return dict(valid=False, reason='image_boundary')
    if float(np.std(template)) < 1e-5 or float(np.std(search)) < 1e-5:
        return dict(valid=False, reason='flat_patch')
    corr = match_template(search.astype(np.float32), template.astype(np.float32), pad_input=False)
    if not np.isfinite(corr).all():
        return dict(valid=False, reason='nonfinite_correlation')
    assert tuple(corr.shape) == tuple(2*radius+1)
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    if np.any(peak == 0) or np.any(peak == np.array(corr.shape)-1):
        return dict(valid=False, reason='search_boundary', ncc=float(corr[tuple(peak)]))
    correction = np.zeros(3)
    for axis in range(3):
        minus, plus = peak.copy(), peak.copy()
        minus[axis] -= 1; plus[axis] += 1
        left, mid, right = float(corr[tuple(minus)]), float(corr[tuple(peak)]), float(corr[tuple(plus)])
        denominator = left - 2*mid + right
        if denominator < -1e-8:
            correction[axis] = np.clip(.5*(left-right)/denominator, -.5, .5)
    coords = np.indices(corr.shape).transpose(1, 2, 3, 0)
    separate = np.linalg.norm((coords-peak)*VOX, axis=-1) >= RECIPE['peak_exclusion_um']
    second_peak = float(corr[separate].max())
    ncc = float(corr[tuple(peak)])
    displacement = cb - ca + peak - radius + correction
    return dict(valid=True, reason='ok', point=np.asarray(point)+displacement,
                displacement=displacement, ncc=ncc, peak_gap=ncc-second_peak)


def local_motion(first, second, point, prior):
    small = register_once(first, second, point, prior, RECIPE['template_small'])
    large = register_once(first, second, point, prior, RECIPE['template_large'])
    if not small['valid'] or not large['valid']:
        return dict(valid=False, reason='forward:' + small['reason'] + '/' + large['reason'])
    disagreement = float(np.linalg.norm((small['point']-large['point'])*VOX))
    estimated = (small['point']+large['point'])/2
    backward = register_once(second, first, estimated, -np.asarray(prior), RECIPE['template_small'])
    result = dict(valid=False, reason='backward:' + backward['reason'], point=estimated,
                  ncc=min(small['ncc'], large['ncc']), peak_gap=min(small['peak_gap'], large['peak_gap']),
                  multiscale_um=disagreement)
    if not backward['valid']:
        return result
    cycle = float(np.linalg.norm((backward['point']-point)*VOX))
    result.update(cycle_um=cycle, backward_ncc=backward['ncc'],
                  peak_gap=min(result['peak_gap'], backward['peak_gap']))
    if min(result['ncc'], backward['ncc']) < RECIPE['min_ncc']:
        result['reason'] = 'low_ncc'
    elif result['peak_gap'] < RECIPE['min_peak_gap']:
        result['reason'] = 'ambiguous_peak'
    elif disagreement > RECIPE['max_multiscale_um']:
        result['reason'] = 'scale_disagreement'
    elif cycle > RECIPE['max_cycle_um']:
        result['reason'] = 'cycle_disagreement'
    else:
        result.update(valid=True, reason='ok')
    return result


def smoke():
    rng = np.random.default_rng(3601)
    first = gaussian_filter(rng.random((64, 128, 128), dtype=np.float32), sigma=(.6, 1.1, 1.1))
    first = (first-first.min())/(first.max()-first.min())
    point = np.array([32.25, 63.6, 65.2])
    cases = []
    for delta in [np.array([0., 0., 0.]), np.array([2., 7., -9.]), np.array([-2., -8., 6.])]:
        second = np.roll(first, delta.astype(int), axis=(0, 1, 2))*1.3+.2
        global_shift = phase_shift(first-first.mean(), second-second.mean())
        assert np.array_equal(global_shift, delta), ('global_sign', global_shift, delta)
        # Deliberately perturb the global prior so the local search must correct it.
        result = local_motion(first, second, point, delta+np.array([0., 3., -2.]))
        assert result['valid'], result
        error = float(np.linalg.norm((result['point']-point-delta)*VOX))
        assert error < .1, ('local_translation', error)
        cases.append(dict(delta=delta.tolist(), error_um=error, cycle_um=result['cycle_um'],
                          min_ncc=result['ncc'], min_peak_gap=result['peak_gap']))
    delta = np.array([.35, 2.3, -3.2])
    second = shift(first, delta, order=1, mode='nearest', prefilter=False)
    result = local_motion(first, second, point, np.zeros(3))
    assert result['valid'], result
    error = float(np.linalg.norm((result['point']-point-delta)*VOX))
    assert error < .6, ('subvoxel_translation', error)
    assert not register_once(np.ones_like(first), np.ones_like(first), point, np.zeros(3), RECIPE['template_small'])['valid']
    assert not register_once(first, first, [1, 1, 1], np.zeros(3), RECIPE['template_small'])['valid']
    return dict(status='complete', integer_cases=cases, subvoxel_error_um=error,
                flat_patch_rejected=True, image_boundary_rejected=True,
                note='Known-translation numerical check only; not an improvement result')


def input_manifest():
    state = json.loads((C035/'status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'complete'
    assert all(sha(ROOT/p) == digest for p, digest in state['source_hashes'].items()), 'C035 source drift'
    files = {}
    for _, stem in evaluation_plan():
        job = state['jobs']['eval_crops:' + stem]
        assert job['status'] == 'complete'
        for path, digest in job['artifacts'].items():
            assert sha(ROOT/path) == digest, ('C035 cached artifact drift', path)
            files[path] = digest
        info = json.loads((C035/'eval'/f'{stem}.json').read_text(encoding='utf-8'))
        assert info['c023_control_verified'] and info['exact_c034_pairs'] and info['original_coordinates']
        for path, digest in info['image_metadata'].items():
            assert sha(ROOT/path) == digest, ('image metadata drift', path)
            files[path] = digest
    return dict(recipe=RECIPE, files=files, baseline='Verified unchanged C023/C034 caches from C035',
                no_training=True, candidate_independent_registration=True)


def extract_movie(stem, out, tick):
    with np.load(C035/'eval'/f'{stem}.npz') as data:
        d = {k: data[k].copy() for k in ['source', 'target', 'label', 'geometry', 'selected', 'source_ix', 'target_ix', 'tzyx']}
    zp = ROOT/'data/train'/(stem+'.zarr')
    meta = json.loads((zp/'0/zarr.json').read_text())
    quantiles = json.loads((zp/'zarr.json').read_text())['attributes']['image_statistics']['quantiles']
    lo, hi = float(quantiles['0.001']), float(quantiles['0.999'])
    assert hi > lo
    shape, dtype = tuple(meta['shape']), np.dtype(meta['data_type'])
    group_by_time = collections.defaultdict(list)
    for sid in np.unique(d['source']):
        indices = np.flatnonzero(d['source'] == sid)
        source_row = d['tzyx'][d['source_ix'][indices[0]]]
        assert np.all(d['tzyx'][d['target_ix'][indices], 0] == source_row[0]+1)
        group_by_time[int(source_row[0])].append((int(sid), indices, source_row[1:].astype(float)))
    cache, chunk_hashes, group_rows = {}, {}, []
    pair_motion = np.full(len(d['source']), np.nan, np.float32)
    pair_coarse = np.full(len(d['source']), np.nan, np.float32)
    for t in sorted(group_by_time):
        tick(f'local_registration:{stem}:frame{t}')
        for tt in [t, t+1]:
            if tt not in cache:
                raw = read_frame(zp, tt, shape, dtype)
                cache[tt] = np.clip((raw.astype(np.float32)-lo)/(hi-lo+1e-6), 0., 3.)
                path = zp/'0/c'/str(tt)/'0/0/0'
                chunk_hashes[str(path.relative_to(ROOT))] = sha(path)
        a, b = cache[t], cache[t+1]
        stride = RECIPE['global_xy_stride']
        # Same estimator as the existing global motion audit, no GT displacement.
        aa, bb = a[:, ::stride, ::stride], b[:, ::stride, ::stride]
        prior = phase_shift(aa-aa.mean(), bb-bb.mean())*np.array([1., stride, stride])
        for sid, indices, point in group_by_time[t]:
            result = local_motion(a, b, point, prior)
            candidates = d['tzyx'][d['target_ix'][indices], 1:]
            coarse_distance = np.linalg.norm((candidates-point-prior)*VOX, axis=1)
            pair_coarse[indices] = coarse_distance
            target = result.get('point', point+prior)
            distances = np.linalg.norm((candidates-target)*VOX, axis=1)
            pair_motion[indices] = distances
            order = np.argsort(distances, kind='stable'); first, second = order[:2]
            labels, cost = d['label'][indices], d['geometry'][indices, 3]
            old = np.flatnonzero(d['selected'][indices])
            assert len(old) <= 1 and len(indices) >= 2 and np.sum(labels == 1) == 1
            old_correct = bool(len(old) and labels[old[0]] == 1)
            new_correct = bool(labels[first] == 1)
            cost_correct = bool(labels[np.argmin(cost)] == 1)
            trusted = bool(result['valid'])
            change = bool(trusted and len(old) == 1 and first != old[0] and
                          distances[first] <= RECIPE['max_candidate_um'] and
                          distances[second]-distances[first] >= RECIPE['min_candidate_margin_um'] and
                          distances[old[0]]-distances[first] >= RECIPE['min_improvement_um'] and
                          cost[first] <= cost[old[0]]+RECIPE['max_c023_cost_increase'])
            row = dict(stem=stem, embryo=stem[:4], source=sid, t=t, trusted=trusted, reason=result['reason'],
                       c023_correct=old_correct, cost_correct=cost_correct,
                       coarse_correct=bool(labels[np.argmin(coarse_distance)] == 1),
                       local_correct=new_correct, proposal=change,
                       fixes=int(change and new_correct and not old_correct),
                       harms=int(change and old_correct and not new_correct),
                       unknown_proposal=int(change and labels[first] < 0),
                       candidate=int(d['target'][indices[first]]), c023_target=int(d['target'][indices[old[0]]]) if len(old) else -1,
                       best_um=float(distances[first]), margin_um=float(distances[second]-distances[first]),
                       coarse_dz=float(prior[0]), coarse_dy=float(prior[1]), coarse_dx=float(prior[2]),
                       source_z=float(point[0]), source_y=float(point[1]), source_x=float(point[2]),
                       predicted_z=float(target[0]), predicted_y=float(target[1]), predicted_x=float(target[2]))
            row.update({k: result.get(k) for k in ['ncc', 'backward_ncc', 'peak_gap', 'cycle_um', 'multiscale_um']})
            group_rows.append(row)
        cache = {t+1: cache[t+1]}
    pd.DataFrame(group_rows).to_csv(out/'groups'/f'{stem}.csv', index=False)
    np.savez_compressed(out/'pairs'/f'{stem}.npz', source=d['source'], target=d['target'],
                        local_um=pair_motion, coarse_um=pair_coarse)
    assert np.isfinite(pair_motion).all() and np.isfinite(pair_coarse).all()
    info = dict(stem=stem, groups=len(group_rows), trusted=sum(r['trusted'] for r in group_rows),
                proposals=sum(r['proposal'] for r in group_rows),
                image_chunk_hashes=chunk_hashes,
                failure_reasons=dict(collections.Counter(r['reason'] for r in group_rows)))
    save_json(out/'groups'/f'{stem}.json', info)
    return {k: v for k, v in info.items() if k != 'image_chunk_hashes'}


def analyse(out):
    data = pd.concat([pd.read_csv(out/'groups'/f'{s}.csv') for _, s in evaluation_plan()], ignore_index=True)
    rows = []
    for stem, group in data.groupby('stem'):
        trusted = group[group['trusted']]
        row = dict(stem=stem, embryo=stem[:4], groups=len(group), trusted=len(trusted),
                   c023_correct=int(group['c023_correct'].sum()),
                   coarse_correct=int(group['coarse_correct'].sum()),
                   local_correct_all=int(group['local_correct'].sum()),
                   trusted_local_correct=int(trusted['local_correct'].sum()),
                   trusted_cost_correct=int(trusted['cost_correct'].sum()),
                   trusted_rank_net=int(trusted['local_correct'].sum()-trusted['cost_correct'].sum()),
                   proposals=int(group['proposal'].sum()), fixes=int(group['fixes'].sum()), harms=int(group['harms'].sum()),
                   unknown_proposals=int(group['unknown_proposal'].sum()))
        row['net'] = row['fixes'] - row['harms']; rows.append(row)
    movies = pd.DataFrame(rows); movies.to_csv(out/'movie_diagnostics.csv', index=False)
    folds = []
    for embryo, group in movies.groupby('embryo'):
        values = {k: int(group[k].sum()) for k in movies.columns if k not in ['stem', 'embryo']}
        positive_movies = int((group['net'] > 0).sum())
        gate = (values['net'] >= RECIPE['min_net_each_embryo'] and positive_movies >= RECIPE['min_positive_movies']
                and values['fixes'] >= RECIPE['fixes_per_harm_min']*values['harms'] and values['trusted_rank_net'] >= 0)
        folds.append(dict(test_embryo=embryo, **values, positive_net_movies=positive_movies,
                          negative_net_movies=int((group['net'] < 0).sum()), compute_gate=gate))
    result = dict(kind='conditional_registration_diagnostic_not_official_graph_score', completed=stamp(),
                  recipe=RECIPE, folds=folds, reasons={str(k): int(v) for k, v in data['reason'].value_counts().items()},
                  advance_to_replay_review=len(folds)==2 and all(f['compute_gate'] for f in folds),
                  limitations=[
                      'No graph changes or official modified-graph score. Per-source proposals ignore global assignment conflicts and later C023 stages.',
                      'Ground truth defines this annotated reachable-source diagnostic only; it never enters the image estimator or proposal rule.',
                      'Unknown targets remain unknown; retrieval misses are not confirmed biological negatives.',
                      'The public detectors saw both embryos. These movie IDs are local validation, not an independent hidden-test distribution.',
                      'Global phase correlation can be wrong; local NCC models translation only and rejects boundaries, ambiguous peaks and inconsistent motion.',
                      'Failure closes this fixed two-scale algorithm, not every possible optical-flow or registration model.'])
    save_json(out/'analysis.json', result)
    lines = ['# C036 local image registration diagnostic', '', 'No graph edits or official-score delta were produced.', '',
             '| Embryo | Groups | Trusted | Proposals | Fixes / harms | Net | Positive movies | Gate |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for f in folds:
        lines.append(f"| {f['test_embryo']} | {f['groups']} | {f['trusted']} | {f['proposals']} | {f['fixes']} / {f['harms']} | {f['net']} | {f['positive_net_movies']} | {f['compute_gate']} |")
    lines += ['', f"Advance to official-replay integration review: {result['advance_to_replay_review']}", '',
              *['- '+x for x in result['limitations']]]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['run', 'analyse'])
    ap.add_argument('--out', type=Path, default=DEST)
    ap.add_argument('--max-hours', type=float, default=4.)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ['groups', 'pairs', 'logs']:
        (out/name).mkdir(exist_ok=True)
    if args.command == 'analyse':
        analyse(out); return
    fd = os.open(out/'study.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    path = out/'status.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(jobs={})
    source_paths = [Path(__file__), ROOT/'src/frame_motion_audit.py', ROOT/'src/reid_augmented_local.py',
                    ROOT/'src/reid_probe_local.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    start = time.time()
    def tick(current):
        if time.time()-start > args.max_hours*3600:
            raise TimeoutError('C036 finite execution budget exhausted; review before restart')
        state.update(current=current, updated=stamp()); save_json(path, state)
    try:
        if state.get('source_hashes') and state['source_hashes'] != hashes:
            raise ValueError('C036 source drift; review before reusing artifacts')
        state.update(status='running', pid=os.getpid(), started=stamp(), source_hashes=hashes,
                     runtime=dict(device='CPU', threads=4, numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__))
        state.pop('error', None); save_json(path, state)
        with threadpool_limits(limits=4):
            tick('verify_inherited_inputs')
            manifest = input_manifest()
            mp = out/'inputs.json'
            if mp.exists():
                assert json.loads(mp.read_text(encoding='utf-8')) == manifest, 'input/recipe drift'
            else:
                save_json(mp, manifest)
            tick('synthetic_translation_smoke')
            result = smoke(); save_json(out/'smoke.json', result)
            state['smoke'] = result['status']; save_json(path, state)
            for split, stem in evaluation_plan():
                old = state['jobs'].get(stem, {})
                if old.get('status') == 'complete':
                    assert all(sha(ROOT/p) == h for p, h in old['artifacts'].items()), ('artifact drift', stem)
                    info = json.loads((out/'groups'/f'{stem}.json').read_text(encoding='utf-8'))
                    assert all(sha(ROOT/p) == h for p, h in info['image_chunk_hashes'].items()), ('image drift', stem)
                    continue
                tick(stem); t0=time.time()
                info = extract_movie(stem, out, tick)
                artifacts = [out/'groups'/f'{stem}.csv', out/'groups'/f'{stem}.json', out/'pairs'/f'{stem}.npz']
                state['jobs'][stem] = dict(status='complete', split=split, seconds=time.time()-t0, **info,
                    artifacts={str(p.relative_to(ROOT)): sha(p) for p in artifacts})
                save_json(path, state); print(stamp(), stem, json.dumps(info), flush=True)
            tick('analyse'); result=analyse(out)
            assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}, 'source changed during execution'
            state.update(status='complete', current=None, ended=stamp(), advance_to_replay_review=result['advance_to_replay_review'])
            save_json(path, state)
    except Exception as exc:
        state.update(status='failed', error=str(exc), traceback=traceback.format_exc(), ended=stamp())
        save_json(path, state); raise
    finally:
        os.close(fd); (out/'study.lock').unlink()


if __name__ == '__main__':
    main()
