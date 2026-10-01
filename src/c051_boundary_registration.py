#!/usr/bin/env python3
"""One support-aware extension of C036; reuse extraction, diagnostics and Queue."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits
import local_registration_probe as old
from reid_probe_local import ROOT, sha, save_json, stamp
from c047_hard_example_study import read, verify_hashes
from run_last_days_local import Queue

DEST = ROOT/'experiments/candidates/c051_boundary_registration'
REFERENCE = old.DEST
REGISTER = old.register_once
MOTION = old.local_motion
SUPPORT = dict(min_retained_fraction=.5, point_margin_voxels=1, mode='minimum_joint_feasible_recenter')
support_calls = []
motion_calls = []


def supported_register(first, second, point, prior, shape):
    baseline = REGISTER(first, second, point, prior, shape)
    if baseline['reason'] != 'image_boundary':
        return baseline
    point, prior = np.asarray(point), np.asarray(prior)
    size, radius = np.asarray(shape), np.asarray(old.RECIPE['search_radius'])
    half = size//2
    # Existing phase_shift produces integer priors, including the reverse call.
    if not np.array_equal(prior, np.rint(prior)):
        return dict(valid=False, reason='support_noninteger_prior')
    shift = prior.astype(int)
    lower = np.maximum(half, half+radius-shift)
    upper = np.minimum(np.asarray(first.shape)-size+half,
                       np.asarray(second.shape)-size-radius+half-shift)
    original_center = np.rint(point).astype(int)
    if np.any(lower > upper):
        support_calls.append(dict(valid_support=False, reason='joint_window_infeasible'))
        return dict(valid=False, reason='support_infeasible')
    center = np.clip(original_center, lower, upper)
    recenter = center-original_center
    retained = float(np.prod(np.maximum(0, size-np.abs(recenter))/size))
    start, end = center-half, center-half+size-1
    valid = bool(retained >= SUPPORT['min_retained_fraction'] and
                 np.all(point >= start+SUPPORT['point_margin_voxels']) and
                 np.all(point <= end-SUPPORT['point_margin_voxels']))
    support_calls.append(dict(valid_support=valid, retained_fraction=retained,
                              recenter_voxels=recenter.tolist(), full_template_voxels=int(np.prod(size))))
    if not valid:
        return dict(valid=False, reason='support_point_or_overlap')
    result = REGISTER(first, second, center.astype(float), prior, shape)
    assert result['reason'] != 'image_boundary', 'joint feasible support calculation failed'
    if result['valid']:
        result = dict(result, point=point+result['displacement'])
    return result


def supported_motion(first, second, point, prior):
    baseline = MOTION(first, second, point, prior)
    boundary = 'image_boundary' in baseline['reason']
    begin = len(support_calls)
    if boundary:
        old.register_once = supported_register
        try:
            result = MOTION(first, second, point, prior)
        finally:
            old.register_once = REGISTER
    else:
        result = baseline
    motion_calls.append(dict(original_reason=baseline['reason'], original_trusted=baseline['valid'],
                             original_point=np.asarray(baseline.get('point', point+prior)).tolist(),
                             boundary=boundary, support_calls=support_calls[begin:]))
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'plan.json').exists()
    inherited = old.input_manifest()
    reference = read(REFERENCE/'status.json'); assert reference['status']=='complete'
    hashes = dict(inherited['files'])
    hashes.update(reference['source_hashes'])
    for job in reference['jobs'].values():
        hashes.update(job['artifacts'])
    for _, stem in old.evaluation_plan():
        hashes.update(read(REFERENCE/'groups'/(stem+'.json'))['image_chunk_hashes'])
    for p in [Path(__file__), out/'README.md', ROOT/'src/run_last_days_local.py',
              ROOT/'src/c047_hard_example_study.py', REFERENCE/'analysis.json', REFERENCE/'status.json',
              ROOT/'state/improvement_feasibility_20260928/boundary_coverage.json']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    verify_hashes(hashes)
    save_json(out/'plan.json', dict(created=stamp(), recipe=old.RECIPE, support=SUPPORT,
                                   inputs=hashes, movies=old.evaluation_plan(), total_jobs=25,
                                   max_start_job_hours=2, estimated_minutes=10))
    print('Prepared 25 finite jobs;',len(hashes),'pinned inputs',flush=True)


def smoke(out):
    rng = np.random.default_rng(5101)
    first = gaussian_filter(rng.random((48,112,112),dtype=np.float32),sigma=(.6,1.1,1.1))
    cases=[]
    for point, delta in [(np.array([24.,55.,55.]), np.array([1.,3.,-2.])),
                         (np.array([5.,55.,55.]),np.array([1.,3.,-2.])),
                         (np.array([24.,16.,55.]),np.array([1.,3.,-2.])),
                         (np.array([24.,55.,16.]),np.array([1.,3.,2.])),
                         (np.array([42.,55.,55.]),np.array([-1.,3.,-2.])),
                         (np.array([24.,98.,55.]),np.array([1.,-3.,-2.])),
                         (np.array([24.,55.,96.]),np.array([1.,3.,-2.]))]:
        second=np.roll(first,delta.astype(int),axis=(0,1,2))*1.3+.2
        prior=delta+np.array([0.,1.,-1.])
        before=MOTION(first,second,point,prior)
        after=supported_motion(first,second,point,prior)
        assert after['valid'], (point,after)
        error=float(np.linalg.norm((after['point']-point-delta)*old.VOX))
        assert error < .1, (point,error)
        if before['valid']:
            assert before.keys()==after.keys()
            for k in before: np.testing.assert_equal(before[k],after[k])
        else: assert 'image_boundary' in before['reason']
        cases.append(dict(point=point.tolist(), original_reason=before['reason'],error_um=error))
    assert not supported_motion(np.ones_like(first),np.ones_like(first),[5.,55.,55.],np.zeros(3))['valid']
    assert not supported_motion(first,first,[24.,1.,55.],np.zeros(3))['valid']
    too_small=first[:8]
    assert not supported_register(too_small,too_small,[4.,55.,55.],np.zeros(3),old.RECIPE['template_large'])['valid']
    save_json(out/'smoke.json',dict(status='passed',cases=cases,flat_rejected=True,insufficient_support_rejected=True))
    print('Real-support numerical controls passed',flush=True)


def extract(out, stem):
    motion_calls.clear();support_calls.clear()
    old.local_motion=supported_motion
    def tick(current): print(stamp(),current,flush=True)
    old.extract_movie(stem,out,tick)
    current=pd.read_csv(out/'groups'/(stem+'.csv'))
    reference=pd.read_csv(REFERENCE/'groups'/(stem+'.csv'))
    assert len(current)==len(reference)==len(motion_calls)
    assert current[['stem','source','t']].equals(reference[['stem','source','t']])
    for row, baseline in zip(motion_calls,reference.to_dict('records')):
        assert row['original_reason']==baseline['reason']
        assert row['original_trusted']==baseline['trusted']
        np.testing.assert_allclose(row['original_point'],[baseline['predicted_z'],baseline['predicted_y'],baseline['predicted_x']],rtol=0,atol=1e-9)
    unchanged=~np.array([r['boundary'] for r in motion_calls])
    for column in reference.columns:
        a,b=current.loc[unchanged,column],reference.loc[unchanged,column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),rtol=0,atol=1e-9,equal_nan=True)
        else: assert a.fillna('').tolist()==b.fillna('').tolist(),column
    with np.load(out/'pairs'/(stem+'.npz')) as a, np.load(REFERENCE/'pairs'/(stem+'.npz')) as b:
        for key in ['source','target','coarse_um']:np.testing.assert_array_equal(a[key],b[key])
        unchanged_sources=current.loc[unchanged,'source'].to_numpy()
        mask=np.isin(a['source'],unchanged_sources)
        np.testing.assert_array_equal(a['local_um'][mask],b['local_um'][mask])
    for row,base in zip(motion_calls,reference.to_dict('records')):
        row.update(stem=stem,source=base['source'])
    save_json(out/'support'/(stem+'.json'),dict(groups=motion_calls,original_control_passed=True,
        nonboundary_unchanged=int(unchanged.sum()),boundary_groups=int((~unchanged).sum())))
    print('Original controls and nonboundary parity passed',stem,flush=True)


def analyse(out):
    result=old.analyse(out)
    result['historical_c036_compute_gate']=result.pop('advance_to_replay_review')
    result['advance_to_replay_review']=None
    result['review_required']=True
    result['support_recipe']=SUPPORT
    result['limitations'].append('Only original boundary failures receive a fixed recentered real-image context; review small effects manually, no old >=5 automatic rejection or direct submission.')
    frames=[]
    for _,stem in old.evaluation_plan():
        a=pd.read_csv(out/'groups'/(stem+'.csv'))
        b=pd.read_csv(REFERENCE/'groups'/(stem+'.csv'))
        a['original_boundary']=b['reason'].str.contains('image_boundary',regex=False).to_numpy()
        frames.append(a)
    data=pd.concat(frames,ignore_index=True)
    result['boundary_rows']=[]
    for embryo,g in data[data['original_boundary']].groupby('embryo'):
        result['boundary_rows'].append(dict(embryo=embryo,groups=len(g),trusted=int(g.trusted.sum()),
            proposals=int(g.proposal.sum()),fixes=int(g.fixes.sum()),harms=int(g.harms.sum()),
            unknown_proposals=int(g.unknown_proposal.sum()),positive_movies=int((g.groupby('stem').fixes.sum()-g.groupby('stem').harms.sum()>0).sum())))
    save_json(out/'analysis.json',result)
    with (out/'RESULTS.md').open('a',encoding='utf-8') as f:
        f.write('\nC051: the gate above is historical C036 output only. Manual review of both groups and support evidence is required; no automatic promotion.\n')
    print(json.dumps(result['boundary_rows']),flush=True)


def verify(out):
    verify_hashes(read(out/'plan.json')['inputs'])
    assert read(out/'smoke.json')['status']=='passed'
    for _,stem in old.evaluation_plan():
        assert read(out/'support'/(stem+'.json'))['original_control_passed']
        verify_hashes(read(out/'groups'/(stem+'.json'))['image_chunk_hashes'])
    files=[out/'smoke.json',out/'analysis.json',out/'movie_diagnostics.csv',out/'RESULTS.md']
    files += [p for folder in ['groups','pairs','support'] for p in (out/folder).iterdir() if p.is_file()]
    save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
    print('Verified inputs,22 actual original controls and',len(files),'artifacts',flush=True)


def run(out):
    assert not (out/'status.json').exists(),'No blind restart'
    plan=read(out/'plan.json');verify_hashes(plan['inputs'])
    for folder in ['groups','pairs','support','logs']:(out/folder).mkdir(exist_ok=True)
    q=Queue(out,2)
    try:
        q.state.update(total_jobs=25,plan_sha256=sha(out/'plan.json'));q.save()
        def task(name,command,*args):q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*args])
        task('smoke','smoke')
        for _,stem in plan['movies']:task('extract_'+stem,'extract','--stem',stem)
        task('analyse','analyse');task('verify','verify');q.close('complete_review_required')
    except BaseException as e:q.close('failed',e);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','smoke','extract','analyse','verify'])
    ap.add_argument('--out',type=Path,default=DEST);ap.add_argument('--stem')
    args=ap.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    with threadpool_limits(limits=4):
        if args.command=='extract':extract(out,args.stem)
        else:globals()[args.command](out)
