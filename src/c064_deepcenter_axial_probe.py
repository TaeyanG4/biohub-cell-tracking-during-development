"""Frozen C023 DeepCenter axial information audit; no fit or graph mutation."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from c055_guarded_readmit import BASE, read, save_json, sha
import c058_localizer_baseline as baseline
import c058_raw_localizer_study as old
import eval_pp_variants_local as harness

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/candidates/c064_deepcenter_axial'
OUT = DEST
C058 = ROOT / 'experiments/candidates/c058_raw_localizer'
C060 = ROOT / 'experiments/candidates/c060_spatial_track_localizer'
SAMPLE = ROOT / 'state/c060_review_20260929/model/sample.csv'
PROTOCOL = ROOT / 'state/c064_next_review/FIXED_PROTOCOL.md'
PACK = ROOT / 'artifacts/pilkwang_deepcenter'
CHECKPOINT = PACK / 'weights/full_frame_center/best.pt'
VOX = np.array([1.625, .40625, .40625])
RECIPE = dict(pool_factor=4, xy_radius=2, z_radius=4, checkpoint_epoch=2,
              score_profile='maximum over unchanged production5x5XY per plane',
              proposal='unique9-plane maximum; tie/no-support/nonfinite->zero; originalxy',
              ownership='diagnostic only; no runtime filter', benchmark_frames=2,
              samples=256, no_fit=True, no_graph=True)


def sample():
    d = pd.read_csv(SAMPLE)
    assert len(d) == 256 and d.groupby('embryo').size().to_dict() == {'44b6':128, '6bba':128}
    assert not d.duplicated(['stem','node_id']).any()
    return d


def frames():
    return sorted((str(s), int(t)) for s,t in sample()[['stem','t']].drop_duplicates().itertuples(index=False,name=None))


def dependencies():
    paths = {Path(__file__), BASE, SAMPLE, PROTOCOL, ROOT/'state/c064_next_review/checkpoint_metadata.json',
             C058/'output_hashes.json', C058/'plan.json', C060/'output_hashes.json',
             ROOT/'state/c060_review_20260929/model/input_hashes.json',
             ROOT/'state/c060_review_20260929/model/selection.json',
             ROOT/'state/remaining_methods_20260929/deepcenter_contract.json'}
    for name in ['c055_guarded_readmit.py','c058_localizer_baseline.py','c058_raw_localizer_study.py',
                 'c058_raw_localizer_model.py','eval_pp_variants_local.py','evaluate_local.py',
                 'run_last_days_local.py','frame_motion_audit.py','local_registration_probe.py']:
        paths.add(ROOT/'src'/name)
    paths.update(p for p in PACK.rglob('*') if p.is_file())
    for stem in sample().stem.unique():
        paths.update([C058/'baseline'/f'{stem}.npz', C058/'baseline'/f'{stem}.json',
                      C058/'labels'/f'{stem}.csv', C060/'train'/f'{stem}.csv',
                      C060/'train'/f'{stem}.npz', C060/'train'/f'{stem}.json'])
        paths.update(p for p in (ROOT/'data/train'/f'{stem}.geff').rglob('*') if p.is_file())
        paths.update([ROOT/'data/train'/f'{stem}.zarr/zarr.json',ROOT/'data/train'/f'{stem}.zarr/0/zarr.json'])
    for stem,t in frames(): paths.add(ROOT/'data/train'/f'{stem}.zarr/0/c/{t}/0/0/0')
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    return sorted(paths)


def verify_recorded(path, manifest, base):
    key = str(path.relative_to(base))
    expected = manifest.get(key, manifest.get(key.replace('\\','/')))
    assert expected and sha(path) == expected, ('recorded artifact drift', str(path))


def identity_check():
    d = sample(); old_hash = read(C058/'output_hashes.json'); c60hash = read(C060/'output_hashes.json')
    recorded = read(ROOT/'state/c060_review_20260929/model/input_hashes.json')
    verify_recorded(SAMPLE, recorded, ROOT)
    import zarr
    for stem,g in d.groupby('stem'):
        for rel in [f'baseline/{stem}.npz',f'baseline/{stem}.json',f'labels/{stem}.csv']:
            verify_recorded(C058/rel,old_hash,C058)
        for rel in [f'train/{stem}.csv',f'train/{stem}.npz',f'train/{stem}.json']:
            verify_recorded(C060/rel,c60hash,C060)
        graph = old.load_baseline(C058,stem)
        labels = pd.read_csv(C058/'labels'/f'{stem}.csv').set_index('node_id')
        source = pd.read_csv(C060/'train'/f'{stem}.csv')
        gt = zarr.open(str(ROOT/'data/train'/f'{stem}.geff'),mode='r')
        gtids = np.asarray(gt['nodes/ids']); gtpoints = np.column_stack([np.asarray(gt[f'nodes/props/{k}/values']) for k in ['t','z','y','x']])
        gtlookup = {int(n):i for i,n in enumerate(gtids)}
        for r in g.itertuples(index=False):
            assert graph['ids'][r.row] == r.node_id and graph['txyz'][r.row,0] == r.t
            assert labels.loc[r.node_id].gt_id == r.gt_id
            assert source.iloc[r.sample_index].node_id == r.node_id
            actual = gtpoints[gtlookup[int(r.gt_id)]]
            assert actual[0] == r.t
            residual = (actual[1:]-graph['txyz'][r.row,1:])*VOX
            assert np.array_equal(residual,np.array([r.dz_um,r.dy_um,r.dx_um]))
            assert abs(np.linalg.norm(residual)-r.residual_um)<1e-12
            assert not graph['gap_synthetic'][r.row]
    return dict(rows=len(d),frames=len(frames()),movies=int(d.stem.nunique()),original_gt_ids=True)


def profile(field, center):
    z,y,x = float(center[0]),float(center[1]),float(center[2])
    z0,yp,xp = round(z),round(y/4),round(x/4)
    if field is None: return None, 'no_field'
    if z0-4<0 or z0+4>=field.shape[0] or yp-2<0 or yp+2>=field.shape[1] or xp-2<0 or xp+2>=field.shape[2]:
        return None, 'boundary'
    patch = field[z0-4:z0+5,yp-2:yp+3,xp-2:xp+3]
    if not np.isfinite(patch).all(): return None, 'nonfinite'
    return patch.max(axis=(1,2)), 'supported'


def decode(values, reason):
    if values is None: return 0., False, reason
    assert np.asarray(values).shape == (9,) and np.isfinite(values).all()
    modes = np.flatnonzero(values == np.max(values))
    if len(modes)!=1: return 0.,False,'tie'
    return float((int(modes[0])-4)*VOX[0]),True,'unique'


def controls(out=DEST):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=identity_check()
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    manifest=read(PACK/'ARTIFACT_MANIFEST.json')
    assert sha(CHECKPOINT)==manifest['model']['best_checkpoint']['sha256']
    assert checkpoint['epoch']==2 and checkpoint['config']['pool_factor']==4
    source=PACK/'source_scripts/train_full_frame_center_detector.py'
    assert sha(source)==manifest['contents']['source_scripts'][source.name]['sha256']
    tree=ast.parse(source.read_text(encoding='utf-8'))
    make=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_heatmap')
    ns=dict(np=np,math=math);exec(compile(ast.Module(body=[make],type_ignores=[]),str(source),'exec'),ns)
    h=ns['make_heatmap']((25,25,25),np.array([[12,40,44]],np.float32),4,1.)
    assert np.unravel_index(h.argmax(),h.shape)[0]==12
    score_source=None;tta_enabled=False
    for code in harness.code_cells(BASE):
        tree=ast.parse(code)
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name=='deepcenter_score_point':score_source=n
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Attribute)
                        and isinstance(target.value.value,ast.Name) and target.value.value.id=='os'
                        and target.value.attr=='environ' and isinstance(target.slice,ast.Constant)
                        and target.slice.value=='BIOHUB_DEEPCENTER_TTA'):
                        tta_enabled=ast.literal_eval(n.value)=='1'
    assert score_source is not None
    assert tta_enabled
    from types import SimpleNamespace
    rng=np.random.default_rng(6401);field=rng.random((25,25,25)).astype(np.float32)
    env=dict(np=np,USE_DEEPCENTER_VETO=True,DEEPCENTER_SCORE_WIN_Z=1,DEEPCENTER_SCORE_WIN_YX=2,
             deepcenter_heatmap_for_frame=lambda *args:field)
    exec(compile(ast.Module(body=[score_source],type_ignores=[]),str(BASE),'exec'),env)
    for xy in [32.,33.,34.,35.,36.,38.,40.]:
        vals,reason=profile(field,[12.,xy,xy+4]); assert reason=='supported'
        production=env['deepcenter_score_point']('control',0,(12.,xy,xy+4),{'cfg':SimpleNamespace(pool_factor=4)},{},{})
        assert production==float(np.max(vals[3:6]))
    for index in range(9):
        vals=np.zeros(9,np.float32);vals[index]=1.
        shift,unique,reason=decode(vals,'supported')
        assert shift==(index-4)*1.625 and unique and reason=='unique'
    assert decode(np.ones(9),'supported')==(0.,False,'tie')
    two=np.zeros(9);two[[2,6]]=1;assert decode(two,'supported')==(0.,False,'tie')
    assert decode(None,'boundary')==(0.,False,'boundary')
    assert profile(None,[12,32,32])==(None,'no_field')
    assert profile(field,[0,32,32])[1]=='boundary'
    bad=field.copy();bad[12,8,8]=np.nan;assert profile(bad,[12,32,32])[1]=='nonfinite'
    d=sample();targets=d[['dz_um','dy_um','dx_um']].to_numpy();zero=np.zeros_like(targets)
    assert np.array_equal(targets-zero,targets)
    split=read(PACK/'weights/full_frame_center/split_manifest.json')
    assert all(s.startswith('44b6') for s in split['train']) and all(s.startswith('6bba') for s in split['val'])
    receipt.update(status='passed',recipe=RECIPE,source_z_contract='native z == heatmap z in shipped target and published inference',
                   xy_contract='historically ambiguous; never invert XY or change XY',
                   checkpoint_sha256=sha(CHECKPOINT),checkpoint_epoch=2,source_train=71,validation_selection=128,
                   production_profile_score_exact=True,c023_notebook_tta_enabled=True,synthetic_all9_sign_tie_boundary_nonfinite=True,
                   sample_sha256=sha(SAMPLE),protocol_sha256=sha(PROTOCOL),probe_source_sha256=sha(Path(__file__)))
    save_json(out/'controls.json',receipt);print(json.dumps(receipt),flush=True)


def namespace():
    old.numeric_policy();ns=baseline.namespace()
    assert os.environ['BIOHUB_DEEPCENTER_TTA']=='1'
    assert ns['DEEPCENTER_SCORE_WIN_Z']==1 and ns['DEEPCENTER_SCORE_WIN_YX']==2
    bundle=ns['DEEPCENTER_VETO_DETECTOR']
    assert Path(bundle['path']).resolve()==CHECKPOINT.resolve()
    assert int(bundle['cfg'].pool_factor)==4
    assert next(bundle['model'].parameters()).dtype==torch.float32
    assert all(not module.training for module in bundle['model'].modules())
    return ns


def verify_controls(out):
    control=read(Path(out)/'controls.json')
    assert control['status']=='passed' and control['recipe']==RECIPE
    assert control['probe_source_sha256']==sha(Path(__file__))
    assert control['sample_sha256']==sha(SAMPLE) and control['protocol_sha256']==sha(PROTOCOL)
    assert control['checkpoint_sha256']==sha(CHECKPOINT)
    return control


def model_state_digest(ns):
    digest=hashlib.sha256()
    for key,value in sorted(ns['DEEPCENTER_VETO_DETECTOR']['model'].state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def field_paths(out,stem,t):
    p=Path(out)/'heatmaps'/stem/f'{t:04d}.npy'
    return p,p.with_suffix('.json')


def field_receipt(ns,stem,t,field,seconds):
    raw=ROOT/'data/train'/f'{stem}.zarr/0/c/{t}/0/0/0'
    return dict(stem=stem,t=int(t),seconds=float(seconds),shape=list(field.shape),dtype=str(field.dtype),
                raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),checkpoint_sha256=sha(CHECKPOINT),
                notebook_sha256=sha(BASE),probe_source_sha256=sha(Path(__file__)),tta='unchanged C023 4 flips plus4squareXY views',
                tta_views=8 if field.shape[-1]==field.shape[-2] else 4,
                normalization=dict(vars(ns['DEEPCENTER_VETO_DETECTOR']['cfg'])))


def load_or_compute(out,ns,stem,t):
    p,receipt_path=field_paths(out,stem,t)
    if p.exists():
        receipt=read(receipt_path)
        assert receipt['sha256']==sha(p) and receipt['checkpoint_sha256']==sha(CHECKPOINT)
        assert receipt['notebook_sha256']==sha(BASE) and receipt['probe_source_sha256']==sha(Path(__file__))
        assert receipt['raw_sha256']==sha(ROOT/receipt['raw_path'])
        field=np.load(p);assert field.dtype==np.float32 and list(field.shape)==receipt['shape']
        return field,receipt,True
    p.parent.mkdir(parents=True,exist_ok=True)
    torch.cuda.synchronize();start=time.perf_counter()
    field=ns['deepcenter_heatmap_for_frame'](stem,t,ns['DEEPCENTER_VETO_DETECTOR'],{}, {})
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    assert field is not None and field.dtype==np.float32 and np.isfinite(field).all()
    np.save(p,field);receipt=field_receipt(ns,stem,t,field,seconds);receipt['sha256']=sha(p)
    save_json(receipt_path,receipt)
    return field,receipt,False


def benchmark(out=DEST):
    out=Path(out);verify_controls(out)
    assert not (out/'benchmark.json').exists()
    ns=namespace();allframes=frames();records=[];state_before=model_state_digest(ns)
    selected=[next((stem,t) for stem,t in allframes if stem.startswith(embryo)) for embryo in ['44b6','6bba']]
    assert len(selected)==RECIPE['benchmark_frames']
    for stem,t in selected:
        field,receipt,reused=load_or_compute(out,ns,stem,t)
        assert not reused,'Fresh benchmark must not silently reuse unregistered outcomes'
        center=np.array([field.shape[0]//2,4*(field.shape[1]//2),4*(field.shape[2]//2)],float)
        values,reason=profile(field,center);assert reason=='supported'
        actual=ns['deepcenter_score_point'](stem,t,tuple(center),ns['DEEPCENTER_VETO_DETECTOR'],{}, {(stem,t):field})
        assert actual==float(values[3:6].max())
        records.append(receipt);print('benchmark',stem,t,receipt['seconds'],flush=True)
    secs=[r['seconds'] for r in records]
    assert state_before==model_state_digest(ns)
    save_json(out/'benchmark.json',dict(status='passed',frames=records,seconds=sum(secs),total_frame_count=len(allframes),
              measured_frames=len(records),frozen_model_state_sha256=state_before,model_state_before_equals_after=True,real_production_score_profile_exact=True,
              mean_frame_seconds=float(np.mean(secs)),max_frame_seconds=float(max(secs)),
              remaining_frames=len(allframes)-len(records),estimated_remaining_seconds=float(np.mean(secs)*(len(allframes)-len(records))),
              no_gt_outcomes_computed=True))


def probe(out=DEST):
    out=Path(out);verify_controls(out);assert read(out/'benchmark.json')['status']=='passed'
    identity_check();ns=namespace();d=sample();rows=[];field_records=[];graphs={};state_before=model_state_digest(ns)
    for stem,t in frames():
        field,receipt,reused=load_or_compute(out,ns,stem,t);field_records.append(receipt)
        if stem not in graphs:graphs[stem]=old.load_baseline(C058,stem)
        graph=graphs[stem];ix=np.flatnonzero(graph['txyz'][:,0]==t)
        tree=cKDTree(graph['txyz'][ix,1:]*VOX)
        for r in d[(d.stem==stem)&(d.t==t)].itertuples(index=False):
            center=graph['txyz'][r.row,1:].astype(float)
            assert np.array_equal(center,np.round(center))
            values,reason=profile(field,center);shift,unique,reason=decode(values,reason)
            proposed=center.copy();proposed[0]+=shift/VOX[0]
            assert np.array_equal(proposed[1:],center[1:]) and abs(shift)<=6.5
            distances,nearest=tree.query(proposed*VOX,k=min(2,len(ix)))
            owned=bool(len(ix)==1 or (ix[nearest[0]]==r.row and distances[1]>distances[0]+1e-9))
            target=np.array([r.dz_um,r.dy_um,r.dx_um]);after=target-np.array([shift,0,0])
            row=dict(stem=stem,t=t,embryo=stem[:4],row=int(r.row),node_id=int(r.node_id),gt_id=int(r.gt_id),
                     sample_index=int(r.sample_index),center_z=center[0],center_y=center[1],center_x=center[2],
                     target_z_um=target[0],target_y_um=target[1],target_x_um=target[2],
                     shift_z_um=shift,shift_y_um=0.,shift_x_um=0.,before_3d_um=float(np.linalg.norm(target)),
                     after_3d_um=float(np.linalg.norm(after)),before_absz_um=abs(target[0]),after_absz_um=abs(after[0]),
                     before_bias_z_um=-target[0],before_bias_y_um=-target[1],before_bias_x_um=-target[2],
                     after_bias_z_um=-after[0],after_bias_y_um=-after[1],after_bias_x_um=-after[2],
                     supported=values is not None,unique=unique,reason=reason,ownership=owned,
                     relation='training_source' if stem.startswith('44b6') else 'checkpoint_validation_selected')
            row.update({f'profile_{k}':float(values[k]) if values is not None else None for k in range(9)})
            rows.append(row)
        print('profile',stem,t,'reused' if reused else 'computed',flush=True)
    folder=out/'probe';folder.mkdir(exist_ok=True)
    df=pd.DataFrame(rows).sort_values(['stem','sample_index']);assert len(df)==256
    assert state_before==model_state_digest(ns)
    df.to_csv(folder/'pairs.csv',index=False,float_format='%.17g')
    save_json(folder/'receipt.json',dict(status='complete',rows=len(df),frames=field_records,recipe=RECIPE,
              frozen_model_state_sha256=state_before,model_state_before_equals_after=True,
              pairs_sha256=sha(folder/'pairs.csv'),sample_sha256=sha(SAMPLE)))


def analyse(out=DEST):
    out=Path(out);verify_controls(out);path=out/'probe/pairs.csv';receipt=read(out/'probe/receipt.json')
    assert receipt['pairs_sha256']==sha(path);d=pd.read_csv(path,float_precision='round_trip')
    assert len(d)==256 and not d.duplicated(['stem','node_id','gt_id']).any()
    original=sample().sort_values(['stem','sample_index']).reset_index(drop=True)
    assert np.array_equal(d[['stem','node_id','gt_id','row','t']].to_numpy(),original[['stem','node_id','gt_id','row','t']].to_numpy())
    assert np.array_equal(d[['target_z_um','target_y_um','target_x_um']].to_numpy(),original[['dz_um','dy_um','dx_um']].to_numpy())
    assert not d[['shift_y_um','shift_x_um']].to_numpy().any()
    actual_receipts={(r['stem'],int(r['t'])):r for r in receipt['frames']}
    assert sorted(actual_receipts)==frames()
    graphs={}
    for (stem,t),group in d.groupby(['stem','t']):
        field_path,info_path=field_paths(out,stem,t);info=read(info_path)
        assert info==actual_receipts[(stem,int(t))]
        assert info['sha256']==sha(field_path) and info['raw_sha256']==sha(ROOT/info['raw_path'])
        assert info['checkpoint_sha256']==sha(CHECKPOINT) and info['notebook_sha256']==sha(BASE)
        assert info['probe_source_sha256']==sha(Path(__file__))
        field=np.load(field_path);assert field.dtype==np.float32 and list(field.shape)==info['shape']
        if stem not in graphs:graphs[stem]=old.load_baseline(C058,stem)
        for r in group.itertuples(index=False):
            center=graphs[stem]['txyz'][r.row,1:]
            assert np.array_equal(center,[r.center_z,r.center_y,r.center_x])
            values,reason=profile(field,center);shift,unique,reason=decode(values,reason)
            assert reason==r.reason and shift==r.shift_z_um and unique==r.unique
            if values is not None: assert np.array_equal(values.astype(float),[getattr(r,f'profile_{k}') for k in range(9)])
            else: assert all(pd.isna(getattr(r,f'profile_{k}')) for k in range(9))
    for r in d.itertuples(index=False):
        values=np.array([getattr(r,f'profile_{k}') for k in range(9)]) if r.supported else None
        shift,unique,reason=decode(values,r.reason)
        assert shift==r.shift_z_um and unique==r.unique and reason==r.reason
        target=np.array([r.target_z_um,r.target_y_um,r.target_x_um]);after=target-np.array([shift,0,0])
        assert abs(np.linalg.norm(target)-r.before_3d_um)<1e-12 and abs(np.linalg.norm(after)-r.after_3d_um)<1e-12
        assert abs(target[0])==r.before_absz_um and abs(after[0])==r.after_absz_um
        assert np.array_equal(-target,[r.before_bias_z_um,r.before_bias_y_um,r.before_bias_x_um])
        assert np.array_equal(-after,[r.after_bias_z_um,r.after_bias_y_um,r.after_bias_x_um])
    summaries=[];permovie=[];gates={}
    for embryo,g in d.groupby('embryo'):
        strata={'all':np.ones(len(g),bool),'tail':g.before_3d_um>3.5,'ztail':g.before_absz_um>3.5,
                'good':g.before_3d_um<=2.5,'eligible':g.supported,'excluded':~g.supported,'no_op':g.shift_z_um==0}
        metrics={}
        for name,mask in strata.items():
            sub=g[mask];row=dict(embryo=embryo,stratum=name,n=len(sub),movies=int(sub.stem.nunique()))
            for col in ['before_3d_um','after_3d_um','before_absz_um','after_absz_um','before_bias_z_um','before_bias_y_um','before_bias_x_um','after_bias_z_um','after_bias_y_um','after_bias_x_um']:
                row[col+'_mean']=float(sub[col].mean()) if len(sub) else None
                row[col+'_median']=float(sub[col].median()) if len(sub) else None
            row.update(improved=int((sub.after_3d_um<sub.before_3d_um-1e-12).sum()),worsened=int((sub.after_3d_um>sub.before_3d_um+1e-12).sum()),
                       ownership_conflicts=int(((sub.shift_z_um!=0)&~sub.ownership).sum()),ties=int(sub.reason.eq('tie').sum()),no_op=int(sub.shift_z_um.eq(0).sum()))
            summaries.append(row);metrics[name]=row
            for stem,m in sub.groupby('stem'):
                permovie.append(dict(embryo=embryo,stem=stem,stratum=name,n=len(m),delta_3d=float((m.after_3d_um-m.before_3d_um).mean()),delta_absz=float((m.after_absz_um-m.before_absz_um).mean())))
        conditions={}
        for name in ['all','tail','ztail']:
            a=metrics[name];conditions[name]=bool(a['n'] and a['after_3d_um_mean']<a['before_3d_um_mean'] and a['after_absz_um_mean']<a['before_absz_um_mean'])
        a=metrics['good'];conditions['good']=bool(a['n'] and a['after_3d_um_mean']<=a['before_3d_um_mean']+1e-12 and a['after_absz_um_mean']<=a['before_absz_um_mean']+1e-12)
        for name in ['tail','ztail']:
            wins=sum(x['embryo']==embryo and x['stratum']==name and x['delta_3d']<0 and x['delta_absz']<0 for x in permovie)
            conditions[name+'_movie_wins']=wins>=2
        gates[embryo]=dict(conditions=conditions,pass_gate=all(conditions.values()))
    folder=out/'analysis';folder.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(folder/'summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(permovie).to_csv(folder/'per_movie.csv',index=False,float_format='%.17g')
    passed=all(x['pass_gate'] for x in gates.values())
    save_json(folder/'decision.json',dict(status='component_review_required' if passed else 'closed_fixed_probe_gate_failed',
              gates=gates,rows=len(d),moved=int((d.shift_z_um!=0).sum()),unsupported=int((~d.supported).sum()),
              ownership_conflicts=int(((d.shift_z_um!=0)&~d.ownership).sum()),no_graph_score_measured=True,
              limitation='Tail-enriched correlated256samples;44b6trained6bbavalidationselected;not population or untouched-embryo validation.',
              original_sample_sha256=sha(SAMPLE),pairs_sha256=sha(path),exact_profile_recount=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('verb',choices=['controls','benchmark','probe','analyse'])
    parser.add_argument('--out',type=Path,default=DEST);args=parser.parse_args()
    globals()[args.verb](args.out)
