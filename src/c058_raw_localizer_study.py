"""C058: one raw-image localizer recipe on a frozen C023 output graph.

Existing C023 replay, official matcher/metric, full-resolution reader, bounded
crop and finite Queue are reused. Unknown nodes never become training negatives.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import c058_raw_localizer_model as model_code
from c055_guarded_readmit import split_stems, sha, save_json, read, BASE
from frame_motion_audit import read_frame, VOX
from local_registration_probe import bounded_crop
from run_last_days_local import Queue, stamp

DEST = ROOT/'experiments/candidates/c058_raw_localizer'
BENCH_STEMS = ['44b6_12dfb391', '6bba_05db0fb1']
CROP = tuple(model_code.RECIPE['crop_zyx'])
JITTER = np.array([1,4,4], dtype=int)
EXPANDED = tuple(map(int,np.array(CROP)+2*JITTER))
RECIPE = dict(model_code.RECIPE, expanded_crop=list(EXPANDED), jitter_vox=JITTER.tolist(),
    sampling='uniform movie, then uniform known eligible point; all residual magnitudes <=7um',
    jitter='uniform integer offsets; use zero jitter if translated target exceeds7um; no reflection',
    folds='train44b6->test6bba;train6bba->test44b6; component validation only',
    inference='all non-gap-synthetic nodes with full real crop support; no GT gate',
    baseline='C023 rounded final nodes; IDs, times, counts and edges frozen',
    epochs_selected_using_target=False, unmatched_nodes='unlabelled')


def numeric_policy():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(RECIPE['seed'])
    np.random.seed(RECIPE['seed'])


def metadata(stem):
    folder=ROOT/'data/train'/f'{stem}.zarr'
    meta=read(folder/'0/zarr.json')
    attrs=read(folder/'zarr.json')['attributes']
    scale=attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1:]
    assert np.array_equal(scale,VOX), (stem,scale)
    assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*meta['shape'][1:]]
    q=attrs['image_statistics']['quantiles']; lo,hi=float(q['0.001']),float(q['0.999'])
    assert hi>lo
    return folder,meta,lo,hi


def get_frame(stem,t,meta_info):
    folder,meta,lo,hi=meta_info
    raw=read_frame(folder,int(t),meta['shape'],np.dtype(meta['data_type']))
    return np.clip((raw.astype(np.float32)-lo)/(hi-lo+1e-6),0.,3.)


def load_baseline(out,stem):
    with np.load(out/'baseline'/f'{stem}.npz') as f:
        return {k:f[k].copy() for k in f.files}


def eligible(graph,shape,crop=CROP):
    center=graph['txyz'][:,1:].astype(int); half=np.array(crop)//2
    return (~graph['gap_synthetic'].astype(bool)) & ((center-half>=0)&(center-half+crop<=np.array(shape)[1:])).all(1)


def training_labels(out,stem,ns):
    import c058_localizer_baseline as baseline
    g=load_baseline(out,stem); meta=metadata(stem); gtpath=ROOT/'data/train'/f'{stem}.geff'
    gt,_=ns['graph_to_plain'](ns['graph_from_geff'](gtpath))
    plain={int(n):tuple(v) for n,v in zip(g['ids'],g['txyz'])}
    p2g=baseline.official_match(g['ids'],g['txyz'],g['edges'],gtpath)
    replica,_=ns['match_nodes_bipartite'](plain,gt,max_dist=7.)
    rows=[]; support=eligible(g,meta[1]['shape'],EXPANDED)
    for row,n in enumerate(g['ids']):
        n=int(n)
        if n not in p2g: continue
        target=np.array(gt[p2g[n]][1:]); center=g['txyz'][row,1:]
        residual=(target-center)*VOX
        assert np.linalg.norm(residual)<=7.+1e-8
        rows.append(dict(row=row,node_id=n,gt_id=int(p2g[n]),t=int(g['txyz'][row,0]),
            dz_um=float(residual[0]),dy_um=float(residual[1]),dx_um=float(residual[2]),
            residual_um=float(np.linalg.norm(residual)),eligible=bool(support[row]),
            gap_synthetic=bool(g['gap_synthetic'][row]),replica_same=replica.get(n)==p2g[n]))
    df=pd.DataFrame(rows)
    # Diagnostic alternatives/identity continuity never select inference nodes.
    lookup={int(n):i for i,n in enumerate(g['ids'])}
    by_t={}
    for gid,v in gt.items():by_t.setdefault(int(v[0]),[]).append((gid,np.array(v[1:])*VOX))
    nearest=[]; ambiguity=[]
    for r in rows:
        alts=by_t[r['t']]; points=np.array([v for _,v in alts]); ids=np.array([n for n,_ in alts])
        d=np.linalg.norm(points-g['txyz'][lookup[r['node_id']],1:]*VOX,axis=1)
        nearest.append(bool(ids[int(d.argmin())]==r['gt_id']))
        ambiguity.append(int((d<=7.).sum()))
    df['assigned_is_nearest_known_gt']=nearest; df['known_gt_in7um']=ambiguity
    folder=out/'labels';folder.mkdir(exist_ok=True)
    df.to_csv(folder/f'{stem}.csv',index=False)
    return df,gt


def extract_one(out,stem,ns):
    start=time.perf_counter(); graph=load_baseline(out,stem); meta=metadata(stem)
    labels,_=training_labels(out,stem,ns); labels=labels[labels.eligible].copy().reset_index(drop=True)
    assert len(labels)>0,stem
    folder=out/'train';folder.mkdir(exist_ok=True)
    crop_path=folder/f'{stem}.npy'
    crops=np.lib.format.open_memmap(crop_path,mode='w+',dtype=np.float16,shape=(len(labels),*EXPANDED))
    for t,selection in labels.groupby('t'):
        frame=get_frame(stem,t,meta)
        for j in selection.index:
            row=int(labels.loc[j,'row']); block,center=bounded_crop(frame,graph['txyz'][row,1:],EXPANDED)
            assert block is not None and np.array_equal(center,graph['txyz'][row,1:])
            # Check axis/order/center against the normalized original frame.
            assert block[tuple(np.array(EXPANDED)//2)]==frame[tuple(center)]
            crops[j]=block
        del frame
    crops.flush();del crops
    targets=labels[['dz_um','dy_um','dx_um']].to_numpy(np.float32)
    np.savez_compressed(folder/f'{stem}.npz',targets=targets,node_ids=labels.node_id.to_numpy(np.int64),
        gt_ids=labels.gt_id.to_numpy(np.int64),rows=labels.row.to_numpy(np.int64))
    result=dict(stem=stem,examples=len(labels),tail_gt3_5=int((labels.residual_um>3.5).sum()),
        seconds=time.perf_counter()-start,crop_bytes=crop_path.stat().st_size,
        known_ambiguous=int((labels.known_gt_in7um>1).sum()),
        not_nearest_known=int((~labels.assigned_is_nearest_known_gt).sum()),
        replica_match_disagreement=int((~labels.replica_same).sum()))
    save_json(folder/f'{stem}.json',result);print(json.dumps(result),flush=True)
    return result


def sample_batch(data,rng):
    # Sampling weights depend only on source-embryo membership, never evaluation.
    stem=sorted(data)[int(rng.integers(len(data)))]; crops,targets=data[stem]
    idx=rng.integers(len(crops),size=RECIPE['batch_size'])
    full=np.asarray(crops[idx],dtype=np.float32)
    target=torch.from_numpy(targets[idx].copy())
    result,target,_=model_code.jitter_batch(torch.from_numpy(full[:,None]),target)
    return result.cuda(),target.cuda(),stem


def load_training(out,group):
    data={}
    files=sorted((out/'train').glob(group+'*.npy'))
    if (out/'plan.json').exists():
        expected=sorted(s for s in read(out/'plan.json')['test_movies'] if s.startswith(group))
        assert [f.stem for f in files]==expected,('unexpected training set',group)
    for file in files:
        with np.load(file.with_suffix('.npz')) as z: targets=z['targets'].copy()
        data[file.stem]=(np.load(file,mmap_mode='r'),targets)
    assert data,group
    return data


def train(out,group):
    numeric_policy(); data=load_training(out,group)
    assert all(s.startswith(group) for s in data)
    model=model_code.RawLocalizer().cuda(); optimizer,scheduler=model_code.make_optimizer(model)
    rng=np.random.default_rng(RECIPE['seed']); history=[];sample_counts={s:0 for s in data}
    start=time.perf_counter()
    for step in range(1,RECIPE['steps']+1):
        x,y,stem=sample_batch(data,rng);sample_counts[stem]+=len(x)
        pred=model(x);loss=model_code.localization_loss(pred,y)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();scheduler.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            history.append(row);print(json.dumps(row),flush=True)
    model.eval();folder=out/'models';folder.mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),recipe=RECIPE,train_embryo=group,
        training_stems=sorted(data),steps=RECIPE['steps']),folder/f'{group}.pt')
    save_json(folder/f'{group}.json',dict(recipe=RECIPE,train_embryo=group,
        eval_embryo='6bba' if group=='44b6' else '44b6',training_stems=sorted(data),
        sample_counts=sample_counts,history=history,seconds=time.perf_counter()-start))


def infer_one(out,stem,model,destination):
    numeric_policy();g=load_baseline(out,stem);meta=metadata(stem);mask=eligible(g,meta[1]['shape'])
    corrected=g['txyz'].copy();shifts=np.zeros((len(mask),3),np.float32);start=time.perf_counter()
    with torch.inference_mode():
        for t in np.unique(g['txyz'][mask,0]):
            indexes=np.flatnonzero(mask & (g['txyz'][:,0]==t));frame=get_frame(stem,t,meta)
            for begin in range(0,len(indexes),64):
                part=indexes[begin:begin+64]
                blocks=[bounded_crop(frame,g['txyz'][r,1:],CROP)[0] for r in part]
                assert all(b is not None for b in blocks)
                x=torch.from_numpy(np.stack(blocks)[:,None]).cuda()
                delta=model(x).cpu().numpy();assert np.isfinite(delta).all()
                assert (np.linalg.norm(delta,axis=1)<7.+1e-6).all()
                shifts[part]=delta
            del frame
    corrected[:,1:]=np.maximum(0,np.rint(g['txyz'][:,1:]+shifts/VOX)).astype(np.int64)
    assert np.array_equal(corrected[~mask],g['txyz'][~mask])
    assert np.array_equal(corrected[:,0],g['txyz'][:,0])
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/f'{stem}.npz',ids=g['ids'],txyz=corrected,edges=g['edges'],
        shifts_um=shifts,eligible=mask,gap_synthetic=g['gap_synthetic'])
    result=dict(stem=stem,nodes=len(mask),eligible=int(mask.sum()),changed=int((corrected!=g['txyz']).any(1).sum()),
        seconds=time.perf_counter()-start,max_shift_um=float(np.linalg.norm(shifts,axis=1).max()))
    save_json(destination/f'{stem}.json',result);print(json.dumps(result),flush=True)
    return result


def infer(out,eval_group):
    numeric_policy(); train_group='6bba' if eval_group=='44b6' else '44b6'
    saved=torch.load(out/'models'/f'{train_group}.pt',weights_only=True,map_location='cpu')
    assert saved['train_embryo']==train_group and saved['steps']==RECIPE['steps'] and saved['recipe']==RECIPE
    assert saved['training_stems']==sorted(s for s in read(out/'plan.json')['test_movies'] if s.startswith(train_group))
    model=model_code.RawLocalizer().cuda();model.load_state_dict(saved['state_dict']);model.eval()
    for stem in read(out/'plan.json')['test_movies']:
        if stem.startswith(eval_group):infer_one(out,stem,model,out/'predictions')


def benchmark(out):
    import c058_localizer_baseline as baseline
    out.mkdir(parents=True,exist_ok=True);numeric_policy(); ns=baseline.namespace()
    rows=[]
    for stem in BENCH_STEMS:
        baseline.capture(stem,'heldout12',out)
        extract_one(out,stem,ns)
        model=model_code.RawLocalizer().cuda().eval()
        row=infer_one(out,stem,model,out/'zero_benchmark')
        old=load_baseline(out,stem)
        with np.load(out/'zero_benchmark'/f'{stem}.npz') as z:
            assert all(np.array_equal(z[k],old[k]) for k in ['ids','txyz','edges'])
        rows.append(row)
    # Training timing on disposable model/source-only crops. No model saved or selected.
    data=load_training(out,'44b6'); model=model_code.RawLocalizer().cuda();opt,sched=model_code.make_optimizer(model)
    rng=np.random.default_rng(RECIPE['seed']);torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(30):
        x,y,_=sample_batch(data,rng);loss=model_code.localization_loss(model(x),y)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();sched.step()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    result=dict(status='passed',created=stamp(),full_movie_zero_rows=rows,zero_graphs_exact=2,
        training_benchmark_steps=30,training_benchmark_seconds=elapsed,
        training_benchmark_checkpoint_saved=False,recipe=RECIPE)
    save_json(out/'benchmark.json',result);print(json.dumps(result),flush=True)


def capture_all(out):
    import c058_localizer_baseline as baseline
    for split,stems in read(out/'plan.json')['splits'].items():
        for stem in stems:
            path=out/'baseline'/f'{stem}.npz'
            if stem in BENCH_STEMS and path.exists():
                proof=read(path.with_suffix('.json'))
                assert proof['status']=='passed' and sha(path)==proof['saved_sha256']
                continue
            baseline.capture(stem,split,out)


def extract_all(out):
    import c058_localizer_baseline as baseline
    with contextlib.redirect_stdout(io.StringIO()):ns=baseline.namespace()
    for stem in read(out/'plan.json')['test_movies']:
        if stem in BENCH_STEMS and (out/'train'/f'{stem}.json').exists():continue
        extract_one(out,stem,ns)


def audit_labels(out):
    rows=[]
    for stem in read(out/'plan.json')['test_movies']:
        labels=pd.read_csv(out/'labels'/f'{stem}.csv'); g=load_baseline(out,stem);meta=metadata(stem)
        assert labels.node_id.is_unique and labels.gt_id.is_unique
        assert labels.residual_um.max()<=7.+1e-8
        support=eligible(g,meta[1]['shape']);expanded=eligible(g,meta[1]['shape'],EXPANDED)
        assert np.array_equal(labels.eligible.to_numpy(),expanded[labels.row.to_numpy(int)])
        assert not labels[labels.eligible].gap_synthetic.any()
        with np.load(out/'train'/f'{stem}.npz') as d:
            assert np.array_equal(d['node_ids'],labels[labels.eligible].node_id.to_numpy(np.int64))
            assert np.allclose(d['targets'],labels[labels.eligible][['dz_um','dy_um','dx_um']].to_numpy(),atol=1e-6,rtol=0)
        rows.append(dict(stem=stem,embryo=stem[:4],nodes=len(support),inference_eligible=int(support.sum()),
            training_eligible=int(labels.eligible.sum()),known_matched=len(labels),
            synthetic=int(g['gap_synthetic'].sum()),boundary_excluded=int((~support & ~g['gap_synthetic']).sum()),
            tail_all=int((labels.residual_um>3.5).sum()),tail_train=int(((labels.residual_um>3.5)&labels.eligible).sum()),
            known_ambiguous=int(((labels.known_gt_in7um>1)&labels.eligible).sum()),
            not_nearest=int(((~labels.assigned_is_nearest_known_gt)&labels.eligible).sum()),
            replica_disagreement=int((~labels.replica_same).sum())))
    df=pd.DataFrame(rows);df.to_csv(out/'label_audit.csv',index=False)
    assert all(df.groupby('embryo').tail_train.sum()>0)
    # Exact zero-head and jitter geometry controls on a real expanded source crop.
    numeric_policy();data=load_training(out,'44b6');first=next(iter(data.values()))
    raw=torch.from_numpy(np.asarray(first[0][:2],np.float32)[:,None]);target=torch.from_numpy(first[1][:2].copy())
    x,y,j=model_code.jitter_batch(raw,target,enabled=False)
    assert torch.equal(x,raw[:,:,1:17,4:68,4:68]) and torch.equal(y,target) and not j.any()
    model=model_code.RawLocalizer();assert torch.count_nonzero(model(x))==0
    x,y,j=model_code.jitter_batch(raw,target)
    assert torch.allclose(y,target-j*torch.tensor(VOX,dtype=target.dtype),atol=1e-6,rtol=0)
    for i,v in enumerate(j.tolist()):
        z,yy,xx=np.array(v)+JITTER
        assert torch.equal(x[i],raw[i,:,z:z+16,yy:yy+64,xx:xx+64])
    loss=model_code.localization_loss(model(x),y);loss.backward()
    assert model.head[-1].weight.grad.abs().sum()>0
    save_json(out/'label_audit.json',dict(status='passed',rows=rows,controls='real crop zero/jitter/finite loss-gradient',
        identity_limitation='Official7um matching is not biological identity ground truth; ambiguity retained and measured, never a runtime gate.',
        fold_limitation='Two biological embryos, overlapping source crops within embryo; frozen public models saw both embryos.'))
    print(df.groupby('embryo').sum(numeric_only=True).to_string(),flush=True)


def evaluate(out):
    import c058_localizer_baseline as baseline
    with contextlib.redirect_stdout(io.StringIO()):ns=baseline.namespace()
    rows=[];diagnostics=[]
    for stem in read(out/'plan.json')['test_movies']:
        old=load_baseline(out,stem)
        with np.load(out/'predictions'/f'{stem}.npz') as f:new={k:f[k].copy() for k in f.files}
        assert all(np.array_equal(old[k],new[k]) for k in ['ids','edges'])
        assert np.array_equal(old['txyz'][:,0],new['txyz'][:,0])
        assert np.array_equal(new['eligible'],eligible(old,metadata(stem)[1]['shape']))
        assert np.array_equal(old['txyz'][~new['eligible']],new['txyz'][~new['eligible']])
        gtpath=ROOT/'data/train'/f'{stem}.geff'
        matches={}
        for label,g in [('zero',old),('localizer',new)]:
            row=baseline.official_score(g['ids'],g['txyz'],g['edges'],gtpath)
            row.update(stem=stem,embryo=stem[:4],arm=label);rows.append(row)
            matches[label]=baseline.official_match(g['ids'],g['txyz'],g['edges'],gtpath)
        before,after=matches['zero'],matches['localizer']
        gt,_=ns['graph_to_plain'](ns['graph_from_geff'](gtpath));lookup={int(n):i for i,n in enumerate(old['ids'])}
        for n,gid in before.items():
            idx=lookup[n];target=np.array(gt[gid][1:]);err_before=np.linalg.norm((old['txyz'][idx,1:]-target)*VOX)
            err_after=np.linalg.norm((new['txyz'][idx,1:]-target)*VOX)
            diagnostics.append(dict(stem=stem,embryo=stem[:4],node_id=n,gt_id=gid,
                eligible=bool(new['eligible'][idx]),before_um=err_before,after_um=err_after,
                before_group='le2_5' if err_before<=2.5 else ('le3_5' if err_before<=3.5 else 'gt3_5'),
                same_identity=after.get(n)==gid,unmatched_after=n not in after,
                changed_identity=n in after and after[n]!=gid))
        print(stem,'matched',len(before),'->',len(after),'same',sum(after.get(n)==g for n,g in before.items()),flush=True)
        save_json(out/'predictions'/f'{stem}_match_transitions.json',dict(
            known_identity_retained=sum(after.get(n)==g for n,g in before.items()),
            known_identity_changed=sum(n in after and after[n]!=g for n,g in before.items()),
            known_identity_lost=sum(n not in after for n in before),
            previously_unmatched_now_matched=sum(n not in before for n in after)))
    pd.DataFrame(rows).to_csv(out/'official_rows.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'paired_residuals.csv',index=False)


def writer_verify(out):
    import c058_localizer_baseline as baseline
    plan=read(out/'plan.json');splits={s:k for k,v in plan['splits'].items() for s in v}
    expected=pd.read_csv(out/'official_rows.csv')
    results=[]
    for label,folder in [('zero','baseline'),('localizer','predictions')]:
        graphs={}
        for stem in plan['test_movies']:
            with np.load(out/folder/f'{stem}.npz') as f:graphs[stem]=tuple(f[k].copy() for k in ['ids','txyz','edges'])
        csv=baseline.write_frozen_csv(graphs,splits,out/'writer'/label)
        evaluation=out/'writer'/label/'official'
        subprocess.run([sys.executable,ROOT/'src/evaluate_local.py','--csv',csv,'--gt-dir',ROOT/'data/train','--out-dir',evaluation],check=True)
        actual=pd.read_csv(evaluation/'per_dataset.csv').set_index('dataset')
        exp=expected[expected.arm==label].set_index('stem')
        assert set(actual.index)==set(exp.index)
        for stem in exp.index:
            for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes']:
                assert int(actual.loc[stem,k])==int(exp.loc[stem,k]),(label,stem,k)
            assert abs(actual.loc[stem,'adj_edge_jaccard']-exp.loc[stem,'adj_edge_jaccard'])<1e-12
        results.append(dict(arm=label,movies=len(actual),status='passed',csv_sha256=sha(csv)))
    save_json(out/'writer_verification.json',dict(status='passed',rows=results))


def analyse(out):
    import c058_localizer_baseline as baseline
    plan=read(out/'plan.json');data=pd.read_csv(out/'official_rows.csv');res=pd.read_csv(out/'paired_residuals.csv')
    groups=dict(all22=plan['test_movies'],**plan['splits'])
    groups.update({g:[s for s in plan['test_movies'] if s.startswith(g)] for g in ['44b6','6bba']})
    rows=[]
    for group,stems in groups.items():
        parts={a:data[(data.arm==a)&data.stem.isin(stems)] for a in ['zero','localizer']}
        scores={a:baseline.official_summary(d.to_dict('records')) for a,d in parts.items()}
        r=dict(group=group,movies=len(stems),zero=scores['zero'],localizer=scores['localizer'],
            delta_total=scores['localizer']['score']-scores['zero']['score'],
            delta_edge=scores['localizer']['adj_edge_jaccard']-scores['zero']['adj_edge_jaccard'])
        for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']:
            r['delta_'+k]=int(parts['localizer'][k].sum()-parts['zero'][k].sum())
        delta=parts['localizer'].set_index('stem').adj_edge_jaccard-parts['zero'].set_index('stem').adj_edge_jaccard
        r.update(movie_wins=int((delta>1e-12).sum()),movie_losses=int((delta< -1e-12).sum()))
        tail=res[res.stem.isin(stems)&res.eligible&(res.before_group=='gt3_5')]
        r.update(tail_count=len(tail),tail_before=float(tail.before_um.mean()),tail_after=float(tail.after_um.mean()))
        rows.append(r)
    index={r['group']:r for r in rows}
    gate=all(index[g]['delta_edge']>0 and index[g]['tail_after']<index[g]['tail_before'] and
             index[g]['delta_edge_tp']>0 and index[g]['movie_wins']>=2 for g in ['44b6','6bba']) and index['all22']['delta_total']>0
    verification=read(out/'writer_verification.json');assert verification['status']=='passed'
    save_json(out/'analysis.json',dict(status='complete_review_required',created=stamp(),rows=rows,
        official_metric='vendored organizer via original CSV writer',limitations='New component whole-embryo only; unknown biological identity remains unverified; no hidden-score claim.'))
    save_json(out/'decision.json',dict(pilot_gate_passed=bool(gate),unchanged97_requires_manual_review=True,
        reason='Both opposite embryos must improve adjusted edge, tail residual and edge TP with >=2 movie wins; positive all22 total.'))
    print(json.dumps(rows,indent=2),flush=True);print('PILOT_GATE',gate,flush=True)


def prepare(out):
    import c055_guarded_readmit as old
    assert not (out/'plan.json').exists(),'Registered recipe is immutable'
    bench=read(out/'benchmark.json');assert bench['status']=='passed' and bench['zero_graphs_exact']==2
    assert bench['recipe']==RECIPE,'Benchmark recipe drift'
    splits={k:v for k,v in split_stems().items() if k!='extension75'};movies=sum(splits.values(),[])
    inputs={Path(__file__),ROOT/'src/c058_localizer_baseline.py',ROOT/'src/c058_raw_localizer_model.py',BASE,
        ROOT/'src/eval_pp_variants_local.py',ROOT/'src/evaluate_local.py',ROOT/'src/run_last_days_local.py',
        ROOT/'src/frame_motion_audit.py',ROOT/'src/local_registration_probe.py',ROOT/'src/c055_guarded_readmit.py',
        out/'benchmark.json',out/'README.md',out/'writer_preflight/verification.json',
        out/'writer_preflight/writer_parity.json',out/'writer_preflight/submission.csv',
        ROOT/'state/research_reset_20260929/c058_writer_preflight.py'}
    for split,stems in splits.items():
        inputs.add(old.DEST/'study'/f'{split}.csv')
        for stem in stems:
            inputs.add(old.DEST/'graphs'/split/f'{stem}_control.npz')
            inputs.add(old.run_dir(split)/'edge_cache'/f'{stem}.npz')
            for folder in [old.pred_path(split,stem),ROOT/'data/train'/f'{stem}.geff',ROOT/'data/train'/f'{stem}.zarr']:
                inputs.update(p for p in folder.rglob('*') if p.is_file())
    import evaluate_local
    inputs.update(p for p in evaluate_local.VENDOR_SRC.parent.rglob('*.py') if p.is_file())
    inputs.update(p for p in (ROOT/'src').glob('*.py'))
    import eval_pp_variants_local as harness
    inputs.update([harness.DEEPCENTER_CHECKPOINT,harness.DEEPCENTER_MANIFEST])
    for folder in ['baseline','train','labels','zero_benchmark']:
        inputs.update(p for p in (out/folder).rglob('*') if p.is_file())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}
    nodes=sum(len(np.load(old.DEST/'graphs'/split/f'{s}_control.npz')['ids']) for split,ss in splits.items() for s in ss)
    rate=sum(r['seconds'] for r in bench['full_movie_zero_rows'])/sum(r['nodes'] for r in bench['full_movie_zero_rows'])
    # Measured inference + two fixed fits + baseline/labels/writer/hash headroom.
    estimate=int(np.ceil((nodes*rate+2*RECIPE['steps']*bench['training_benchmark_seconds']/30)/60+25))
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,hashes=hashes,splits=splits,test_movies=movies,
        total_jobs=10,max_start_job_hours=4,estimated_minutes=estimate,
        estimate_basis=dict(pilot_nodes=nodes,seconds_per_full_node=rate,training30seconds=bench['training_benchmark_seconds'],fixed_overhead_minutes=25),
        training_selection='all22 known official7um pairs in opposite embryo with real expanded crop support',
        gate='Both opposite embryos edge>0,tail residual improves,edgeTP>0,>=2 movie wins;all22total>0',
        deployment=False,kaggle_writes=False))
    print('Prepared C058',len(hashes),'pinned inputs; ETA minutes',estimate,flush=True)


def run(out):
    assert not (out/'status.json').exists(),'No blind resume'
    plan=read(out/'plan.json');q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=10,plan_sha256=sha(out/'plan.json'));q.save()
        for file,digest in plan['hashes'].items():assert sha(ROOT/file)==digest,('input drift',file)
        for verb in ['capture_all','extract_all','audit_labels']:
            q.run(verb,[sys.executable,'-u',Path(__file__),verb,'--out',out])
        for verb in ['train','infer']:
            for group in ['44b6','6bba']:
                q.run(verb+'_'+group,[sys.executable,'-u',Path(__file__),verb,'--out',out,'--group',group])
        for verb in ['evaluate','writer_verify','analyse']:
            q.run(verb,[sys.executable,'-u',Path(__file__),verb,'--out',out])
        for file,digest in plan['hashes'].items():assert sha(ROOT/file)==digest,('input drift',file)
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['queue.lock','status.json','launcher.stdout.log','launcher.stderr.log','output_hashes.json']}
        save_json(out/'output_hashes.json',hashes);q.close('complete_review_required')
    except Exception as exc:
        q.close('failed',exc);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('verb',choices=['benchmark','train','infer','capture_all','extract_all','audit_labels','evaluate','writer_verify','analyse','prepare','run'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--group',choices=['44b6','6bba'])
    a=p.parse_args()
    if a.verb=='benchmark':benchmark(a.out)
    elif a.verb=='train':train(a.out,a.group)
    elif a.verb=='infer':infer(a.out,a.group)
    else:globals()[a.verb](a.out)


if __name__=='__main__':main()
