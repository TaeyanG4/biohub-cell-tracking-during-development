"""Paired C023 division-feature diagnostic using the existing C016 learner.

One original48/temporal56 comparison; whole-embryo fits, fixed threshold0.9.
No scorer rewrite, graph mutation, parameter sweep, or Kaggle operations.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import division_scorer_train as learner
from c055_guarded_readmit import ROOT,BASE,split_stems,sha,save_json,read
from run_last_days_local import Queue,stamp

DEST=ROOT/'experiments/candidates/c059_temporal_division'
RECIPE=dict(epochs=60,lr=.002,weight_decay=.001,hidden=32,seed=0,max_pos_weight=100.,threshold=.9,
    positive_kinds=['strict'],fold='whole_embryo',arms=['original48','temporal56'],
    unknown_cells_are_negative=False,stage='C023 final graph',deployment=False)


def extract_rest(out):
    from c059_temporal_features import extract_movie
    plan=read(out/'plan.json')
    for split,stems in plan['splits'].items():
        for stem in stems:
            if stem in plan['benchmark_stems']:continue
            extract_movie(out,split,stem)


def load(out):
    plan=read(out/'plan.json');xs=[];ys=[];stems=[];parents=[];kinds=[];events=[];records=[]
    for split,movies in plan['splits'].items():
        for stem in movies:
            path=out/'features'/f'{stem}.npz'
            with np.load(path,allow_pickle=False) as z:
                x=z['x'];y=z['y'];kind=z['kinds'];p=z['parents'];ev=z['gt_event_id']
                assert x.shape==(len(y),56) and np.isfinite(x).all()
                mask=(y>=0)&~((y==1)&(kind!='strict'))
                xs.append(x[mask]);ys.append(y[mask]);stems.extend([stem]*int(mask.sum()))
                parents.extend([f'{stem}:{int(v)}' for v in p[mask]])
                kinds.extend(kind[mask]);events.extend([f'{stem}:{int(v)}' if v>=0 else '' for v in ev[mask]])
                records.append(dict(stem=stem,split=split,all_rows=len(y),known_strict_rows=int(mask.sum()),
                    positives=int((y[mask]==1).sum()),unknown=int((y<0).sum()),
                    nonstrict_positive=int(((y==1)&(kind!='strict')).sum()),sha256=sha(path)))
    x=np.concatenate(xs).astype(np.float32);y=np.concatenate(ys).astype(np.float32)
    np.savez_compressed(out/'known_labels.npz',x=x,y=y,stems=np.array(stems),parents=np.array(parents),
        kinds=np.array(kinds),events=np.array(events))
    rows=[]
    for group in ['44b6','6bba']:
        selected=[r for r in records if r['stem'].startswith(group)]
        rows.append(dict(embryo=group,movies=len(selected),rows=sum(r['known_strict_rows'] for r in selected),
            positives=sum(r['positives'] for r in selected),unknown=sum(r['unknown'] for r in selected)))
    assert all(r['positives']>0 for r in rows)
    save_json(out/'label_coverage.json',dict(rows=rows,records=records,data_sha256=sha(out/'known_labels.npz'),
        limitation='Strict candidate rows are not independent biological events; overlapping crops retained inside each embryo only.'))
    print(json.dumps(rows),flush=True)


def fit(out,group,arm):
    torch.set_num_threads(4);coverage=read(out/'label_coverage.json')
    assert sha(out/'known_labels.npz')==coverage['data_sha256']
    with np.load(out/'known_labels.npz') as z:x,y,stems=[z[k] for k in ['x','y','stems']]
    train=np.array([s.startswith(group) for s in stems]);n=48 if arm=='original48' else 56
    train_stems=sorted({str(s) for s in stems[train]})
    assert {s[:4] for s in train_stems}=={group}
    weight=min(RECIPE['max_pos_weight'],float((1-y[train]).sum()/y[train].sum()))
    model,mean,scale=learner.fit(x[train,:n],y[train],60,.002,.001,0,weight,32)
    model_path=out/'models'/f'{group}_{arm}.pt';model_path.parent.mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),
        hidden=32,features=n,training_stems=train_stems,train_embryo=group,recipe=RECIPE),model_path)
    saved=torch.load(model_path,weights_only=True,map_location='cpu');model.load_state_dict(saved['state_dict'])
    assert saved['training_stems']==train_stems and saved['recipe']==RECIPE
    prediction_folder=out/'predictions'/f'{group}_{arm}';prediction_folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for rec in coverage['records']:
        stem=rec['stem']
        if stem.startswith(group):continue
        path=out/'features'/f'{stem}.npz';assert sha(path)==rec['sha256']
        with np.load(path,allow_pickle=False) as z:
            x_all=z['x'][:,:n];prediction=learner.predict(model,mean,scale,x_all)
            assert np.isfinite(prediction).all()
            output=prediction_folder/f'{stem}.npz'
            np.savez_compressed(output,p=prediction,y=z['y'],kinds=z['kinds'],parents=z['parents'],gt_event_id=z['gt_event_id'])
        records.append(dict(stem=stem,file=str(output.relative_to(out)),sha256=sha(output)))
    save_json(out/'models'/f'{group}_{arm}.json',dict(train_embryo=group,test_embryo='6bba' if group=='44b6' else '44b6',
        arm=arm,train_stems=train_stems,training_rows=int(train.sum()),training_positives=int(y[train].sum()),
        pos_weight=weight,recipe=RECIPE,model_sha256=sha(model_path),predictions=records))
    print(group,arm,'complete',len(records),'opposite-embryo movies',flush=True)


def analyse(out):
    rows=[]
    for train in ['44b6','6bba']:
        for arm in RECIPE['arms']:
            meta=read(out/'models'/f'{train}_{arm}.json');assert meta['recipe']==RECIPE
            ys=[];ps=[];parents=[];kinds=[];event_keys=[]
            full=dict(selected=0,unknown=0,strict_positive=0,nonstrict_positive=0,known_negative=0)
            selected_events=set();available_events=set();unknown_candidates=0
            for rec in meta['predictions']:
                path=out/rec['file'];assert sha(path)==rec['sha256'];stem=rec['stem']
                assert not stem.startswith(train)
                with np.load(path) as z:p,y,kind,par,event=[z[k] for k in ['p','y','kinds','parents','gt_event_id']]
                keep=(y>=0)&~((y==1)&(kind!='strict'))
                ys.extend(y[keep]);ps.extend(p[keep]);parents.extend([f'{stem}:{int(i)}' for i in par[keep]])
                kinds.extend(kind[keep]);event_keys.extend([f'{stem}:{int(i)}' for i in event[keep]])
                available_events.update(f'{stem}:{int(i)}' for i in event[(y==1)&(kind=='strict')] if i>=0)
                # All-candidate best choice includes unknown rows; no unknown-negative conversion.
                best={}
                for i,parent in enumerate(par):
                    if int(parent) not in best or p[i]>p[best[int(parent)]]:best[int(parent)]=i
                for i in best.values():
                    if p[i]<.9:continue
                    full['selected']+=1
                    if y[i]<0:full['unknown']+=1
                    elif y[i]==0:full['known_negative']+=1
                    elif kind[i]!='strict':full['nonstrict_positive']+=1
                    else:
                        full['strict_positive']+=1;selected_events.add(f'{stem}:{int(event[i])}')
                unknown_candidates+=int(((y<0)&(p>=.9)).sum())
            y=np.asarray(ys);p=np.asarray(ps);parent=np.asarray(parents);kind=np.asarray(kinds)
            _,tp,fp,fn,by_kind=learner.per_parent(y,p,parent,[.9],kind)[0]
            rows.append(dict(train_embryo=train,test_embryo=meta['test_embryo'],arm=arm,
                known_rows=len(y),positive_rows=int(y.sum()),candidate_average_precision=learner.average_precision(y,p),
                threshold=.9,parent_tp=tp,parent_fp=fp,parent_fn=fn,parent_precision=tp/max(tp+fp,1),
                parent_recall=tp/max(tp+fn,1),fp_by_kind=by_kind,full_parent_choice=full,
                full_choice_strict_annotation_keys=len(selected_events),available_strict_annotation_keys=len(available_events),
                unknown_candidates_above_threshold=unknown_candidates))
    passed=True
    for group in ['44b6','6bba']:
        pair={r['arm']:r for r in rows if r['test_embryo']==group};a,b=pair['temporal56'],pair['original48']
        passed &= a['parent_precision']>=.3 and a['parent_recall']>=.3 and a['parent_precision']>b['parent_precision'] and a['parent_recall']>=b['parent_recall']
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,
        limitation='Restricted known strict-candidate diagnostics; official graph/hidden scores not evaluated. Unknown best choices explicitly retained. Only two embryos; frozen public detector/head saw both.'))
    save_json(out/'decision.json',dict(candidate_gate_passed=bool(passed),graph_integration_requires_review=True,
        gate='Both opposite embryos precision>=.3,restricted recall>=.3,precision>same-data original48,recall>=original48 at fixed.9. Unknown burden/event dedup also require review.'))
    print(json.dumps(rows,indent=2),flush=True)


def prepare(out):
    from c059_temporal_features import FEATURE_NAMES,BENCH_STEMS
    import c055_guarded_readmit as control
    import eval_pp_variants_local as harness
    assert not (out/'plan.json').exists(),'Registered source is immutable'
    splits=split_stems();bench=[read(out/'features'/f'{s}.json') for s in BENCH_STEMS]
    assert len(FEATURE_NAMES)==56 and all(r['status']=='passed' for r in bench)
    inputs={BASE,Path(__file__),ROOT/'src/c059_temporal_features.py',out/'README.md',harness.DEEPCENTER_CHECKPOINT,harness.DEEPCENTER_MANIFEST}
    inputs.update((ROOT/'src').glob('*.py'))
    import c059_temporal_features as feature
    inputs.add(feature.PROBE_SOURCE)
    for split,stems in splits.items():
        inputs.add(control.DEST/'study'/f'{split}.csv')
        for stem in stems:
            inputs.add(control.DEST/'graphs'/split/f'{stem}_control.npz')
            inputs.add(control.run_dir(split)/'edge_cache'/f'{stem}.npz')
            for folder in [control.pred_path(split,stem),ROOT/'data/train'/f'{stem}.geff',ROOT/'data/train'/f'{stem}.zarr']:
                inputs.update(p for p in folder.rglob('*') if p.is_file())
    inputs.update(p for p in (out/'features').glob('*') if p.is_file())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}
    average=sum(r['total_seconds'] for r in bench)/len(bench)
    minutes=int(np.ceil(average*95/60*1.25+15))
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,features=FEATURE_NAMES,splits=splits,
        benchmark_stems=BENCH_STEMS,hashes=hashes,total_jobs=8,max_start_job_hours=6,
        estimated_minutes=minutes,estimate_basis=dict(benchmark_mean_seconds=average,unextracted_movies=95,margin=1.25,learner_hash_minutes=15),
        kaggle_writes=False,graph_mutation=False))
    print('Prepared',len(hashes),'inputs; estimated',minutes,'minutes',flush=True)


def verify(out):
    plan=read(out/'plan.json')
    for p,d in plan['hashes'].items():assert sha(ROOT/p)==d,('input drift',p)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['status.json','queue.lock','launcher.stdout.log','launcher.stderr.log','output_hashes.json'] and not p.name.endswith('.log')}
    save_json(out/'output_hashes.json',files)


def run(out):
    assert not (out/'status.json').exists(),'No blind resume';plan=read(out/'plan.json');q=Queue(out,6)
    try:
        q.state.update(total_jobs=8,plan_sha256=sha(out/'plan.json'));q.save()
        for p,d in plan['hashes'].items():assert sha(ROOT/p)==d,('input drift',p)
        def job(name,verb,*args):q.run(name,[sys.executable,'-u',Path(__file__),verb,'--out',out,*args])
        job('extract_remaining95','extract_rest');job('load_known_strict','load')
        for g in ['44b6','6bba']:
            for arm in RECIPE['arms']:job(f'fit_{g}_{arm}','fit','--group',g,'--arm',arm)
        job('analyse','analyse');job('verify','verify');q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('verb',choices=['prepare','run','extract_rest','load','fit','analyse','verify'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--group',choices=['44b6','6bba']);p.add_argument('--arm',choices=RECIPE['arms'])
    a=p.parse_args()
    if a.verb=='fit':fit(a.out,a.group,a.arm)
    else:globals()[a.verb](a.out)
