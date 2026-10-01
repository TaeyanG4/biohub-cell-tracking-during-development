#!/usr/bin/env python3
"""Thin corrected-fold orchestration of the original C016 loader/learner/metrics."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import division_scorer_train as old
from reid_probe_local import ROOT,sha,save_json,stamp
from c047_hard_example_study import read,verify_hashes
from run_last_days_local import Queue

DEST=ROOT/'experiments/candidates/c050_division_candidate_audit'
BASE=ROOT/'experiments/candidates/c016_division_scorer'
RECIPE=dict(epochs=60,lr=.002,weight_decay=.001,hidden=32,seed=0,max_pos_weight=100.,threshold=.9,
            positive_kinds=['strict'],split='whole_embryo',production=False)
NAMES=['candidates_b00.csv','candidates_b01.csv','candidates_b02.csv','candidates_heldout12.csv','candidates_confirm10.csv']


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'plan.json').exists()
    files=[BASE/n for n in NAMES]
    inputs=files+[Path(__file__),out/'README.md',ROOT/'src/division_scorer_train.py',ROOT/'src/division_scorer_stage.py',ROOT/'src/run_last_days_local.py',ROOT/'src/c047_hard_example_study.py',BASE/'train_logs/C_mlp32_strict.log']
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,files=[str(p.relative_to(ROOT)) for p in files],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},total_jobs=5,max_start_job_hours=2,estimated_minutes=15,
        raw_bytes=sum(p.stat().st_size for p in files)))
    print('Prepared5 jobs;existing C016 learner;estimated15minutes',flush=True)


def load(out):
    plan=read(out/'plan.json');print(stamp(),'loading existing strict-known-label parser',flush=True)
    x,y,stems,rule,parents,kinds=old.load([ROOT/p for p in plan['files']],{'strict'})
    assert x.shape[1]==len(old.FEATURES) and np.isfinite(x).all() and set(np.unique(y))=={0.,1.}
    assert len(set(stems))==97 and {s[:4] for s in stems}=={'44b6','6bba'}
    assert len(set(zip(stems,parents)))<=len(y)
    np.savez_compressed(out/'known_labels.npz',x=x,y=y,stems=stems,rule=rule,parents=parents,kinds=kinds)
    rows=[]
    for g in ['44b6','6bba']:
        ix=np.array([s.startswith(g) for s in stems]);assert y[ix].sum()>0 and (1-y[ix]).sum()>0
        rows.append(dict(embryo=g,movies=len(set(stems[ix])),rows=int(ix.sum()),positive_candidates=int(y[ix].sum()),positive_parents=len(set(parents[ix & (y==1)])),negative_candidates=int((1-y[ix]).sum())))
    save_json(out/'label_coverage.json',dict(rows=rows,data_sha256=sha(out/'known_labels.npz'),features=old.FEATURES))
    print(json.dumps(rows),flush=True)


def fit(out,group):
    torch.set_num_threads(4);plan=read(out/'plan.json');coverage=read(out/'label_coverage.json');assert sha(out/'known_labels.npz')==coverage['data_sha256']
    with np.load(out/'known_labels.npz',allow_pickle=False) as d:x,y,stems,parents,kinds=[d[k] for k in ['x','y','stems','parents','kinds']]
    tr=np.array([s.startswith(group) for s in stems]);te=~tr;training=sorted(set(stems[tr]));testing=sorted(set(stems[te]));assert not set(training)&set(testing)
    pw=min(100.,float((1-y[tr]).sum()/y[tr].sum()));print(stamp(),'fit',group,len(training),'movies',int(tr.sum()),'rows',flush=True)
    model,mean,scale=old.fit(x[tr],y[tr],60,.002,.001,0,pw,32);p=old.predict(model,mean,scale,x[te]);assert np.isfinite(p).all()
    path=out/'models'/(group+'.pt');torch.save(dict(state_dict=model.state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),features=old.FEATURES,hidden=32,train_movies=training,recipe=RECIPE),path)
    pred=out/'predictions'/(group+'.npz');np.savez_compressed(pred,y=y[te],p=p,stems=stems[te],parents=parents[te],kinds=kinds[te])
    save_json(out/'models'/(group+'.json'),dict(train_embryo=group,test_embryo='6bba' if group=='44b6' else '44b6',train_movies=training,test_movies=testing,pos_weight=pw,model_sha256=sha(path),prediction_sha256=sha(pred),embryo_disjoint=True))
    print(stamp(),'fold complete',group,flush=True)


def analyse(out):
    rows=[]
    for group in ['44b6','6bba']:
        info=read(out/'models'/(group+'.json'));path=out/'models'/(group+'.pt');pred=out/'predictions'/(group+'.npz')
        assert sha(path)==info['model_sha256'] and sha(pred)==info['prediction_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);assert state['train_movies']==info['train_movies']
        assert {s[:4] for s in state['train_movies']}=={group} and {s[:4] for s in info['test_movies']}=={info['test_embryo']}
        with np.load(pred,allow_pickle=False) as d:y,p,parents,kinds=[d[k] for k in ['y','p','parents','kinds']]
        threshold,tp,fp,fn,by_kind=old.per_parent(y,p,parents,[.9],kinds)[0]
        rows.append(dict(train_embryo=group,test_embryo=info['test_embryo'],test_movies=len(info['test_movies']),known_rows=len(y),positive_candidates=int(y.sum()),candidate_average_precision=old.average_precision(y,p),threshold=threshold,parent_tp=tp,parent_fp=fp,parent_fn=fn,parent_precision=tp/max(tp+fp,1),parent_recall=tp/max(tp+fn,1),fp_by_kind=by_kind))
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,
        limitations=['Restricted strict-positive known-label candidate diagnostic,not all-division recall.','Historical C012 candidate caches,not C023 integration.','Only2 source embryos,frozen public detectors not held out.','No threshold selection,graph score or submission.']))
    print(json.dumps(rows),flush=True)


def verify(out):
    plan=read(out/'plan.json');verify_hashes(plan['hashes']);assert sha(out/'known_labels.npz')==read(out/'label_coverage.json')['data_sha256']
    files=[out/'known_labels.npz',out/'label_coverage.json',out/'analysis.json']+[p for f in ['models','predictions'] for p in (out/f).iterdir() if p.is_file()]
    save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})


def run(out):
    assert not (out/'status.json').exists(),'No blind restart';plan=read(out/'plan.json');verify_hashes(plan['hashes'])
    for f in ['models','predictions','logs']:(out/f).mkdir(exist_ok=True)
    q=Queue(out,2)
    try:
        q.state.update(total_jobs=5,plan_sha256=sha(out/'plan.json'));q.save()
        def task(name,command,*extra):q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*extra])
        task('load_known_labels','load')
        for group in ['44b6','6bba']:task('fit_'+group,'fit','--group',group)
        task('analyse','analyse');task('verify','verify');q.close('complete_review_required')
    except BaseException as e:q.close('failed',e);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','load','fit','analyse','verify']);parser.add_argument('--out',type=Path,default=DEST);parser.add_argument('--group',choices=['44b6','6bba']);a=parser.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    if a.command=='fit':fit(out,a.group)
    else:globals()[a.command](out)
