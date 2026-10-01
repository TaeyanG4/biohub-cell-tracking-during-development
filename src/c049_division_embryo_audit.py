#!/usr/bin/env python3
"""Corrected two-embryo C031 diagnostic, reusing its fit/predict and standard metrics."""
from __future__ import annotations
import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
import division_cnn_train as cnn
from reid_probe_local import ROOT, CONTROL, sha, save_json, stamp, setup_ns
from run_last_days_local import Queue
from c047_hard_example_study import read, verify_hashes

DEST=ROOT/'experiments/candidates/c049_division_embryo_audit'
C031=ROOT/'experiments/candidates/c031_division_cnn'
COVERAGE=ROOT/'state/split_audit_20260928/division_audit/coverage.json'
RECIPE=dict(epochs=25,lr=.001,pos_weight=3.,seed=0,threshold=.5,
            split='whole embryo,opposite evaluation only',training='unchanged C031 fit/predict',production=False)


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'plan.json').exists()
    coverage=read(COVERAGE);verify_hashes(coverage['source_hashes'])
    files=sorted((C031/'crops').glob('*.npz'));assert len(files)==199
    inputs={Path(__file__),out/'README.md',COVERAGE,ROOT/'src/division_cnn_train.py',ROOT/'src/division_crops_extract.py',
        ROOT/'src/reid_probe_local.py',ROOT/'src/run_last_days_local.py',ROOT/'src/c047_hard_example_study.py',
        C031/'division_cnn.json',C031/'train_cv.log'}
    inputs.update(files)
    for p in files:inputs.update(x for x in (ROOT/'data/train'/(p.stem+'.geff')).rglob('*') if x.is_file())
    folds=[]
    for train in ['44b6','6bba']:
        test='6bba' if train=='44b6' else '44b6';a=[p.stem for p in files if p.stem.startswith(train)];b=[p.stem for p in files if p.stem.startswith(test)]
        assert a and b and not set(a)&set(b)
        folds.append(dict(train_embryo=train,test_embryo=test,train_movies=a,test_movies=b))
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,folds=folds,total_jobs=4,max_start_job_hours=6,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)},coverage=coverage['summary']))
    print('Prepared4 finite jobs;199 existing crops;2 whole-embryo fits;no new scorer',flush=True)


def runtime():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    return dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),fp32=True,amp=False,tf32=False)


def preflight(out):
    plan=read(out/'plan.json');verify_hashes(plan['hashes']);rt=runtime()
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    checked=[];smoke={}
    for p in sorted((C031/'crops').glob('*.npz')):
        nodes,edges=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(p.stem+'.geff')))
        by_xyz=collections.defaultdict(list);outdegree=collections.Counter(a for a,b in edges)
        for nid,xyz in nodes.items():by_xyz[tuple(int(round(float(x))) for x in xyz)].append(nid)
        with np.load(p,allow_pickle=False) as d:
            y=d['label'];x=d['crops'];coords=d['tzyx'];kind=d['kind']
            assert set(np.unique(y))<={0,1} and np.isfinite(x).all() and x.shape[1:]==(3,8,32,32)
            for label,xyz,k in zip(y,coords,kind):
                ids=by_xyz[tuple(map(int,xyz))];assert ids,(p.name,xyz)
                assert any(outdegree[i]>=2 if label==1 else outdegree[i]==1 for i in ids),(p.name,xyz,label)
                assert str(k)==('gt_division' if label==1 else 'gt_single_child')
            checked.append(dict(stem=p.stem,samples=len(y),positives=int(y.sum())))
            if p.stem[:4] not in smoke and int(y.sum())>0:
                ix=np.r_[np.flatnonzero(y==1)[:1],np.flatnonzero(y==0)[:1]]
                smoke[p.stem[:4]]=(x[ix].astype(np.float32),y[ix].astype(np.float32))
    assert len(checked)==199 and sum(r['samples'] for r in checked)==8111 and sum(r['positives'] for r in checked)==151
    torch.manual_seed(0);smoke_rows=[]
    for group,(x,y) in smoke.items():
        model=cnn.DivisionCNN().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        batch=torch.from_numpy(x).cuda();labels=torch.from_numpy(y).cuda();before=model.head[-1].weight.detach().clone()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(batch),labels)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
        assert not torch.equal(before,model.head[-1].weight);model.eval();clone=cnn.DivisionCNN().cuda().eval();clone.load_state_dict(model.state_dict())
        with torch.no_grad():assert torch.equal(model(batch),clone(batch))
        smoke_rows.append(dict(group=group,gradient_update_reload=True,loss=float(loss)))
    save_json(out/'preflight.json',dict(status='passed',runtime=rt,gt_checks=checked,smoke=smoke_rows,known_labels=8111))


def fit_fold(out,group):
    plan=read(out/'plan.json');fold=next(f for f in plan['folds'] if f['train_embryo']==group);runtime()
    verify_hashes(plan['hashes']);assert read(out/'preflight.json')['status']=='passed'
    def load(stems):
        xs=[];ys=[];meta=[]
        for stem in stems:
            with np.load(C031/'crops'/(stem+'.npz'),allow_pickle=False) as d:
                xs.append(d['crops'].astype(np.float32));ys.append(d['label'].astype(np.int64))
                meta.extend((stem,i) for i in range(len(d['label'])))
        return np.concatenate(xs),np.concatenate(ys),meta
    x,y,_=load(fold['train_movies']);device=torch.device('cuda')
    print(stamp(),'fit',group,'movies',len(fold['train_movies']),'samples',len(y),'positives',int(y.sum()),'epochs25',flush=True)
    model=cnn.fit(x,y,25,.001,device,0,3.);del x,y
    xv,yv,meta=load(fold['test_movies']);pv=cnn.predict(model,xv,device);assert len(pv)==len(yv) and np.isfinite(pv).all()
    assert {s[:4] for s in fold['train_movies']}=={group} and {s[:4] for s in fold['test_movies']}=={fold['test_embryo']}
    path=out/'models'/(group+'.pt');torch.save(dict(state_dict=model.cpu().state_dict(),recipe=RECIPE,train_movies=fold['train_movies'],train_embryo=group,plan_sha256=sha(out/'plan.json')),path)
    pred=out/'predictions'/(fold['test_embryo']+'.npz');np.savez_compressed(pred,label=yv,probability=pv,stem=np.array([s for s,i in meta]),row=np.array([i for s,i in meta]))
    save_json(out/'models'/(group+'.json'),dict(train_embryo=group,test_embryo=fold['test_embryo'],train_movies=fold['train_movies'],test_movies=fold['test_movies'],model_sha256=sha(path),prediction_sha256=sha(pred),embryo_disjoint=True))
    print(stamp(),'complete',group,'opposite-embryo samples',len(yv),flush=True)


def analyse(out):
    plan=read(out/'plan.json');rows=[];all_y=[];all_p=[]
    for fold in plan['folds']:
        info=read(out/'models'/(fold['train_embryo']+'.json'));model=out/'models'/(fold['train_embryo']+'.pt');pred=out/'predictions'/(fold['test_embryo']+'.npz')
        assert sha(model)==info['model_sha256'] and sha(pred)==info['prediction_sha256']
        state=torch.load(model,map_location='cpu',weights_only=True);assert state['train_movies']==fold['train_movies'] and not set(state['train_movies'])&set(fold['test_movies'])
        with np.load(pred,allow_pickle=False) as d:y,p=d['label'],d['probability']
        precision,recall,thresholds=precision_recall_curve(y,p);eligible=np.flatnonzero(recall>=.5);idx=int(eligible[-1])
        tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
        row=dict(test_embryo=fold['test_embryo'],train_embryo=fold['train_embryo'],test_movies=len(fold['test_movies']),samples=len(y),positives=int(y.sum()),prevalence=float(y.mean()),average_precision=float(average_precision_score(y,p)),threshold=.5,tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),precision_at_recall_atleast_half=float(precision[idx]),achieved_recall=float(recall[idx]),descriptive_threshold=float(thresholds[idx]),threshold_must_not_be_deployed=True)
        rows.append(row);all_y.append(y);all_p.append(p)
        np.savez_compressed(out/'predictions'/(fold['test_embryo']+'_pr_curve.npz'),precision=precision,recall=recall,thresholds=thresholds)
    save_json(out/'analysis.json',dict(status='complete_review_required',folds=rows,
        pooled_average_precision=float(average_precision_score(np.concatenate(all_y),np.concatenate(all_p))),
        limitations=['Only2 embryos;sampled GT-centre crops,not all predicted candidates.','No official graph score or notebook integration.','No threshold selected for deployment;PR operating points are descriptive.','Frozen public pipeline saw both embryos;this diagnostic only tests new classifier transfer.','Exact raw-crop uniqueness is not independent-event count.'],
        prior_report_correction='C031 precision44/76=.579 at top76 corresponded recall44/151=.291,not.5.'))
    print(json.dumps(rows),flush=True)


def run(out):
    assert not (out/'status.json').exists(),'Review failure and artifacts before any resume'
    plan=read(out/'plan.json');verify_hashes(plan['hashes'])
    for folder in ['models','predictions','logs']:(out/folder).mkdir(exist_ok=True)
    q=Queue(out,6)
    try:
        q.state.update(total_jobs=4,plan_sha256=sha(out/'plan.json'));q.save()
        def task(name,command,*extra):q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*extra])
        task('preflight','preflight')
        for group in ['44b6','6bba']:task('fit_'+group,'fit_fold','--group',group)
        task('analyse','analyse');verify_hashes(plan['hashes'])
        files=[p for folder in ['models','predictions'] for p in (out/folder).rglob('*') if p.is_file()]+[out/'preflight.json',out/'analysis.json']
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files});q.close('complete_review_required')
    except BaseException as e:q.close('failed',e);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','preflight','fit_fold','analyse']);parser.add_argument('--out',type=Path,default=DEST);parser.add_argument('--group',choices=['44b6','6bba']);args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    if args.command=='fit_fold':fit_fold(out,args.group)
    else:globals()[args.command](out)
