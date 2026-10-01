#!/usr/bin/env python3
"""Whole-embryo data expansion; reuse C035 training and original official replay."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import reid_augmented_local as aug
import c038_followup_local as audit
from c041_fixed_models import make_notebook
from c047_hard_example_study import read, verify_hashes, smoke as real_crop_smoke
from reid_probe_local import ROOT, BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from run_last_days_local import Queue, replay_command, extension_stems

DEST=ROOT/'experiments/candidates/c048_embryo_holdout_appearance'
C035=ROOT/'experiments/candidates/c035_augmented_reid'
C041=ROOT/'experiments/candidates/c041_fixed_models'
C046=ROOT/'experiments/candidates/c046_fixed_appearance_extension'
C038=ROOT/'experiments/candidates/c038_complementary_fusion/followup_extension'
AUDIT=ROOT/'state/split_audit_20260928'
POLICY=dict(split='whole embryo, opposite model only',max_triplets_per_movie=192,hard_cap=96,
            hard_separation_um=[6.,12.],train_movies='ALL GT-eligible movies in training embryo',
            sampling='even temporal coverage of up to96 hard; fill cap192 from remaining known eligible triplets',
            training='unchanged C035 weak_aug1200steps per embryo,seed3501,equal movie sampling',
            no_unknown_negatives=True,no_checkpoint_search=True,deployment=False)


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'plan.json').exists()
    inventory=read(AUDIT/'all199_training_inventory.json')
    selected=[dict(r,kind='new_hard' if r['hard_links'] else 'general') for r in inventory['rows'] if r['eligible_links']>0]
    assert len(selected)==190
    hashes={};inputs={Path(__file__),out/'README.md',AUDIT/'REVIEW.md',AUDIT/'all199_training_inventory.json',
        ROOT/'src/c047_hard_example_study.py',ROOT/'src/reid_augmented_local.py',ROOT/'src/c041_fixed_models.py'}
    for folder in [C046,C038]:
        previous=read(folder/'plan.json');verify_hashes(previous['hashes']);hashes.update(previous['hashes']);inputs.add(folder/'plan.json')
    for item in inventory['rows']:
        folder=ROOT/'data/train'/(item['stem']+'.geff');inputs.update(p for p in folder.rglob('*') if p.is_file())
        for name,digest in item['image_metadata'].items(): assert sha(ROOT/name)==digest;hashes[name]=digest
    old_status=read(C035/'status.json');assert old_status['status']=='complete';verify_hashes(old_status['source_hashes'])
    inputs.add(C035/'status.json')
    for group in ['44b6','6bba']:
        path=C035/'models'/(group+'_weak_aug.pt')
        recorded=old_status['jobs']['model:'+group+':weak_aug']['artifacts'][str(path.relative_to(ROOT))]
        assert sha(path)==recorded;inputs.add(path)
    source=(C041/'portable_appearance.py').read_text(encoding='utf-8')
    inputs.add(C041/'portable_appearance.py')
    save_json(out/'variants.json',{'off':{'C038_MODE':'off'},'appearance':{'C038_MODE':'appearance'}});inputs.add(out/'variants.json')
    pilot=aug.evaluation_plan();extension,_=extension_stems({s for _,s in pilot})
    jobs=[]
    for split in ['heldout12','confirm10','extension75']:
        stems=extension if split=='extension75' else [s for sp,s in pilot if sp==split]
        for test in ['44b6','6bba']:
            subset=[s for s in stems if s.startswith(test)]
            if not subset:continue
            train='6bba' if test=='44b6' else '44b6'
            old_name=('extension75_' if split=='extension75' else 'pilot_'+split+'_')+test
            name='holdout_'+split+'_'+test
            inputs.add(make_notebook(out,name,out/'models'/(train+'_weak_aug.pt'),source))
            jobs.append(dict(name=name,kind='embryo_holdout',split=split,test_embryo=test,train_embryo=train,
                stems=subset,run=str(CONTROL/('control_fp32_'+split)),old_name=old_name))
            inputs.add(C038/'replay'/(old_name+'.csv'))
            inputs.update(p for p in (C038/'graphs'/old_name).iterdir() if p.is_file())
            if split=='heldout12':
                label='reference_'+test
                inputs.add(make_notebook(out,label,C035/'models'/(train+'_weak_aug.pt'),source))
                jobs.append(dict(name=label,kind='reference',split=split,test_embryo=test,train_embryo=train,
                    stems=subset[:1],run=str(CONTROL/('control_fp32_'+split)),old_name=old_name))
    for path in inputs:
        assert path.is_file(),path
        hashes[str(path.relative_to(ROOT))]=sha(path)
    batches=[[r['stem'] for r in selected[i:i+10]] for i in range(0,len(selected),10)]
    assert len(batches)==19 and len(jobs)==7
    folds=[]
    for test in ['44b6','6bba']:
        train='6bba' if test=='44b6' else '44b6'
        folds.append(dict(train_embryo=train,test_embryo=test,
            train_movies=[r['stem'] for r in selected if r['stem'].startswith(train)],
            test_movies=[s for j in jobs if j['kind']=='embryo_holdout' and j['test_embryo']==test for s in j['stems']]))
    save_json(out/'plan.json',dict(created=stamp(),policy=POLICY,recipe_compatibility=aug.RECIPE,
        selected=selected,inventory_summary=inventory['summary'],folds=folds,extract_batches=batches,jobs=jobs,
        hashes=hashes,total_jobs=32,max_start_job_hours=8,deployment=False))
    print('Prepared32 jobs;190 eligible train movies;two embryo-held-out fits;97 cached evaluations;',len(hashes),'pinned inputs',flush=True)


def choose(triples,distances,nodes):
    def ordered(ix):return np.array(sorted(ix,key=lambda i:(nodes[int(triples[i,0])][0],*map(int,triples[i]))),int)
    def spread(ix,count):return ix if len(ix)<=count else ix[np.linspace(0,len(ix)-1,count,dtype=int)]
    hard=ordered(np.flatnonzero((distances>=6.) & (distances<=12.)))
    keep=spread(hard,min(96,len(hard)))
    rest=ordered(np.setdiff1d(np.arange(len(triples)),keep))
    keep=np.r_[keep,spread(rest,min(192-len(keep),len(rest)))]
    assert len(keep)==min(192,len(triples)) and len(keep)==len(np.unique(keep))
    return ordered(keep)


def extract_batch(out,index):
    plan=read(out/'plan.json');by_stem={r['stem']:r for r in plan['selected']}
    for stem in plan['extract_batches'][index]:
        item=by_stem[stem]
        with contextlib.redirect_stdout(io.StringIO()): ns,frames,_=setup_ns(CONTROL/'control_fp32_heldout12')
        nodes,triples,distances,digest=aug.gt_triplets(ns,stem)
        assert digest==item['graph_sha256'] and aug.data_meta(stem)==item['image_metadata']
        assert len(triples)==item['eligible_links'] and int((distances<=12.).sum())==item['hard_links']
        keep=choose(triples,distances,nodes);triples,distances=triples[keep],distances[keep]
        ids=np.unique(triples);crops,coords=aug.patches(ns,frames,stem,ids,nodes)
        target=out/'train'/(stem+'.npz')
        np.savez_compressed(target,crops=crops,triplets=np.searchsorted(ids,triples),ids=ids,tzyx=coords,
                            negative_distance_um=distances.astype(np.float32))
        save_json(out/'train'/(stem+'.json'),dict(stem=stem,graph_sha256=digest,image_metadata=item['image_metadata'],
            patches=len(ids),triplets=len(triples),hard_triplets=int((distances<=12.).sum()),crop_sha256=sha(target),policy=POLICY))
        print(stamp(),stem,'triplets',len(triples),'hard',int((distances<=12.).sum()),flush=True)


def train(out,group):
    plan=read(out/'plan.json');verify_hashes(read(out/'training_data_hashes.json'),out);aug.runtime_policy()
    result=aug.train(out,plan,group,'weak_aug',lambda s:print(stamp(),s,flush=True))
    path=out/'models'/(group+'_weak_aug.pt');state=torch.load(path,map_location='cpu',weights_only=True)
    state.update(training_policy=POLICY,c048_plan_sha256=sha(out/'plan.json'),
                 recipe_note='recipe is immutable C041 runtime compatibility; training selection/sampling is in training_policy and train_movies',
                 training_data_manifest_sha256=sha(out/'training_data_hashes.json'))
    torch.save(state,path);result.update(model_sha256=sha(path),training_policy=POLICY)
    save_json(path.with_suffix('.json'),result)


def verify_folds(out):
    plan=read(out/'plan.json');rows=[]
    for fold in plan['folds']:
        path=out/'models'/(fold['train_embryo']+'_weak_aug.pt');state=torch.load(path,map_location='cpu',weights_only=True)
        assert sha(path)==read(path.with_suffix('.json'))['model_sha256']
        assert state['training_policy']==POLICY and state['c048_plan_sha256']==sha(out/'plan.json')
        assert set(state['train_movies'])==set(fold['train_movies'])
        assert not set(state['train_movies']) & set(fold['test_movies'])
        assert {s[:4] for s in state['train_movies']}=={fold['train_embryo']}
        assert {s[:4] for s in fold['test_movies']}=={fold['test_embryo']}
        assert fold['train_embryo']!=fold['test_embryo']
        rows.append(dict(train_embryo=fold['train_embryo'],test_embryo=fold['test_embryo'],
            train_movies=len(fold['train_movies']),eval_movies=len(fold['test_movies']),model_sha256=sha(path),embryo_disjoint=True))
    save_json(out/'fold_proof.json',dict(status='passed',folds=rows,
        limit='Frozen detector/head saw both embryos; only newly fitted appearance component is held out.'))


def check_reference(out):
    for j in read(out/'plan.json')['jobs']:
        if j['kind']!='reference':continue
        frame=pd.read_csv(out/'replay'/(j['name']+'.csv'));ref=pd.read_csv(C038/'replay'/(j['old_name']+'.csv'))
        ref=ref[ref.stem.isin(j['stems'])];audit.verify(frame,ref,{'off':'off','appearance':'appearance'})
        for stem in j['stems']:
            for mode in ['off','appearance']:
                assert audit.graph(out/'graphs'/j['name'],stem,mode)==audit.graph(C038/'graphs'/j['old_name'],stem,mode)
    save_json(out/'reference_parity.json',dict(status='passed',exact_full_graphs=4,exact_metric_rows=4))


def analyse(out):
    plan=read(out/'plan.json');verify_folds(out);check_reference(out)
    jobs=[j for j in plan['jobs'] if j['kind']=='embryo_holdout']
    data=pd.concat([pd.read_csv(out/'replay'/(j['name']+'.csv')).assign(split=j['split']) for j in jobs],ignore_index=True)
    base=pd.concat([pd.read_csv(OLD/('control_fp32_'+s+'.csv')) for s in ['heldout12','confirm10','extension75']])
    old=pd.concat([pd.read_csv(C038/'replay'/(j['old_name']+'.csv')) for j in jobs],ignore_index=True)
    assert data.stem.nunique()==97;audit.verify(data,base,{'off':'as_configured'})
    for j in jobs:
        for stem in j['stems']:
            assert audit.graph(out/'graphs'/j['name'],stem,'off')==audit.graph(C038/'graphs'/j['old_name'],stem,'off')
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    rows=[]
    for split in ['heldout12','confirm10','pilot22','extension75','aggregate97']:
        stems=set(data.stem) if split=='aggregate97' else set(data[data.split.isin(['heldout12','confirm10'])].stem) if split=='pilot22' else set(data[data.split==split].stem)
        for group in ['all','44b6','6bba']:
            keep=stems if group=='all' else {s for s in stems if s.startswith(group)}
            if not keep:
                rows.append(dict(split=split,group=group,movies=0,score=None,status='not_available'));continue
            current=data[(data.config=='appearance') & data.stem.isin(keep)];ref=base[base.stem.isin(keep)]
            previous=old[(old.config=='appearance') & old.stem.isin(keep)]
            assert len(current)==len(ref)==len(previous)==len(keep)
            a,b,c=[ns['aggregate_official'](v.to_dict('records')) for v in [current,ref,previous]]
            delta=current.set_index('stem').adjusted_edge_jaccard-ref.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(split=split,group=group,movies=len(keep),score=a['proxy_score'],delta_vs_C023=a['proxy_score']-b['proxy_score'],
                delta_vs_C038_cross_embryo=a['proxy_score']-c['proxy_score'],edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
                wins=int((delta>1e-10).sum()),losses=int((delta < -1e-10).sum()),div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False);data.to_csv(out/'official_per_movie.csv',index=False)
    prior=audit.MODES
    try:audit.MODES=['off','appearance'];audit.edge_audit(out,dict(jobs=jobs))
    finally:audit.MODES=prior
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,off_controls=97,exact_off_graphs=97,
        fold_proof=read(out/'fold_proof.json'),training_data=read(out/'smoke.json')['training_summary'],
        warning='Only2 biological groups;97 correlated/repeatedly examined crop movies. Frozen public detector/head not held out. Do not use diagnostic model routing at deployment.'))


def run(out):
    assert not (out/'status.json').exists(),'No blind restart; review completed artifacts and preserve original budget'
    plan=read(out/'plan.json');verify_hashes(plan['hashes'])
    for folder in ['train','models','logs','replay']:(out/folder).mkdir(exist_ok=True)
    q=Queue(out,8)
    try:
        q.state.update(plan_sha256=sha(out/'plan.json'),total_jobs=32);q.save()
        def task(name,command,*extra):q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*extra])
        for i in range(len(plan['extract_batches'])):task('extract_batch_'+str(i).zfill(2),'extract_batch','--batch',i)
        task('real_crop_smoke','smoke')
        for group in ['44b6','6bba']:task('train_'+group,'train','--group',group)
        task('verify_folds','verify_folds')
        for kind in ['reference','embryo_holdout']:
            for j in [r for r in plan['jobs'] if r['kind']==kind]:
                if kind=='embryo_holdout':verify_folds(out)
                q.run(j['name'],replay_command(out/(j['name']+'.ipynb'),Path(j['run']),j['stems'],out/'variants.json',out/'replay'/(j['name']+'.csv')))
            if kind=='reference':task('reference_parity','check_reference')
        task('analyse','analyse');verify_hashes(plan['hashes']);verify_hashes(read(out/'training_data_hashes.json'),out)
        artifacts={str(p.relative_to(out)):sha(p) for folder in ['train','models','graphs','replay'] for p in (out/folder).rglob('*') if p.is_file()}
        artifacts.update({p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.csv'] and p.name not in ['plan.json','status.json','artifact_hashes.json']})
        save_json(out/'artifact_hashes.json',artifacts);assert len(q.state['jobs'])==32;q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','extract_batch','smoke','train','verify_folds','check_reference','analyse'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--batch',type=int);p.add_argument('--group',choices=['44b6','6bba'])
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    if a.command=='extract_batch':extract_batch(out,a.batch)
    elif a.command=='train':train(out,a.group)
    elif a.command=='smoke':real_crop_smoke(out)
    else:globals()[a.command](out)
