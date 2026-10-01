#!/usr/bin/env python3
"""Finite data-coverage experiment using C035 training and the existing replay/Queue."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import reid_augmented_local as aug
import c038_followup_local as audit
from c041_fixed_models import make_notebook
from reid_probe_local import ROOT, BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from run_last_days_local import Queue, replay_command

DEST = ROOT/'experiments/candidates/c047_hard_example_appearance'
C035 = ROOT/'experiments/candidates/c035_augmented_reid'
C041 = ROOT/'experiments/candidates/c041_fixed_models'
C046 = ROOT/'experiments/candidates/c046_fixed_appearance_extension'
FEASIBILITY = ROOT/'state/improvement_feasibility_20260928'
POLICY = dict(min_hard_links=10, hard_separation_um=[6., 12.], max_new_triplets_per_movie=192,
              selection='all previously unused eligible movies; evenly spaced ordered temporal triplets',
              existing24='byte-identical C035 training crops and triplets',
              training='C035 weak_aug function unchanged,1200 steps per embryo,seed3501,equal movie sampling',
              deployment='fixed mean cosine of BOTH encoders on EVERY movie; no embryo routing',
              changed_factor='training data coverage/sampling only; no optimizer,checkpoint,threshold sweep')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_hashes(hashes, root=ROOT):
    for name, digest in hashes.items():
        assert sha(root/name) == digest, ('input drift', name)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'plan.json').exists(), 'Registered plan exists'
    old = read(C035/'plan.json')
    inventory = {r['stem']: r for r in old['inventory']}
    eligible = [r for r in read(FEASIBILITY/'hard_negative_inventory.json')['rows']
                if not r['old_selected'] and r['hard_6_12'] >= POLICY['min_hard_links']]
    added = [dict(inventory[r['stem']], hard_links=r['hard_6_12'], kind='new_hard')
             for r in sorted(eligible, key=lambda r:r['stem'])]
    selected = [dict(r, kind='existing') for r in old['selected']] + added
    official = set(old['excluded_official97'])
    assert len(official) == 97 and len(selected) == 53 and len(added) == 29
    assert len({r['stem'] for r in selected}) == 53
    assert not official & {r['stem'] for r in selected}
    prior = read(C046/'plan.json'); verify_hashes(prior['hashes'])
    gt_hashes = read(FEASIBILITY/'gt_input_hashes.json'); verify_hashes(gt_hashes)
    hashes = dict(prior['hashes']); hashes.update(gt_hashes)
    inputs = {Path(__file__), out/'README.md', C035/'plan.json', C035/'status.json'}
    inputs.update(FEASIBILITY.glob('*.json'))
    inputs.update(ROOT/'src'/name for name in [
        'reid_augmented_local.py','reid_probe_local.py','division_cnn_train.py',
        'division_crops_extract.py','c041_fixed_models.py','c038_followup_local.py',
        'run_last_days_local.py','eval_pp_variants_local.py'])
    old_status = read(C035/'status.json')
    assert old_status['status'] == 'complete' and old_status['plan_sha256'] == sha(C035/'plan.json')
    verify_hashes(old_status['source_hashes'])
    old_artifacts = {name:digest for job in old_status['jobs'].values() for name,digest in job['artifacts'].items()}
    for item in selected:
        for name, digest in item['image_metadata'].items():
            assert sha(ROOT/name) == digest
            hashes[name] = digest
        if item['kind'] == 'existing':
            for suffix in ['.npz','.json']:
                path = C035/'train'/(item['stem']+suffix)
                key = str(path.relative_to(ROOT))
                assert sha(path) == old_artifacts[key], ('C035 crop drift', key)
                inputs.add(path)
    source = (C041/'portable_appearance.py').read_text(encoding='utf-8')
    save_json(out/'variants.json', {'off': {'C038_MODE':'off'}, 'appearance': {'C038_MODE':'appearance'}})
    inputs.add(out/'variants.json')
    jobs = []
    for group in ['44b6','6bba']:
        stem = next(s for _, s in aug.evaluation_plan() if s.startswith(group))
        name = 'reference_'+group
        inputs.add(make_notebook(out,name,C041/'appearance_mean_cosine.pt',source))
        jobs.append(dict(name=name,kind='reference',split='heldout12',stems=[stem],
                         run=str(CONTROL/'control_fp32_heldout12')))
    for split in ['heldout12','confirm10']:
        name = 'hard_'+split
        inputs.add(make_notebook(out,name,out/'appearance_mean_cosine.pt',source))
        jobs.append(dict(name=name,kind='hard',split=split,
                         stems=[s for sp,s in aug.evaluation_plan() if sp==split],
                         run=str(CONTROL/('control_fp32_'+split))))
        inputs.add(C046/'replay'/(split+'.csv'))
        inputs.update(p for p in (C046/'graphs'/split).iterdir() if p.is_file())
    for path in inputs:
        assert path.is_file(), path
        hashes[str(path.relative_to(ROOT))] = sha(path)
    save_json(out/'plan.json',dict(created=stamp(),policy=POLICY,recipe=aug.RECIPE,
        selected=selected,excluded_official97=sorted(official),jobs=jobs,hashes=hashes,
        total_jobs=40,max_start_job_hours=8,deployment=False,
        intended_new_hard_triplets=sum(min(192,r['hard_links']) for r in added)))
    print('Prepared40 finite jobs;53 train movies;29 new;1881 additional hard triplets;',len(hashes),'inputs',flush=True)


def copy_existing(out):
    plan=read(out/'plan.json')
    for item in plan['selected']:
        if item['kind'] != 'existing': continue
        for suffix in ['.npz','.json']:
            src=C035/'train'/(item['stem']+suffix); dst=out/'train'/src.name
            assert sha(src)==plan['hashes'][str(src.relative_to(ROOT))]
            shutil.copyfile(src,dst); assert sha(dst)==sha(src)
    print('Copied24 original training caches exactly',flush=True)


def extract(out, stem):
    plan=read(out/'plan.json'); item=next(r for r in plan['selected'] if r['stem']==stem)
    assert item['kind']=='new_hard'
    with contextlib.redirect_stdout(io.StringIO()): ns,frames,_=setup_ns(CONTROL/'control_fp32_heldout12')
    nodes,triples,distances,digest=aug.gt_triplets(ns,stem)
    assert digest==item['graph_sha256'] and aug.data_meta(stem)==item['image_metadata']
    keep=np.flatnonzero((distances>=6.) & (distances<=12.))
    assert len(keep)==item['hard_links'] and len(keep)>=10
    keep=sorted(keep,key=lambda i:(nodes[int(triples[i,0])][0],*map(int,triples[i])))
    if len(keep)>192:
        keep=np.asarray(keep)[np.linspace(0,len(keep)-1,192,dtype=int)]
    triples,distances=triples[keep],distances[keep]
    ids=np.unique(triples); crops,coords=aug.patches(ns,frames,stem,ids,nodes)
    target=out/'train'/(stem+'.npz')
    np.savez_compressed(target,crops=crops,triplets=np.searchsorted(ids,triples),ids=ids,
                        tzyx=coords,negative_distance_um=distances.astype(np.float32))
    save_json(out/'train'/(stem+'.json'),dict(stem=stem,kind='new_hard',graph_sha256=digest,
        image_metadata=item['image_metadata'],patches=len(ids),triplets=len(triples),
        negatives_within_12um=len(triples),frames=len(set(coords[:,0])),crop_sha256=sha(target),
        selection=POLICY['selection'],unknown_used_as_negative=False))
    print(stem,'hard triplets',len(triples),'patches',len(ids),flush=True)


def smoke(out):
    plan=read(out/'plan.json'); runtime=aug.runtime_policy(); rows=[]
    torch.manual_seed(aug.RECIPE['seed'])
    for group in ['44b6','6bba']:
        item=next(r for r in plan['selected'] if r['kind']=='new_hard' and r['stem'].startswith(group))
        with np.load(out/'train'/(item['stem']+'.npz')) as d:
            batch=torch.from_numpy(d['crops'][d['triplets'][:2].T.reshape(-1)].astype(np.float32))[:,None].cuda()
        model=aug.AppearanceEncoder().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=aug.RECIPE['lr'])
        before=model.head.weight.detach().clone()
        loss,_=aug.objective(model,aug.augmentation(batch.clone(),'weak_aug'))
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step(); assert not torch.equal(before,model.head.weight)
        model.cpu().eval(); clone=aug.AppearanceEncoder().eval(); clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            v=model(batch.cpu()); assert torch.equal(v,clone(batch.cpu()))
            assert torch.allclose(v.norm(dim=1),torch.ones(len(v)),atol=1e-5)
        rows.append(dict(group=group,hard_crop_gradient_update_reload=True,loss=float(loss)))
    hashes={str(p.relative_to(out)):sha(p) for p in (out/'train').iterdir() if p.is_file()}
    summary=[]
    for group in ['44b6','6bba']:
        distances=[]; count=0
        for item in plan['selected']:
            if item['stem'].startswith(group):
                count+=1
                with np.load(out/'train'/(item['stem']+'.npz')) as d: distances.extend(d['negative_distance_um'])
        distances=np.asarray(distances)
        summary.append(dict(group=group,movies=count,triplets=len(distances),hard=int((distances<=12.).sum()),
                            negative_distance_median=float(np.median(distances))))
    save_json(out/'training_data_hashes.json',hashes)
    save_json(out/'smoke.json',dict(status='passed',runtime=runtime,checks=rows,training_summary=summary))


def train(out, group):
    verify_hashes(read(out/'training_data_hashes.json'),out)
    aug.runtime_policy(); plan=read(out/'plan.json')
    result=aug.train(out,plan,group,'weak_aug',lambda s: print(stamp(),s,flush=True))
    path=out/'models'/(group+'_weak_aug.pt'); state=torch.load(path,map_location='cpu',weights_only=True)
    state['training_policy']=POLICY; state['c047_plan_sha256']=sha(out/'plan.json')
    state['training_data_manifest_sha256']=sha(out/'training_data_hashes.json'); torch.save(state,path)
    result.update(model_sha256=sha(path),training_policy=POLICY)
    save_json(path.with_suffix('.json'),result)


def pack(out):
    plan=read(out/'plan.json'); official=set(plan['excluded_official97'])
    spec=importlib.util.spec_from_file_location('c047_fixed_runtime',C041/'portable_appearance.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    states={}; training=set(); models=[]; component_hashes={}
    for i,group in enumerate(['44b6','6bba']):
        path=out/'models'/(group+'_weak_aug.pt'); state=torch.load(path,map_location='cpu',weights_only=True)
        assert sha(path)==read(path.with_suffix('.json'))['model_sha256']
        assert state['recipe']==aug.RECIPE and state['training_policy']==POLICY
        expected={r['stem'] for r in plan['selected'] if r['stem'].startswith(group)}
        assert set(state['train_movies'])==expected
        training.update(expected); models.append(module.load_appearance(path)); component_hashes[group]=sha(path)
        states.update({f'models.{i}.{k}':v for k,v in state['state_dict'].items()})
    assert len(training)==53 and not training & official
    path=out/'appearance_mean_cosine.pt'
    torch.save(dict(state_dict=states,n_models=2,recipe=aug.RECIPE,arm='weak_aug',
        train_movies=sorted(training),training_policy=POLICY,component_hashes=component_hashes),path)
    torch.set_num_threads(4); torch.manual_seed(4101); x=torch.randn(4,1,8,32,32)
    with torch.no_grad():
        actual=module.load_appearance(path)(x); expected=sum(m(x)@m(x).T for m in models)/2
        error=float((actual@actual.T-expected).abs().max()); assert error<1e-6
    assert sha(path)!=sha(C041/'appearance_mean_cosine.pt')
    save_json(out/'fixed_model.json',dict(status='passed',model_sha256=sha(path),component_hashes=component_hashes,
        runtime_sha256=sha(C041/'portable_appearance.py'),mean_cosine_max_error=error,
        training_movies=53,no_training_overlap_official97=True,no_prefix_routing=True))


def check_reference(out):
    for job in read(out/'plan.json')['jobs']:
        if job['kind']!='reference': continue
        frame=pd.read_csv(out/'replay'/(job['name']+'.csv'))
        ref=pd.read_csv(C046/'replay'/(job['split']+'.csv')); ref=ref[ref.stem.isin(job['stems'])]
        audit.verify(frame,ref,{'off':'off','appearance':'appearance'})
        for stem in job['stems']:
            for mode in ['off','appearance']:
                assert audit.graph(out/'graphs'/job['name'],stem,mode)==audit.graph(C046/'graphs'/job['split'],stem,mode)
    save_json(out/'reference_parity.json',dict(status='passed',full_movies=2,exact_graphs=4,exact_metric_rows=4))


def analyse(out):
    plan=read(out/'plan.json'); check_reference(out)
    jobs=[j for j in plan['jobs'] if j['kind']=='hard']
    data=pd.concat([pd.read_csv(out/'replay'/(j['name']+'.csv')) for j in jobs],ignore_index=True)
    base=pd.concat([pd.read_csv(OLD/('control_fp32_'+s+'.csv')) for s in ['heldout12','confirm10']])
    previous=pd.concat([pd.read_csv(C046/'replay'/(s+'.csv')) for s in ['heldout12','confirm10']])
    assert data.stem.nunique()==22
    audit.verify(data,base,{'off':'as_configured'})
    for j in jobs:
        for stem in j['stems']:
            assert audit.graph(out/'graphs'/j['name'],stem,'off')==audit.graph(C046/'graphs'/j['split'],stem,'off')
    with contextlib.redirect_stdout(io.StringIO()): ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    rows=[]
    for group in ['all22','heldout12','confirm10','44b6','6bba']:
        stems={s for split,s in aug.evaluation_plan() if group=='all22' or split==group or s.startswith(group)}
        cur=data[(data.config=='appearance') & data.stem.isin(stems)]
        original=base[base.stem.isin(stems)]; old=previous[(previous.config=='appearance') & previous.stem.isin(stems)]
        assert len(cur)==len(original)==len(old)==len(stems)>0
        a,b,c=[ns['aggregate_official'](f.to_dict('records')) for f in [cur,original,old]]
        deltas=cur.set_index('stem').adjusted_edge_jaccard-original.set_index('stem').adjusted_edge_jaccard
        rows.append(dict(group=group,movies=len(stems),score=a['proxy_score'],delta_vs_C023=a['proxy_score']-b['proxy_score'],
            delta_vs_C046=a['proxy_score']-c['proxy_score'],edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
            wins=int((deltas>1e-10).sum()),losses=int((deltas < -1e-10).sum()),
            div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    data.to_csv(out/'official_per_movie.csv',index=False)
    before=audit.MODES
    try: audit.MODES=['off','appearance']; audit.edge_audit(out,dict(jobs=jobs))
    finally: audit.MODES=before
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,
        off_controls=22,exact_off_graphs=22,training_data=read(out/'smoke.json')['training_summary'],
        note='Data-only changed factor. Fixed models on every movie;22 reused retrospective evaluation movies,not independent validation. Review signed harms and graph effects before extension or portable/T4 work.'))


def run(out):
    assert not (out/'status.json').exists(), 'No blind restart; review failure and completed artifacts first'
    plan=read(out/'plan.json'); verify_hashes(plan['hashes'])
    for folder in ['train','models','logs','replay']: (out/folder).mkdir(exist_ok=True)
    q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(plan_sha256=sha(out/'plan.json'),total_jobs=plan['total_jobs']);q.save()
        def task(name,command,*extra):
            q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*extra])
        task('copy_existing24','copy_existing')
        for item in plan['selected']:
            if item['kind']=='new_hard': task('extract_'+item['stem'],'extract','--stem',item['stem'])
        task('hard_crop_smoke','smoke')
        for group in ['44b6','6bba']: task('train_'+group,'train','--group',group)
        task('fixed_model_pack','pack')
        for kind in ['reference','hard']:
            for j in [r for r in plan['jobs'] if r['kind']==kind]:
                assert sha(out/'appearance_mean_cosine.pt')==read(out/'fixed_model.json')['model_sha256']
                q.run(j['name'],replay_command(out/(j['name']+'.ipynb'),Path(j['run']),j['stems'],out/'variants.json',out/'replay'/(j['name']+'.csv')))
            if kind=='reference': task('reference_parity','check_reference')
        task('analyse','analyse')
        verify_hashes(plan['hashes']); verify_hashes(read(out/'training_data_hashes.json'),out)
        artifacts={str(p.relative_to(out)):sha(p) for folder in ['train','models','graphs','replay']
                   for p in (out/folder).rglob('*') if p.is_file()}
        artifacts.update({p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.pt','.json','.csv']
                          and p.name not in ['plan.json','status.json','artifact_hashes.json']})
        save_json(out/'artifact_hashes.json',artifacts)
        assert len(q.state['jobs'])==plan['total_jobs']
        q.close('complete_review_required')
    except BaseException as exc: q.close('failed',exc); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','copy_existing','extract','smoke','train','pack','check_reference','analyse'])
    parser.add_argument('--out',type=Path,default=DEST);parser.add_argument('--stem');parser.add_argument('--group',choices=['44b6','6bba'])
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    if args.command=='extract': extract(out,args.stem)
    elif args.command=='train': train(out,args.group)
    else: globals()[args.command](out)
