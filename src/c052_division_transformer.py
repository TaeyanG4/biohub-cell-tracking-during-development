#!/usr/bin/env python3
"""C052: division-bearing production inputs, existing C037 learner and replay.

Only the training data sampler changes. Original C037 sources stay immutable.
No competition scoring implementation, deployment routing or threshold search.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import hashlib
import inspect
import io
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
import c037_transformer_study as old
import c038_followup_local as audit
from c047_hard_example_study import read, verify_hashes
from reid_probe_local import ROOT, BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue, replay_command
from eval_pp_variants_local import load_raw_graph

DEST = ROOT/'experiments/candidates/c052_division_transformer'
C037 = old.DEST
GROUPS = ['44b6', '6bba']
RECIPE = dict(old.RECIPE, checkpoints=[600], arms={'division600': [600, 1.]},
    sampling='odd steps division-context cluster cycle; even steps original24 uniform movie/window',
    ordinary_steps=300, division_steps=300, event_context_shape=[3,6,16,16],
    context_offset_voxels=[-1,0,1],
    split='train one whole embryo, evaluate ONLY the opposite embryo')


def ns_for():
    with contextlib.redirect_stdout(io.StringIO()):
        ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    return ns


def graph_path(run, stem):
    return next((run/'predictions').rglob(stem+'.geff'))


def control_run(stem):
    pilot=dict((s,sp) for sp,s in evaluation_plan())
    return CONTROL/('control_fp32_'+pilot.get(stem,'extension75'))


def event_audit(stems, ns):
    """Merge only exact, informative raw 3-frame contexts, including voxel jitter.

    Does not merge by reused GT IDs. Unresolved near-duplicates remain explicit.
    """
    events=[]; seen={}; parents=[]; evidence=[]
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    for stem in stems:
        gt,edges=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(stem+'.geff')))
        children=collections.defaultdict(list)
        for a,b in edges: children[a].append(b)
        vol=zarr.open_group(str(ROOT/'data/train'/(stem+'.zarr')),mode='r')['0']
        for p,cs in sorted(children.items()):
            if len(cs)!=2: continue
            t,z,y,x=map(lambda v:int(round(v)),gt[p]); i=len(events); parents.append(i)
            event=dict(index=i,stem=stem,embryo=stem[:4],parent=int(p),children=list(map(int,cs)),
                t=t,tzyx=[t,z,y,x],context_hashes=[],bounded_contexts=0)
            # Load a single raw parent-centred block; all jitter contexts are views.
            lo=np.array([t-1,z-4,y-9,x-9]); hi=np.array([t+2,z+4,y+9,x+9])
            if np.all(lo>=0) and np.all(hi<=np.array(vol.shape)):
                block=np.asarray(vol[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))])
                for dz,dy,dx in itertools.product(range(3),repeat=3):
                    raw=np.ascontiguousarray(block[:,dz:dz+6,dy:dy+16,dx:dx+16])
                    assert raw.shape==(3,6,16,16)
                    if np.std(raw)<1 or np.unique(raw).size<64: continue
                    digest=hashlib.sha256(str(raw.dtype).encode()+raw.tobytes()).hexdigest()
                    key=(stem[:4],digest); event['context_hashes'].append(digest)
                    if key in seen:
                        other=seen[key]
                        if root(i)!=root(other):
                            parents[root(i)]=root(other)
                            evidence.append(dict(a=other,b=i,sha256=digest))
                    else: seen[key]=i
                event['bounded_contexts']=len(event['context_hashes'])
            events.append(event)
    for e in events: e['cluster']=root(e['index'])
    return dict(events=events,exact_context_links=evidence,
        summary=[dict(embryo=g,annotations=sum(e['embryo']==g for e in events),
            exact_context_clusters=len({e['cluster'] for e in events if e['embryo']==g}),
            no_valid_context=sum(e['embryo']==g and not e['bounded_contexts'] for e in events)) for g in GROUPS],
        limitation='Exact local image identity supports shared-context grouping, not a complete biological event ID. Different contexts/near-duplicates may share an event. Only two embryos; no independent-event significance claim.')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'plan.json').exists(), 'Registered study already exists'
    verify_hashes(read(C037/'status.json')['source_hashes'])
    old_manifest=read(C037/'training_manifest.json')
    assert not any(r['division_targets'] for r in old_manifest['records'])
    coverage=read(ROOT/'state/split_audit_20260928/division_audit/coverage.json')
    division=sorted(r['stem'] for r in coverage['rows'] if r['positives']>0)
    ordinary=read(C037/'plan.json')['train_movies']
    assert len(division)==87 and len(ordinary)==24 and not set(division)&set(ordinary)
    for part in ['src','scripts']:
        shutil.copytree(C037/'tracking_repo'/part,out/'tracking_repo'/part,
            ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=False)
    for name,stems in [('division_stems',division)]+[(f'test_{g}',[s for _,s in evaluation_plan() if s.startswith(g)]) for g in GROUPS]:
        (out/(name+'.txt')).write_text('\n'.join(stems)+'\n',encoding='utf-8')
    save_json(out/'variants.json',{'as_configured':{}})
    event_info=event_audit(division,ns_for()); save_json(out/'event_audit.json',event_info)
    inputs={Path(__file__),out/'README.md',BASE,old.HEAD,old.PRIMARY_WEIGHTS,old.SECONDARY_WEIGHTS,
        C037/'training_manifest.json',C037/'plan.json',C037/'status.json',
        ROOT/'state/split_audit_20260928/division_audit/coverage.json'}
    inputs.update(ROOT/p for p in read(C037/'status.json')['source_hashes'])
    inputs.update(ROOT/'src'/p for p in ['c038_followup_local.py','c047_hard_example_study.py','reid_probe_local.py','reid_augmented_local.py'])
    inputs.update((out/'tracking_repo').rglob('*.py'))
    # Old packets and known-only labels are read in place, without changing them.
    for r in old_manifest['records']:
        for key,digest in [('file','packet_sha256'),('labels','label_sha256')]:
            p=C037/r[key]; assert sha(p)==r[digest]; inputs.add(p)
    pilot={s for _,s in evaluation_plan()}
    for stem in sorted(set(division)|pilot):
        for p in [ROOT/'data/train'/(stem+'.geff'), ROOT/'data/train'/(stem+'.zarr'),graph_path(control_run(stem),stem)]:
            inputs.update(f for f in p.rglob('*') if f.is_file())
        inputs.add(control_run(stem)/'edge_cache'/(stem+'.npz'))
    for split in ['heldout12','confirm10','extension75']: inputs.add(OLD/f'control_fp32_{split}.csv')
    inputs.update(C037/'replay'/f'late600_{g}.csv' for g in GROUPS)
    for p in out.iterdir():
        if p.is_file(): inputs.add(p)
    print(stamp(),'pinning',len(inputs),'inputs',flush=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs) if p.is_file()}
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,train_movies=division,
        ordinary_movies=ordinary,test_movies=sorted(pilot),hashes=hashes,
        total_jobs=14,estimated_minutes=150,max_start_job_hours=5,
        estimate_basis='87 capture movies at C037 1513sec/24 plus22 inference1423sec/replay~300sec, controls~300sec and verification margin',
        deployment=False,event_summary=event_info['summary']))
    print('Prepared C052',len(hashes),'pinned files;',event_info['summary'],flush=True)


def label(out):
    old.RECIPE=RECIPE
    old.label(out)
    new=read(out/'training_manifest.json')
    events=read(out/'event_audit.json')['events']
    by_window=collections.defaultdict(set)
    for e in events: by_window[(e['stem'],e['t'])].add(e['cluster'])
    records=[]
    for r in new['records']:
        if r['division_targets']<=0: continue
        with np.load(out/r['file']) as z: t=int(z['t_src'])
        clusters=sorted(by_window[(r['stem'],t)]); assert clusters
        r.update(pool='division',event_clusters=clusters); records.append(r)
    for r in read(C037/'training_manifest.json')['records']:
        r=dict(r,pool='ordinary',event_clusters=[])
        r['file']=str(C037/r['file']); r['labels']=str(C037/r['labels'])
        records.append(r)
    # Windows containing several events join their clusters, so no window is
    # oversampled just because it contains multiple annotated parents.
    for g in GROUPS:
        rows=[r for r in records if r['stem'].startswith(g) and r['pool']=='division']
        assert rows, ('no actual division packets',g)
        parent={c:c for r in rows for c in r['event_clusters']}
        def root(c):
            while parent[c]!=c: c=parent[c]
            return c
        for r in rows:
            for c in r['event_clusters'][1:]: parent[root(c)]=root(r['event_clusters'][0])
        for r in rows: r['sample_cluster']=str(root(r['event_clusters'][0]))
    save_json(out/'training_manifest.json',dict(new,recipe=RECIPE,records=records))
    summary=[]
    for g in GROUPS:
        rows=[r for r in records if r['stem'].startswith(g)]
        summary.append(dict(embryo=g,ordinary_windows=sum(r['pool']=='ordinary' for r in rows),
            division_windows=sum(r['pool']=='division' for r in rows),
            division_targets=sum(r['division_targets'] for r in rows),
            division_sampling_clusters=len({r['sample_cluster'] for r in rows if r['pool']=='division'})))
    save_json(out/'label_audit.json',dict(summary=summary,unknown_sources_are_negatives=False,
        known_parent_competition_minimum=2,matching='unchanged C037 official7um on rounded production coordinates',
        independence='Sampling clusters are not certified independent biological events. Whole-embryo isolation is the validation unit.'))
    print(json.dumps(summary),flush=True)


class FixedSampler:
    def __init__(self,out,group,records,rng):
        self.out,self.group,self.rng=out,group,rng
        self.ordinary=collections.defaultdict(list); self.division=collections.defaultdict(list)
        for r in records:
            (self.ordinary[r['stem']] if r['pool']=='ordinary' else self.division[r['sample_cluster']]).append(r)
        assert self.ordinary and self.division
        self.order=[];self.logs=[]
    def __call__(self,step):
        if step%2:
            if not self.order:self.order=list(self.rng.permutation(sorted(self.division)))
            key=str(self.order.pop()); pool=self.division[key]; kind='division'
        else:
            keys=sorted(self.ordinary);key=keys[int(self.rng.integers(len(keys)))];pool=self.ordinary[key];kind='ordinary'
        rec=pool[int(self.rng.integers(len(pool)))]; self.logs.append(dict(step=step,pool=kind,
            cluster=key,stem=rec['stem'],file=rec['file'],division_targets=rec['division_targets'],positive_targets=rec['positive_targets']))
        return rec
    def finish(self):
        assert len(self.logs)==600
        assert sum(r['pool']=='division' for r in self.logs)==300
        assert sum(r['pool']=='ordinary' for r in self.logs)==300
        assert {r['cluster'] for r in self.logs if r['pool']=='division'}==set(self.division)
        save_json(self.out/'models'/f'{self.group}.sampled_batches.json',dict(rows=self.logs,
            division_steps=300,ordinary_steps=300,division_clusters=len(self.division),all_clusters_sampled=True))


def train(out,group):
    """Retain the original objective/optimizer/forward, replace only sampling."""
    old.RECIPE=RECIPE
    code=inspect.getsource(old.train)
    anchor="    opt=torch.optim.AdamW(student.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])"
    assert code.count(anchor)==1
    code=code.replace(anchor,'    sampler=FixedSampler(out,group,records,rng)\n'+anchor)
    before="        movies=list(by_movie);movie=movies[int(rng.integers(len(movies)))];pool=by_movie[movie]\n        rec=pool[int(rng.integers(len(pool)))]"
    assert code.count(before)==1
    code=code.replace(before,"        movies=list(by_movie);rec=sampler(step);movie=rec['stem']")
    code+='\n    sampler.finish()\n'
    scope=dict(old.__dict__,FixedSampler=FixedSampler)
    exec(compile(code,str(Path(__file__))+'::original_train_with_sampler','exec'),scope)
    scope['train'](out,group)


def verify_folds(out):
    plan=read(out/'plan.json');rows=[]
    for train_group in GROUPS:
        test='6bba' if train_group=='44b6' else '44b6'
        path=out/'models'/f'{train_group}_step600.pt';meta=read(path.with_suffix('.json'))
        assert sha(path)==meta['model_sha256']
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        movies=set(ckpt['train_movies']);assert movies and all(s.startswith(train_group) for s in movies)
        eval_movies={s for s in plan['test_movies'] if s.startswith(test)}
        assert not movies&eval_movies and not any(s.startswith(test) for s in movies)
        batches=read(out/'models'/f'{train_group}.sampled_batches.json')
        assert batches['division_steps']==batches['ordinary_steps']==300 and batches['all_clusters_sampled']
        rows.append(dict(train_embryo=train_group,evaluate_embryo=test,train_movies=sorted(movies),
            test_movies=sorted(eval_movies),checkpoint_sha256=sha(path),division_steps=300,ordinary_steps=300))
    save_json(out/'fold_proof.json',dict(rows=rows,public_frozen_models_previously_saw_both_embryos=True,
        deployment_router=False))


def capture_check(out):
    ns=ns_for(); rows=[]
    for stem in read(out/'plan.json')['train_movies']:
        current=out/'e2e/capture_division87'; reference=control_run(stem)
        with np.load(current/'edge_cache'/(stem+'.npz')) as a,np.load(reference/'edge_cache'/(stem+'.npz')) as b:
            # Additional cached fused probabilities are diagnostic-only.
            for key in ['coords','low_coords','low_score','admitted']:
                assert np.array_equal(a[key],b[key],equal_nan=True),('detector control',stem,key)
        a,b=[load_raw_graph(ns,graph_path(r,stem)) for r in [current,reference]]
        assert a[0]==b[0]
        key=lambda e:(e['source_id'],e['target_id'])
        assert sorted(a[1],key=key)==sorted(b[1],key=key),('ILP off control',stem)
        packets=list((out/'features'/stem).glob('*.npz'));assert packets
        rows.append(dict(stem=stem,packets=len(packets),exact_detection_and_ilp=True))
    save_json(out/'capture_controls.json',dict(status='passed',rows=rows))


def install_stage_audit(ns,folder):
    """Passive GT-free capture around the existing executed notebook functions."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    active={}
    def capture(stage,nodes,edges):
        plain=ns['nodes_by_id_to_plain'](nodes);ids=sorted(plain)
        np.savez_compressed(folder/(active['stem']+'_'+stage+'.npz'),ids=np.array(ids,np.int64),
            txyz=np.array([[int(plain[i][0]),*[max(0,int(round(v))) for v in plain[i][1:]]] for i in ids],np.int64),
            edges=np.array(sorted((int(e['source_id']),int(e['target_id'])) for e in edges),np.int64).reshape(-1,2))
    original=ns['filter_output_graph']; motion=ns['motion_relink_edges']; unrestored=ns['_unrestored_filter_output_graph']
    def wrap_motion(nodes,*args,**kwargs):
        result=motion(nodes,*args,**kwargs);capture('motion',nodes,result or []);return result
    def wrap_unrestored(*args,**kwargs):
        result=unrestored(*args,**kwargs);capture('pre_restore',result[0],result[1]);return result
    def wrap_filter(nodes,edges,*args,**kwargs):
        active['stem']=kwargs['dataset'];capture('ilp',nodes,edges)
        result=original(nodes,edges,*args,**kwargs);capture('final',result[0],result[1]);return result
    ns.update(motion_relink_edges=wrap_motion,_unrestored_filter_output_graph=wrap_unrestored,filter_output_graph=wrap_filter)


def replay(out,name,stems,run):
    # The harness loads the notebook's own functions. Add a read-only callback.
    nb=read(BASE);s=''.join(nb['cells'][5]['source']);anchor='\nwrite_test_submission("base")\n'
    assert s.count(anchor)==1
    addition='\nfrom c052_division_transformer import install_stage_audit\n'+f'install_stage_audit(globals(), {str(out/"graphs"/name)!r})\n'
    nb['cells'][5]['source']=s.replace(anchor,addition+anchor).splitlines(keepends=True)
    path=out/(name+'.ipynb');path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    return replay_command(path,run,stems,out/'variants.json',out/'replay'/(name+'.csv'))


def analyse(out):
    verify_folds(out)
    ns=ns_for();baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{sp}.csv') for sp in ['heldout12','confirm10']])
    controls=pd.concat([pd.read_csv(out/'replay'/f'off_{sp}.csv') for sp in ['heldout12','confirm10']])
    audit.verify(controls,baseline,{'as_configured':'as_configured'})
    data=pd.concat([pd.read_csv(out/'replay'/f'division600_{g}.csv') for g in GROUPS])
    previous=pd.concat([pd.read_csv(C037/'replay'/f'late600_{g}.csv') for g in GROUPS])
    assert len(data)==22 and set(data.stem)==set(baseline.stem)
    data.to_csv(out/'official_per_movie.csv',index=False)
    rows=[]
    for name,members in [('all22',set(data.stem))]+[(sp,{s for p,s in evaluation_plan() if p==sp}) for sp in ['heldout12','confirm10']]+[(g,{s for s in data.stem if s.startswith(g)}) for g in GROUPS]:
        a,b,c=[ns['aggregate_official'](f[f.stem.isin(members)].to_dict('records')) for f in [data,baseline,previous]]
        delta=data.set_index('stem').loc[sorted(members),'adjusted_edge_jaccard']-baseline.set_index('stem').loc[sorted(members),'adjusted_edge_jaccard']
        rows.append(dict(group=name,movies=len(members),score=a['proxy_score'],delta_vs_C023=a['proxy_score']-b['proxy_score'],
            delta_vs_C037_late600=a['proxy_score']-c['proxy_score'],edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
            edge_delta_vs_C037=a['adjusted_edge_jaccard']-c['adjusted_edge_jaccard'],
            div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn'],edge_wins=int((delta>1e-10).sum()),edge_losses=int((delta< -1e-10).sum())))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    changes=[]
    for split,stem in evaluation_plan():
        old_folder=out/'graphs'/('off_'+split);new_folder=out/'graphs'/('division600_'+stem[:4])
        # Require exact final off graph identity with the historical C038 capture.
        ref=audit.DEST/'graphs'/('pilot_'+split+'_'+stem[:4])
        assert audit.graph(old_folder,stem,'final')==audit.graph(ref,stem,'off'),('final off graph',stem)
        for stage in ['ilp','motion','pre_restore','final']:
            a,ae=audit.graph(old_folder,stem,stage);b,be=audit.graph(new_folder,stem,stage)
            # Coordinate endpoints allow synthetic ID differences; preserve multiplicity.
            ac=collections.Counter((a[s],a[t]) for s,t in ae);bc=collections.Counter((b[s],b[t]) for s,t in be)
            changes.append(dict(stem=stem,stage=stage,removed=sum((ac-bc).values()),added=sum((bc-ac).values()),
                before_nodes=len(a),after_nodes=len(b),before_forks=sum(v==2 for v in collections.Counter(s for s,t in ae).values()),
                after_forks=sum(v==2 for v in collections.Counter(s for s,t in be).values())))
    pd.DataFrame(changes).to_csv(out/'stage_graph_changes.csv',index=False)
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,off_controls=22,exact_off_graphs=22,
        capture_controls=87,actual_training=read(out/'label_audit.json'),folds=read(out/'fold_proof.json'),
        note='Two embryos and correlated movies. Frozen public detector/head saw both. Small consistent gains need review/75 extension; no automatic promotion. Primary packets and fused edge caches retained for division survival audit before deployment.'))


def run(out):
    assert not (out/'status.json').exists(),'No blind restart of registered queue'
    plan=read(out/'plan.json');q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(out/'plan.json'));q.save()
        verify_hashes(plan['hashes'])
        def task(name,command,*args): q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*args])
        task('real_equivalence_smoke','smoke')
        q.run('capture_division87',old.inference(out,'capture_division87',out/'division_stems.txt',[
            f'BIOHUB_C037_CAPTURE_DIR={out/"features"}','BIOHUB_C037_CHECKPOINT=','BIOHUB_CACHE_EDGE_THRESHOLD=0.02']))
        task('capture_controls87','capture_check');task('label_production_inputs','label')
        for g in GROUPS:task('train_'+g,'train','--group',g)
        task('whole_embryo_fold_proof','verify_folds')
        for sp in ['heldout12','confirm10']:
            stems=[s for split,s in evaluation_plan() if split==sp]
            q.run('off_'+sp,replay(out,'off_'+sp,stems,CONTROL/('control_fp32_'+sp)))
        for test in GROUPS:
            train_group='6bba' if test=='44b6' else '44b6';name='division600_'+test
            verify_folds(out)
            q.run('infer_'+test,old.inference(out,name,out/f'test_{test}.txt',[
                f'BIOHUB_C037_CHECKPOINT={out/"models"/(train_group+"_step600.pt")}',
                'BIOHUB_C037_ALPHA=1.0',f'BIOHUB_C037_CAPTURE_DIR={out/"evaluation_features"}',
                'BIOHUB_CACHE_EDGE_THRESHOLD=0.02']))
            q.run('replay_'+test,replay(out,name,[s for _,s in evaluation_plan() if s.startswith(test)],out/'e2e'/name))
        task('analyse','analyse');verify_hashes(plan['hashes'])
        files=[p for folder in ['features','labels','models','graphs','replay','evaluation_features','e2e'] for p in (out/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.ipynb'] and p.name not in ['plan.json','status.json','launch.json']]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        assert len(q.state['jobs'])==plan['total_jobs'];q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','smoke','label','train','verify_folds','capture_check','analyse'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--group',choices=GROUPS)
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    if a.command=='train':train(out,a.group)
    elif a.command=='smoke':old.smoke(out)
    else:globals()[a.command](out)
