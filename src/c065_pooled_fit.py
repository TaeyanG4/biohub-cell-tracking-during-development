"""One production refit of the supported C052 recipe, with unchanged objective."""
from pathlib import Path
import argparse
import collections
import copy
import json
import os
import shutil
import sys
import numpy as np
import torch
import c052_division_transformer as pilot
import c037_transformer_study as old
from c037_transformer_runtime import state_digest,apply_primary
from reid_probe_local import ROOT,BASE,sha,save_json,stamp
from c047_hard_example_study import read,verify_hashes
from run_last_days_local import Queue

DEST=ROOT/'experiments/candidates/c065_pooled_transformer'
FIT=DEST/'fit'

def prepare():
    assert not (FIT/'plan.json').exists(),'Pinned phase is immutable'
    FIT.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'state/validation_audit_20260929/legacy/independent_verification.json')['status']=='passed'
    source=read(pilot.DEST/'training_manifest.json')
    records=copy.deepcopy(source['records']); inputs={Path(__file__),DEST/'README.md',FIT/'launch.ps1',
        pilot.DEST/'training_manifest.json',pilot.DEST/'label_audit.json',pilot.DEST/'event_audit.json',
        ROOT/'src/c037_transformer_study.py',ROOT/'src/c052_division_transformer.py',
        ROOT/'src/c037_transformer_runtime.py',ROOT/'src/run_last_days_local.py',
        ROOT/'src/c047_hard_example_study.py',ROOT/'src/reid_probe_local.py',
        ROOT/'tools/notify_background_completion.ps1',BASE,old.PRIMARY_WEIGHTS,old.HEAD,
        ROOT/'state/validation_audit_20260929/actual_summary.csv',
        ROOT/'state/validation_audit_20260929/legacy/actual_summary.csv'}
    for rec in records:
        for key,digest in [('file','packet_sha256'),('labels','label_sha256')]:
            p=(pilot.DEST/rec[key]).resolve()
            p.relative_to(ROOT);assert sha(p)==rec[digest],p
            rec[key]=str(p);inputs.add(p)
    # Original cluster IDs must not accidentally conflate embryo pools.
    domains=collections.defaultdict(set)
    for r in records:
        if r['pool']=='division': domains[r['sample_cluster']].add(r['stem'][:4])
    assert all(len(x)==1 for x in domains.values())
    assert len(domains)==144
    manifest=dict(source,records=records,deployment_fit=True,
        validation='pooled fit-domain; C052 opposite folds supply transfer evidence; no new independent test claim')
    save_json(FIT/'training_manifest.json',manifest)
    for part in ['scripts','src']:
        shutil.copytree(pilot.DEST/'tracking_repo'/part,FIT/'tracking_repo'/part,
            ignore=shutil.ignore_patterns('__pycache__'))
    sampler=pilot.FixedSampler(FIT,'pooled',records,np.random.default_rng(pilot.RECIPE['seed']))
    for step in range(1,601):sampler(step)
    assert collections.Counter(r['pool'] for r in sampler.logs)==dict(division=300,ordinary=300)
    assert {r['cluster'] for r in sampler.logs if r['pool']=='division'}==set(domains)
    save_json(FIT/'expected_batches.json',sampler.logs)
    coverage=[dict(embryo=g,pool=p,windows=sum(r['stem'].startswith(g) and r['pool']==p for r in records),
        movies=len({r['stem'] for r in records if r['stem'].startswith(g) and r['pool']==p}),
        actual_planned_steps=sum(r['stem'].startswith(g) and r['pool']==p for r in sampler.logs))
        for g in pilot.GROUPS for p in ['ordinary','division']]
    save_json(FIT/'preflight.json',dict(status='passed',records=len(records),movies=len({r['stem'] for r in records}),
        division_clusters=len(domains),coverage=coverage,source_packets_labels_exact=True,
        unchanged_learner='actual c052.train(FIT,pooled), existing group=pooled support in c037',
        no_averaged_initialization=True,no_unknown_negative_labels=True))
    inputs.update(p for p in FIT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}
    save_json(FIT/'plan.json',dict(created=stamp(),recipe=pilot.RECIPE,hashes=hashes,total_jobs=2,
        estimated_minutes=3,estimate_basis='C052 same600step fits took27/31seconds; add two full hash passes and actual runtime reload controls',
        train_movies=sorted({r['stem'] for r in records}),new_fit=True,no_kaggle_writes=True,
        user_scope='Revisit prior validation and prioritize improvement; finish including submission before2026-09-30T00:00:00+09:00',
        decision='One fixed direct production refit, not checkpoint/epoch/strength sweep'))
    print(json.dumps(dict(inputs=len(hashes),coverage=coverage,phase='prepared')),flush=True)

def train():
    verify_hashes(read(FIT/'plan.json')['hashes'])
    pilot.train(FIT,'pooled')

def verify():
    plan=read(FIT/'plan.json');verify_hashes(plan['hashes'])
    trace=read(FIT/'models/pooled.sampled_batches.json')
    assert trace['rows']==read(FIT/'expected_batches.json')
    assert trace['division_steps']==trace['ordinary_steps']==300 and trace['all_clusters_sampled']
    path=FIT/'models/pooled_step600.pt';meta=read(path.with_suffix('.json'))
    assert sha(path)==meta['model_sha256']
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['step']==600 and checkpoint['train_embryo']=='pooled'
    assert checkpoint['recipe']==pilot.RECIPE
    assert sorted(checkpoint['train_movies'])==plan['train_movies']
    assert meta['base_transformer_unchanged']
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    pu=old.engine(FIT)
    full,_,_=pu.load_model(old.PRIMARY_WEIGHTS,torch.device('cuda'));full.eval()
    assert state_digest(full.transformer.state_dict())==checkpoint['base_transformer_sha256']
    frozen_hash=state_digest({k:v for k,v in full.state_dict().items() if not k.startswith('transformer.')})
    direct=copy.deepcopy(full.transformer);direct.load_state_dict(checkpoint['state_dict'],strict=True);direct.eval()
    os.environ['BIOHUB_C037_CHECKPOINT']=str(path);os.environ['BIOHUB_C037_ALPHA']='1.0'
    apply_primary(full)
    assert state_digest(full.transformer.state_dict())==state_digest(checkpoint['state_dict'])
    assert frozen_hash==state_digest({k:v for k,v in full.state_dict().items() if not k.startswith('transformer.')})
    records=read(FIT/'training_manifest.json')['records'];proof=[]
    for group in pilot.GROUPS:
        for pool in ['ordinary','division']:
            rec=next(r for r in records if r['stem'].startswith(group) and r['pool']==pool)
            with np.load(rec['file']) as z:packet={k:z[k].copy() for k in z.files}
            with torch.inference_mode():a=old.forward(direct,packet);b=old.forward(full.transformer,packet)
            assert torch.equal(a,b)
            proof.append(dict(stem=rec['stem'],pool=pool,packet_sha256=rec['packet_sha256'],logit_max_error=float((a-b).abs().max())))
    save_json(FIT/'fit_verification.json',dict(status='passed',final_steps=600,actual_sample_trace_exact=True,
        source_movies=len(plan['train_movies']),all144_clusters_sampled=True,teacher_frozen=True,
        runtime_hook_matches_direct_reload=True,frozen_nontransformer_unchanged=True,real_packet_checks=proof,
        model_sha256=sha(path),production_single_checkpoint=True,
        no_graph_score_or_generalization_claim=True))
    verify_hashes(plan['hashes']);print(json.dumps(read(FIT/'fit_verification.json')),flush=True)

def run():
    assert not (FIT/'status.json').exists(),'No blind rerun'
    plan=read(FIT/'plan.json');q=Queue(FIT,1)
    try:
        q.state.update(total_jobs=2,plan_sha256=sha(FIT/'plan.json'));q.save();verify_hashes(plan['hashes'])
        for stage in ['train','verify']:q.run(stage,[sys.executable,'-u',Path(__file__),stage])
        verify_hashes(plan['hashes'])
        outputs=list((FIT/'models').glob('*'))+[FIT/'fit_verification.json']+list((FIT/'logs').glob('*'))
        save_json(FIT/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in outputs})
        q.close('complete_review_required')
    except BaseException as e:q.close('failed',e);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','verify','run'])
    globals()[p.parse_args().command]()
