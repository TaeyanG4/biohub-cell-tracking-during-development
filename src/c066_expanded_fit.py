"""Expand only C052 ordinary packets; preserve learner and final-step protocol."""
from pathlib import Path
import argparse,collections,copy,json,os,shutil,sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import c052_division_transformer as pilot
import c037_transformer_study as old
from c037_transformer_runtime import state_digest,apply_primary
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,BASE,sha,save_json,stamp
from run_last_days_local import Queue

DEST=ROOT/'experiments/candidates/c066_expanded_ordinary'
FIT=DEST/'fit'
GROUPS=['44b6','6bba','pooled']

def selected(records,group):return [r for r in records if group=='pooled' or r['stem'].startswith(group)]

def prepare():
    assert not (FIT/'plan.json').exists()
    FIT.mkdir(parents=True,exist_ok=True)
    source=read(pilot.DEST/'training_manifest.json');records=copy.deepcopy(source['records'])
    for r in records:
        for key in ['file','labels']:r[key]=str((pilot.DEST/r[key]).resolve())
    coverage=ROOT/'state/c066_coverage';inventory=pd.read_csv(coverage/'additional_packet_inventory.csv')
    added=inventory[inventory.additional_ordinary]
    assert len(added)==7791
    for r in added.to_dict('records'):
        assert r['division_targets']==0
        records.append({**{key:r[key] for key in ['stem','file','labels','packet_sha256','label_sha256','source_nodes','target_nodes','known_sources','positive_targets','division_targets']},
                        'pool':'ordinary','event_clusters':[]})
    assert len(records)==10182 and len({r['stem'] for r in records})==111
    assert len({r['file'] for r in records})==len(records)
    inputs={Path(__file__),DEST/'README.md',FIT/'launch.ps1',coverage/'additional_packet_inventory.csv',coverage/'summary.json',
        pilot.DEST/'training_manifest.json',pilot.DEST/'artifact_hashes.json',ROOT/'src/c052_division_transformer.py',
        ROOT/'src/c037_transformer_study.py',ROOT/'src/c037_transformer_runtime.py',ROOT/'src/run_last_days_local.py',
        ROOT/'src/c047_hard_example_study.py',ROOT/'src/reid_probe_local.py',ROOT/'tools/notify_background_completion.ps1',BASE,old.HEAD,old.PRIMARY_WEIGHTS}
    hashes={}
    for r in records:
        for key,digest in [('file','packet_sha256'),('labels','label_sha256')]:
            p=Path(r[key]);p.relative_to(ROOT);actual=sha(p);assert actual==r[digest],p
            hashes[str(p.relative_to(ROOT))]=actual
        with np.load(r['labels']) as z:
            assert len(z['targets'])==r['positive_targets'] and int((z['weight']>1).sum())==r['division_targets']
            assert z['parent_index'].min()>=0 and z['parent_index'].max()<len(z['known_src'])
            if r['pool']=='ordinary':assert (z['weight']==1).all()
    save_json(FIT/'training_manifest.json',dict(source,records=records,ordinary_pool='all retained ordinary packets from24+87movies',
        unknown_sources_are_negatives=False,new_data_capture=False))
    for part in ['scripts','src']:
        shutil.copytree(pilot.DEST/'tracking_repo'/part,FIT/'tracking_repo'/part,ignore=shutil.ignore_patterns('__pycache__'))
    summaries=[]
    for group in GROUPS:
        eligible=selected(records,group)
        sampler=pilot.FixedSampler(FIT,group,eligible,np.random.default_rng(3701))
        for step in range(1,601):sampler(step)
        expected_clusters={r['sample_cluster'] for r in eligible if r['pool']=='division'}
        assert {r['cluster'] for r in sampler.logs if r['pool']=='division'}==expected_clusters
        assert collections.Counter(r['pool'] for r in sampler.logs)==dict(division=300,ordinary=300)
        save_json(FIT/f'expected_{group}.json',sampler.logs)
        summaries.append(dict(group=group,eligible_movies=len({r['stem'] for r in eligible}),windows=len(eligible),
            division_clusters=len(expected_clusters),actual_planned_movies=len({r['stem'] for r in sampler.logs})))
    save_json(FIT/'preflight.json',dict(status='passed',summaries=summaries,records=len(records),
        original_packets_preserved=2391,additional_ordinary_packets=7791,known_label_algebra_passed=True,
        whole_embryo_filter_for_transfer=True,pooled_checkpoint_is_not_validation=True))
    inputs.update(p for p in FIT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)})
    save_json(FIT/'plan.json',dict(created=stamp(),hashes=hashes,recipe=pilot.RECIPE,groups=GROUPS,
        total_jobs=6,estimated_minutes=6,train_movies=sorted({r['stem'] for r in records}),
        estimated_remaining_after_fit_minutes=335,completion_target='2026-09-30T00:00:00+09:00',
        no_kaggle_writes=True,scientific_change='Only add7791existing ordinary windows; unchanged learner/division pool/600steps',
        estimate_basis='ThreeC052stylefits plus hashes/reload controls; subsequent transfer97 and production97 must fit midnight budget'))
    print(json.dumps(dict(prepared=True,inputs=len(hashes),summaries=summaries)),flush=True)

def train(group):
    verify_hashes(read(FIT/'plan.json')['hashes']);pilot.train(FIT,group)

def verify(group):
    plan=read(FIT/'plan.json');verify_hashes(plan['hashes'])
    records=selected(read(FIT/'training_manifest.json')['records'],group)
    trace=read(FIT/'models'/f'{group}.sampled_batches.json')
    assert trace['rows']==read(FIT/f'expected_{group}.json')
    assert trace['division_steps']==trace['ordinary_steps']==300 and trace['all_clusters_sampled']
    model=FIT/'models'/f'{group}_step600.pt';meta=read(model.with_suffix('.json'))
    assert sha(model)==meta['model_sha256'];ckpt=torch.load(model,map_location='cpu',weights_only=True)
    assert ckpt['step']==600 and ckpt['train_embryo']==group and ckpt['recipe']==pilot.RECIPE
    assert set(ckpt['train_movies'])=={r['stem'] for r in records}
    assert group=='pooled' or all(s.startswith(group) for s in ckpt['train_movies'])
    assert meta['base_transformer_unchanged'] and all(torch.isfinite(x).all() for x in ckpt['state_dict'].values())
    pu=old.engine(FIT);full,_,_=pu.load_model(old.PRIMARY_WEIGHTS,torch.device('cuda'));full.eval()
    assert state_digest(full.transformer.state_dict())==ckpt['base_transformer_sha256']
    direct=copy.deepcopy(full.transformer);direct.load_state_dict(ckpt['state_dict'],strict=True);direct.eval()
    frozen=state_digest({k:v for k,v in full.state_dict().items() if not k.startswith('transformer.')})
    os.environ['BIOHUB_C037_CHECKPOINT']=str(model);os.environ['BIOHUB_C037_ALPHA']='1.0';apply_primary(full)
    assert state_digest(full.transformer.state_dict())==state_digest(ckpt['state_dict'])
    assert frozen==state_digest({k:v for k,v in full.state_dict().items() if not k.startswith('transformer.')})
    checks=[]
    for pool in ['ordinary','division']:
        rec=next(r for r in records if r['pool']==pool)
        with np.load(rec['file']) as z:packet={k:z[k].copy() for k in z.files}
        with torch.inference_mode():a=old.forward(direct,packet);b=old.forward(full.transformer,packet)
        assert torch.equal(a,b);checks.append(dict(stem=rec['stem'],pool=pool,max_logit_error=0.))
    save_json(FIT/f'verification_{group}.json',dict(status='passed',group=group,step=600,model_sha256=sha(model),
        source_movies=ckpt['train_movies'],actual_samples=600,trace_exact=True,teacher_frozen=True,
        actual_runtime_reload=checks,no_graph_score_claim=True))

def run():
    assert not (FIT/'status.json').exists()
    assert datetime.now().astimezone()<datetime.fromisoformat('2026-09-29T18:00:00+09:00'),'Insufficient time for the full remaining validation/deployment path'
    prior=ROOT/'experiments/candidates/c065_pooled_transformer/finish/status.json'
    assert read(prior)['status']=='complete_local_verified_review_required','C065 local phase still owns priority'
    plan=read(FIT/'plan.json');q=Queue(FIT,1)
    try:
        q.state.update(total_jobs=6,plan_sha256=sha(FIT/'plan.json'));q.save();verify_hashes(plan['hashes'])
        for group in GROUPS:
            for stage in ['train','verify']:q.run(stage+'_'+group,[sys.executable,'-X','utf8','-u',Path(__file__),stage,'--group',group])
        verify_hashes(plan['hashes'])
        files=list((FIT/'models').glob('*'))+list(FIT.glob('verification_*.json'))+list((FIT/'logs').glob('*'))
        save_json(FIT/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
        q.close('complete_review_required')
    except BaseException as e:q.close('failed',e);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','verify','run']);p.add_argument('--group',choices=GROUPS)
    a=p.parse_args()
    if a.command in ['prepare','run']:globals()[a.command]()
    else:
        assert a.group is not None;globals()[a.command](a.group)
