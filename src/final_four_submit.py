"""One authorized exact-version submission after integrity/science/quota gates.

No score polling, retries, notebook pushes, or candidate selection. An existing
action (including an uncertain one) prevents any second submission attempt.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,csv,json,os
from kaggle.api.kaggle_api_extended import KaggleApi
from c067_collect import get_version
from final_four_remote import root,CONFIG
from c047_hard_example_study import read
from reid_probe_local import ROOT,sha,stamp

COMPETITION='biohub-cell-tracking-during-development'
CUTOFF=datetime.fromisoformat('2026-09-30T00:00:00+09:00')

def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)

def submit(key):
    assert datetime.now().astimezone()<CUTOFF
    out=root(key);remote=out/'remote';dest=remote/(key+'_v1')
    action_path=remote/'submit_action.json'
    assert not action_path.exists(),'Existing attempt; inspect acceptance before any action'
    policy_path=ROOT/'state/submission_budget_20260927/policy.json';policy=read(policy_path)
    authorization=policy['latest_explicit_autonomous_authorization']
    assert key in authorization['conditional_candidate_scope'] and not authorization['renewed_approval_required']
    monitor_path=ROOT/'state/biohub_t4_monitor.json';monitor=read(monitor_path)
    prior_new={k:v for k,v in monitor['submitted_candidates'].items() if k in CONFIG}
    assert key not in prior_new and len(prior_new)<4
    decision=read(out/'decision.json');assert decision['status']=='advance_clean_t4_exploratory'
    review_path=ROOT/'state'/CONFIG[key][1];assert read(review_path)['status']=='passed'
    local=read(out/'local_verification.json');t4=read(dest/'T4_VERIFICATION.json');push=read(remote/'push_action.json')
    assert t4['status']==local['status']=='passed' and push['status']=='pushed'
    assert push['clean_production'] and t4['version']==push['version']==1
    assert '-train-audit' not in t4['ref'] and t4['ref']==push['ref']
    assert local['portable12']==local['writer4']=='exact'
    assert local['repair_fallback']==local['deadline_degraded']==t4['all4_repair_fallback']==t4['all4_deadline_degraded']==0
    assert t4['local_csv_byte_identical'] and t4['exact_predictor_runtime_head'] and t4['all4_FP32_mathSDPA']
    assert abs(t4['actual_T4_visible4']-local['local_visible4'])<1e-12
    assert push['decision_sha256']==sha(out/'decision.json') and push['independent_review_sha256']==sha(review_path)
    hashes={}
    def add(values):
        for name,digest in values.items():
            name=str(Path(name));assert name not in hashes or hashes[name]==digest,name
            hashes[name]=digest
    add(read(dest/'artifact_hashes.json'))
    add(read(out/'plan.json')['hashes'])
    if key=='c065':
        finish=out.parent/'finish'
        add(read(finish/'contract.json')['hashes']);add(read(finish/'plan.json')['hashes']);add(read(finish/'output_hashes.json'))
    if key in ('c065','c066'):
        add({str((out/p).relative_to(ROOT)):h for p,h in read(out/'artifact_hashes.json').items()})
    else:
        add(read(out/'local/plan.json')['hashes']);add(read(out/'local/output_hashes.json'))
    for i,(name,digest) in enumerate(hashes.items()):
        assert sha(ROOT/name)==digest,name
        if (i+1)%10000==0:print('pre-submit rehashed',i+1,flush=True)
    meta=read(out/'kernel-metadata.json')
    assert sha(out/meta['code_file'])==t4['notebook_sha256']==local['notebook_sha256']==push['notebook_sha256']
    assert sha(out/'dataset/transformer_mean600.pt')==t4['model_sha256']==local['model_sha256']==push['model_sha256']
    digest=sha(dest/'output/submission.csv');assert digest==t4['submission_sha256']==local['csv_sha256']==push['local_csv_sha256']
    prior_proofs={ROOT/v['proof'] for v in monitor['submitted_candidates'].values() if v.get('proof')}
    prior_proofs.update((ROOT/'experiments/candidates').glob('*/remote/*_v*/T4_VERIFICATION.json'))
    prior_proofs.update((ROOT/'state/c042_c043_portable/remote').glob('*_v*/T4_VERIFICATION.json'))
    compared=[]
    for p in sorted(prior_proofs):
        if p==dest/'T4_VERIFICATION.json':continue
        record=read(p)
        prior_digest=record.get('submission_sha256') or record.get('csv_sha256') or record.get('submission_csv_sha256')
        csv_path=p.parent/'output/submission.csv'
        if csv_path.exists():
            actual=sha(csv_path)
            if prior_digest:assert prior_digest==actual,str(p)
            prior_digest=actual
        assert prior_digest,'Missing prior output digest: '+str(p)
        assert prior_digest!=digest,('Duplicate output',str(p))
        compared.append({'proof':str(p.relative_to(ROOT)),'sha256':prior_digest})
    ledger_path=ROOT/'experiments/submission_log.csv'
    with ledger_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    exact_ref=t4['ref']+' v1'
    assert not any(r['submission_id'] and r['kaggle_ref']==exact_ref for r in rows)
    api=KaggleApi();api.authenticate()
    canonical=get_version(api,t4['ref']);actual=json.loads(canonical.blob.source);expected=read(out/meta['code_file'])
    assert len(actual['cells'])==len(expected['cells']) and all(a['cell_type']==b['cell_type'] and ''.join(a['source'])==''.join(b['source']) for a,b in zip(actual['cells'],expected['cells']))
    # Only metadata is retained from the shared list; score fields are never read.
    entries=api.competition_submissions(COMPETITION,page_size=100) or []
    metadata=[{'ref':str(s.ref),'date':str(s.date),'description':s.description,'submitted_by':str(s.submitted_by)} for s in entries]
    assert not any(key.upper() in (s['description'] or '') for s in metadata),'Possible existing candidate; reconcile without resubmission'
    limits=api.competition_get_submission_limits(COMPETITION)
    quota={'checked':stamp(),'num_today':limits.num_today,'num_total':limits.num_total,'num_allowed_now':limits.num_allowed_now,'limited_by_total':limits.limited_by_total}
    save(remote/'pre_submit_readback.json',{'quota':quota,'submission_metadata':metadata,'score_fields_not_read':True})
    assert limits.num_allowed_now>0,'Fresh shared quota exhausted'
    assert datetime.now().astimezone()<CUTOFF
    save(remote/'pre_submit_integrity.json',{'checked':stamp(),'unique_rehashed_files':len(hashes),'t4_manifest_sha256':sha(dest/'artifact_hashes.json'),'independent_review_sha256':sha(review_path),'decision_sha256':sha(out/'decision.json'),'prior_outputs':compared,'duplicate':False,'version':1,'ref':t4['ref']})
    message=f'{key.upper()} fixed Transformer v1; actual97 review and exact local/T4 source-model-CSV verified'
    action={'status':'submit_started','started':stamp(),'ref':t4['ref'],'version':1,'message':message,'quota':quota,
            'notebook_sha256':t4['notebook_sha256'],'model_sha256':t4['model_sha256'],'submission_sha256':digest}
    # Exclusive creation prevents duplicate dispatch by another observer.
    with action_path.open('x',encoding='utf-8') as f:json.dump(action,f,indent=2)
    try:
        response=api.competition_submit_code(file_name='submission.csv',message=message,competition=COMPETITION,kernel=t4['ref'],kernel_version=1,quiet=True)
        save(remote/'submit_response.json',json.loads(response.to_json()))
        assert response.ref>0,'No accepted ID returned; inspect uncertain action without retry'
        action.update(status='accepted',submission_id=response.ref,accepted_at=stamp());save(action_path,action)
    except BaseException as exc:
        action.update(status='uncertain_or_rejected_no_retry',error=str(exc),ended=stamp());save(action_path,action);raise
    # Persist acceptance before any further remote operation (none follows).
    with ledger_path.open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(dict(submission_id=action['submission_id'],datetime=action['accepted_at'],experiment_id=out.parent.name if key in ('c065','c066') else out.name,kaggle_ref=exact_ref,message=message,public_lb='',private_lb='',status='submitted_pending',notes='Autonomous final-four authorization; scientific review + exact T4/local proof, fresh team quota and output/version dedup. No score polling.'))
    monitor=read(monitor_path)
    monitor['submitted_candidates'][key]={'submission_id':action['submission_id'],'version':1,'ref':t4['ref'],'submitted_at':action['accepted_at'],'proof':str((dest/'T4_VERIFICATION.json').relative_to(ROOT)),'action':str(action_path.relative_to(ROOT))}
    monitor.update(updated=stamp(),remaining_submissions_after_our_action_upper_bound=min(limits.num_allowed_now-1,3-len(prior_new)))
    save(monitor_path,monitor)
    policy=read(policy_path);policy['our_new_submission_ids'].append(action['submission_id']);policy.update(updated=stamp(),observed_remaining_upper_bound=limits.num_allowed_now-1,last_readback_path=str((remote/'pre_submit_readback.json').relative_to(ROOT)),last_readback_at=quota['checked'],quota_day_utc=datetime.now(timezone.utc).date().isoformat())
    save(policy_path,policy)
    print(json.dumps(action),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',choices=list(CONFIG),required=True)
    submit(p.parse_args().candidate)
