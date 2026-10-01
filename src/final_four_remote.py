"""Bounded clean production T4 push/watch for reviewed final candidates; no submit."""
from pathlib import Path
import argparse,json,os,time
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetDatasetRequest
from c047_hard_example_study import read
from c067_remote_research import save
from reid_probe_local import ROOT,sha,stamp

CONFIG={'c065':('c065_pooled_transformer/deploy','c065_deploy_review.json'),
        'c066':('c066_expanded_ordinary/deploy','c066_deploy_review.json'),
        'c067':('c067_single_source_44b6','c067_local_review.json'),
        'c068':('c068_single_source_6bba','c068_local_review.json')}
def root(key):return ROOT/'experiments/candidates'/CONFIG[key][0]
def push(key):
    out=root(key);folder=out/'remote';folder.mkdir(exist_ok=True)
    review=read(ROOT/'state'/CONFIG[key][1]);assert review['status']=='passed'
    decision=read(out/'decision.json');assert decision['status']=='advance_clean_t4_exploratory'
    local=read(out/'local_verification.json');assert local['status']=='passed'
    assert local['portable12']==local['writer4']=='exact' and local['repair_fallback']==local['deadline_degraded']==0
    metadata=read(out/'kernel-metadata.json');assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert '-train-audit' not in metadata['id']
    assert sha(out/metadata['code_file'])==local['notebook_sha256']
    assert sha(out/'dataset/transformer_mean600.pt')==local['model_sha256']
    assert not (folder/'push_action.json').exists(),'Inspect exact accepted/uncertain version; never blindly repush'
    assert datetime.now().astimezone()<datetime.fromisoformat('2026-09-29T23:00:00+09:00')
    api=KaggleApi();api.authenticate();dataset=read(out/'dataset/dataset-metadata.json')['id']
    existing=(out/'remote_audit/dataset_action.json') if key in ['c067','c068'] else folder/'dataset_action.json'
    files={p.name:sha(p) for p in (out/'dataset').iterdir() if p.is_file()}
    try:
        if existing.exists():
            created=read(existing);assert created['status']=='create_returned' and created['dataset']==dataset and created['files']==files
            response=json.loads(created['returned']);assert response['status']=='Ok' and not response.get('error')
        else:
            action=dict(status='create_started',created=stamp(),dataset=dataset,private=True,files=files)
            save(existing,action)
            response=api.dataset_create_new(str(out/'dataset'),public=False,quiet=True,convert_to_csv=False)
            action.update(status='create_returned',returned=str(response),ended=stamp());save(existing,action)
            result=json.loads(str(response));assert result['status']=='Ok' and not result.get('error')
        for attempt in range(30):
            try:
                with api.build_kaggle_client() as client:
                    req=ApiGetDatasetRequest();req.owner_slug,req.dataset_slug=dataset.split('/')
                    data=json.loads(client.datasets.dataset_api_client.get_dataset(req).to_json())
                assert data['ref']==dataset and data['isPrivate'] and data['currentVersionNumber']==1
                status=next(v['status'].lower() for v in data['versions'] if v['versionNumber']==1)
                save(folder/'dataset_status.json',dict(checked=stamp(),canonical=data,status=status))
                if status=='ready':break
                if status in ['error','failed']:raise RuntimeError('Dataset preparation failed')
            except Exception as exc:
                save(folder/('dataset_read_error_'+str(attempt)+'.json'),dict(checked=stamp(),error=str(exc)))
                if attempt==29:raise
            time.sleep(10)
        else:raise RuntimeError('Dataset readiness timeout; creation not repeated')
        downloaded=folder/'dataset_readback';downloaded.mkdir(exist_ok=True)
        for name,digest in files.items():
            if name=='dataset-metadata.json':continue
            if not (downloaded/name).is_file():api.dataset_download_file(dataset+'/1',name,path=str(downloaded),quiet=True)
            assert sha(downloaded/name)==digest,name
        save(folder/'dataset_readback_proof.json',dict(status='passed',checked=stamp(),dataset=dataset,version=1,files={n:h for n,h in files.items() if n!='dataset-metadata.json'}))
        action=dict(status='push_started',started=stamp(),ref=metadata['id'],notebook_sha256=local['notebook_sha256'],model_sha256=local['model_sha256'],
            local_csv_sha256=local['csv_sha256'],metadata_sha256=sha(out/'kernel-metadata.json'),decision_sha256=sha(out/'decision.json'),
            independent_review_sha256=sha(ROOT/'state'/CONFIG[key][1]),clean_production=True,no_competition_submission=True)
        save(folder/'push_action.json',action)
        response=api.kernels_push(str(out),acc='NvidiaTeslaT4')
        save(folder/'push_response.json',dict(json.loads(response.to_json()),kernelId=response.kernel_id,ref=response.ref,versionNumber=response.version_number))
        if response.error=='Maximum batch GPU session count of 2 reached.' and not response.ref and response.kernel_id==0:
            action.update(status='rejected_capacity',ended=stamp(),error=response.error,no_kernel_created=True)
            save(folder/'push_action.json',action)
            save(folder/'deferred_capacity.json',dict(status='await_existing_research_slot',created=stamp(),
                priority='C065 clean production validation before any later new GPU research',no_competition_submission=True))
            print(json.dumps(action),flush=True);return
        assert not response.error and response.version_number==1 and response.ref.removeprefix('/code/')==metadata['id'],str(response)
        action.update(status='pushed',version=1,ended=stamp(),url=response.url);save(folder/'push_action.json',action)
        print(json.dumps(action),flush=True)
    except BaseException as exc:
        save(folder/('action_error_'+str(time.time_ns())+'.json'),dict(checked=stamp(),error=str(exc),no_blind_write_retry=True));raise

def retry_capacity(key):
    folder=root(key)/'remote';action=read(folder/'push_action.json');response=read(folder/'push_response.json')
    assert action['status']=='rejected_capacity' and action['no_kernel_created']
    assert response.get('kernelId',0)==0 and not response.get('ref') and response.get('versionNumber') is None
    assert response['error']=='Maximum batch GPU session count of 2 reached.'
    api=KaggleApi();api.authenticate();free=[]
    for tag in ['c067','c068']:
        state=read(root(tag)/'remote_audit/status.json')
        if state['status'] not in ['complete_review_required','failed']:continue
        current=api.kernels_status(state['ref'])
        if current.status.name.lower() in ['complete','error','failed','cancelled','canceled']:free.append(state['ref'])
    assert free,'Capacity retry requires fresh terminal evidence from one of our two research sessions'
    suffix=str(time.time_ns())
    save(folder/('capacity_retry_evidence_'+suffix+'.json'),dict(checked=stamp(),terminal_research_refs=free))
    (folder/'push_action.json').rename(folder/('push_action_rejected_'+suffix+'.json'))
    (folder/'push_response.json').rename(folder/('push_response_rejected_'+suffix+'.json'))
    push(key)

def watch(key):
    folder=root(key)/'remote';action=read(folder/'push_action.json')
    assert action['status']=='pushed' and action['version']==1 and action['clean_production']
    assert not (folder/'status.json').exists()
    api=KaggleApi();api.authenticate();failures=0
    state=dict(status='running',pid=os.getpid(),started=stamp(),ref=action['ref'],version=1,current='clean_production_T4',no_competition_submission=True)
    save(folder/'status.json',state)
    while datetime.now().astimezone()<datetime.fromisoformat('2026-09-30T00:00:00+09:00'):
        try:
            response=api.kernels_status(action['ref']);name=response.status.name.lower();failures=0
            state.update(updated=stamp(),remote_status=name,remote_failure=response.failure_message)
            if name=='complete':
                state.update(status='complete_review_required',ended=stamp(),current='exact_version_T4_verification_required');save(folder/'status.json',state);return
            if name in ['error','failed','cancelled','canceled']:raise RuntimeError('Remote '+name+': '+response.failure_message)
            save(folder/'status.json',state)
        except Exception as exc:
            failures+=1;state.update(updated=stamp(),last_poll_error=str(exc),consecutive_poll_errors=failures)
            if failures>=3 or 'Remote ' in str(exc):
                state.update(status='failed',ended=stamp(),error=str(exc));save(folder/'status.json',state);raise
            save(folder/'status.json',state)
        time.sleep(300)
    state.update(status='failed',ended=stamp(),error='Midnight working cutoff');save(folder/'status.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['push','watch','retry_capacity']);p.add_argument('--candidate',choices=list(CONFIG),required=True)
    a=p.parse_args();globals()[a.command](a.candidate)
