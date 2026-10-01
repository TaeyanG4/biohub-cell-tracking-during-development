"""Private bounded T4 research uploads/watch; no competition submission function."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetDatasetRequest
import c067_single_source as candidate
from reid_probe_local import ROOT,sha,stamp
from c047_hard_example_study import read

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8');temporary.replace(path)

def push(key, resume=False):
    out=candidate.root(key);folder=out/'remote_audit';plan=read(out/'plan.json')
    proof=read(ROOT/'state/c067_prepare_review.json');assert proof['status']=='passed'
    row=next(r for r in proof['candidates'] if r['candidate']==key)
    assert row['plan_sha256']==sha(out/'plan.json')
    metadata=read(folder/'kernel-metadata.json');assert metadata['is_private'] and metadata['id'].endswith('-train-audit')
    assert row['research_notebook_sha256']==sha(folder/metadata['code_file'])
    assert sha(out/'dataset/transformer_mean600.pt')==row['model_sha256']
    assert not (folder/'push_action.json').exists(),'No blind repeated kernel writes'
    assert (folder/'dataset_action.json').exists()==resume,'Use explicit resume only for a recorded successful dataset creation'
    assert datetime.now().astimezone()<datetime.fromisoformat('2026-09-29T19:00:00+09:00')
    api=KaggleApi();api.authenticate()
    dataset=read(out/'dataset/dataset-metadata.json')['id']
    files={p.name:sha(p) for p in (out/'dataset').iterdir() if p.is_file()}
    if resume:
        action=read(folder/'dataset_action.json')
        assert action['status']=='create_returned' and action['dataset']==dataset and action['private']
        assert action['files']==files
        returned=json.loads(action['returned']);assert returned['status']=='Ok' and not returned.get('error')
    else:
        action=dict(status='create_started',created=stamp(),dataset=dataset,private=True,files=files)
        save(folder/'dataset_action.json',action)
    try:
        if not resume:
            result=api.dataset_create_new(str(out/'dataset'),public=False,quiet=True,convert_to_csv=False)
            action.update(status='create_returned',returned=str(result),ended=stamp());save(folder/'dataset_action.json',action)
        for _ in range(30):
            with api.build_kaggle_client() as client:
                request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=dataset.split('/')
                result=client.datasets.dataset_api_client.get_dataset(request)
            record=json.loads(result.to_json())
            assert record['ref']==dataset and record['isPrivate'] and record['currentVersionNumber']==1
            status=next(v['status'].lower() for v in record['versions'] if v['versionNumber']==1)
            save(folder/'dataset_status.json',dict(checked=stamp(),status=status,canonical_get_dataset=record,resumed_existing_creation=resume))
            if status=='ready':break
            if status in ['error','failed']:raise RuntimeError('Dataset preparation failed: '+status)
            time.sleep(10)
        else:raise RuntimeError('Dataset not ready after bounded five-minute wait; do not duplicate create')
        listed=api.dataset_list_files(dataset+'/1',page_size=100)
        listed_record=json.loads(listed.to_json())
        assert not listed_record.get('errorMessage') and not listed_record.get('nextPageToken')
        expected={p.name:p.stat().st_size for p in (out/'dataset').iterdir() if p.is_file() and p.name!='dataset-metadata.json'}
        assert {r['name']:r['totalBytes'] for r in listed_record['datasetFiles']}==expected
        download=folder/'dataset_readback';download.mkdir(exist_ok=True)
        for name in expected:
            if not (download/name).exists():api.dataset_download_file(dataset+'/1',name,path=str(download),quiet=True)
            assert sha(download/name)==files[name],name
        save(folder/'dataset_readback_proof.json',dict(checked=stamp(),dataset=dataset,version=1,files={n:sha(download/n) for n in expected}))
        attempt=dict(status='push_started',started=stamp(),ref=metadata['id'],research_only_never_submit=True,
                     notebook_sha256=row['research_notebook_sha256'],model_sha256=row['model_sha256'],metadata_sha256=sha(folder/'kernel-metadata.json'))
        save(folder/'push_action.json',attempt)
        response=api.kernels_push(str(folder),timeout='18000',acc='NvidiaTeslaT4')
        save(folder/'push_response.json',json.loads(response.to_json()))
        if response.error:raise RuntimeError(response.error)
        assert response.version_number==1 and response.ref.removeprefix('/code/')==metadata['id'],str(response)
        attempt.update(status='pushed',version=response.version_number,ended=stamp(),url=response.url)
        save(folder/'push_action.json',attempt)
        print(json.dumps(attempt),flush=True)
    except BaseException as exc:
        error_path=folder/'action_error.json'
        if error_path.exists():error_path=folder/('action_error_'+str(time.time_ns())+'.json')
        save(error_path,dict(time=stamp(),error=str(exc),no_automatic_retry=True))
        raise

def watch(key):
    folder=candidate.root(key)/'remote_audit';action=read(folder/'push_action.json')
    assert action['status']=='pushed' and action['research_only_never_submit'] and action['version']==1
    assert not (folder/'status.json').exists(),'No duplicate watcher'
    api=KaggleApi();api.authenticate()
    state=dict(status='running',pid=os.getpid(),started=stamp(),ref=action['ref'],version=1,
               current='remote_train_audit',research_only_never_submit=True)
    save(folder/'status.json',state)
    failures=0
    while datetime.now().astimezone()<datetime.fromisoformat('2026-09-30T00:00:00+09:00'):
        try:
            response=api.kernels_status(action['ref']);name=response.status.name.lower();failures=0
            state.update(updated=stamp(),remote_status=name,remote_failure=response.failure_message)
            if name=='complete':
                state.update(status='complete_review_required',ended=stamp(),current='download_exact_version_then_verify')
                save(folder/'status.json',state);print(json.dumps(state),flush=True);return
            if name in ['error','failed','cancelled','canceled']:
                raise RuntimeError('Remote '+name+': '+response.failure_message)
            save(folder/'status.json',state)
        except Exception as exc:
            failures+=1;state.update(updated=stamp(),last_poll_error=str(exc),consecutive_poll_errors=failures)
            if failures>=3 or 'Remote ' in str(exc):
                state.update(status='failed',error=str(exc),ended=stamp());save(folder/'status.json',state);raise
            save(folder/'status.json',state)
        time.sleep(300)
    state.update(status='failed',error='Midnight completion boundary reached',ended=stamp());save(folder/'status.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['push','resume','watch']);parser.add_argument('--candidate',choices=list(candidate.CONFIG),required=True)
    args=parser.parse_args()
    if args.command=='resume':push(args.candidate,resume=True)
    else:globals()[args.command](args.candidate)
