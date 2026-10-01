"""Clean T4/exact-version pipeline for fixed appearance replacements.

Shared scientific sources stay immutable. Existing verification is extended with
explicit appearance asset/source checks; no training notebooks are submitted.
"""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,inspect,json
from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetDatasetRequest
from c069_appearance_replacement import root,notebook,PARENTS,FIXED
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,sha,save_json,stamp
import final_four_remote as remote
import final_four_verify as verifier
import final_four_submit as submitter
from kaggle_schannel_transport import install

for key in PARENTS: remote.CONFIG[key]=(root(key).name,key+'_local_review.json')

def configure_verifier():
    source=inspect.getsource(verifier.inspect_evidence)
    marker="'c053':'fixed_both_no_routing'"
    assert source.count(marker)==1
    source=source.replace(marker,marker+",'c069':'44b6','c070':'pooled'",1)
    scope=dict(verifier.__dict__)
    exec(compile(source,str(Path(__file__))+'::extended_model_labels','exec'),scope)
    verifier.inspect_evidence=scope['inspect_evidence']

def push(key):
    out=root(key);folder=out/'remote';folder.mkdir(exist_ok=True)
    assert datetime.now().astimezone()+timedelta(minutes=30+15)<datetime.fromisoformat('2026-09-30T00:00:00+09:00')
    assert not (folder/'push_action.json').exists(), 'No duplicate or uncertain push retry'
    local=read(out/'local_verification.json');review=ROOT/f'state/{key}_local_review.json'
    assert local['status']==read(review)['status']=='passed'
    assert read(out/'decision.json')['status']=='advance_clean_t4_exploratory'
    assert local['portable12']==local['writer4']=='exact' and local['repair_fallback']==local['deadline_degraded']==0
    verify_hashes(read(out/'local/plan.json')['hashes']);verify_hashes(read(out/'local/output_hashes.json'))
    meta=read(out/'kernel-metadata.json');assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert sha(notebook(key))==local['notebook_sha256']
    assert sha(out/'dataset/transformer_mean600.pt')==local['model_sha256']
    assert sha(out/'dataset/appearance_mean_cosine.pt')==local['appearance_model_sha256']
    install(['SaveKernel']);api=KaggleApi();api.authenticate();datasets=[]
    for ref in [s for s in meta['dataset_sources'] if s.startswith('taeyangg4/')]:
        with api.build_kaggle_client() as client:
            req=ApiGetDatasetRequest();req.owner_slug,req.dataset_slug=ref.split('/')
            data=json.loads(client.datasets.dataset_api_client.get_dataset(req).to_json())
        assert data['ref']==ref and data['isPrivate'] and data['currentVersionNumber']==1
        assert next(v['status'].lower() for v in data['versions'] if v['versionNumber']==1)=='ready'
        datasets.append(dict(ref=ref,version=1,private=True,ready=True))
    assert len(datasets)==2
    save_json(folder/'dataset_provenance.json',dict(checked=stamp(),existing_assets=True,datasets=datasets,
        expected_assets={n:sha(out/'dataset'/n) for n in ['transformer_mean600.pt','appearance_mean_cosine.pt']},
        actual_runtime_hashes_still_required=True))
    action=dict(status='push_started',started=stamp(),ref=meta['id'],clean_production=True,no_competition_submission=True,
        notebook_sha256=local['notebook_sha256'],model_sha256=local['model_sha256'],appearance_model_sha256=local['appearance_model_sha256'],
        local_csv_sha256=local['csv_sha256'],metadata_sha256=sha(out/'kernel-metadata.json'),decision_sha256=sha(out/'decision.json'),independent_review_sha256=sha(review))
    with (folder/'push_action.json').open('x',encoding='utf-8') as f:json.dump(action,f,indent=2)
    try:
        response=api.kernels_push(str(out),acc='NvidiaTeslaT4');save_json(folder/'push_response.json',json.loads(response.to_json()))
        assert not response.error and response.version_number==1 and response.ref.removeprefix('/code/')==meta['id']
        action.update(status='pushed',version=1,ended=stamp(),url=response.url);save_json(folder/'push_action.json',action)
        print(json.dumps(action),flush=True)
    except BaseException as exc:
        action.update(status='uncertain_or_rejected_no_retry',ended=stamp(),error=str(exc));save_json(folder/'push_action.json',action);raise

def watch(key):
    install();remote.watch(key)

def fetch(key):
    install();verifier.fetch(key)

def verify(key):
    install();configure_verifier();verifier.verify(key)
    out=root(key);dest=out/'remote'/(key+'_v1');local=read(out/'local_verification.json');push=read(out/'remote/push_action.json')
    log=''.join(x['data'] for x in read(next((dest/'output').glob('*.log'))))
    records=[s for s in log.splitlines() if s.startswith('FIXED_ASSET appearance_mean_cosine.pt ')]
    assert len(records)==1 and records[0].split()[-1]==local['appearance_model_sha256']==push['appearance_model_sha256']
    actual=read(dest/'source/notebook.ipynb');source=(FIXED/'portable_appearance.py').read_text(encoding='utf-8')
    assert repr(source) in ''.join(actual['cells'][5]['source'])
    assert sha(FIXED/'portable_appearance.py')==local['appearance_runtime_sha256']
    stats=__import__('pandas').read_csv(dest/'output/run_stats.csv')
    assert len(stats)==4
    report=read(dest/'T4_VERIFICATION.json');report.update(appearance_asset_hash_in_actual_log=True,
        appearance_model_sha256=local['appearance_model_sha256'],appearance_runtime_sha256=local['appearance_runtime_sha256'],
        actual_appearance_source_exact=True,shared_verifier_extension='Only model-group labels extended; original exact source/model/CSV/FP32/fallback gates retained')
    save_json(dest/'T4_VERIFICATION.json',report)
    hashes=read(dest/'artifact_hashes.json')
    for p in [dest/'T4_VERIFICATION.json',out/'dataset/appearance_mean_cosine.pt',Path(__file__),ROOT/'src/kaggle_schannel_transport.py',out/'remote/dataset_provenance.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    save_json(dest/'artifact_hashes.json',hashes);print(json.dumps(report),flush=True)

def submit(key):
    out=root(key);dest=out/'remote'/(key+'_v1');proof=read(dest/'T4_VERIFICATION.json')
    assert proof['appearance_asset_hash_in_actual_log'] and proof['actual_appearance_source_exact']
    review=read(ROOT/f'state/{key}_clean_t4_review.json')
    assert review['status']=='passed' and review['manifest_sha256']==sha(dest/'artifact_hashes.json')
    install(['CreateCodeSubmission']);submitter.submit(key)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['push','watch','fetch','verify','submit']);p.add_argument('--candidate',choices=list(PARENTS),required=True)
    a=p.parse_args();globals()[a.command](a.candidate)
