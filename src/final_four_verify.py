"""Exact-version clean T4 verification. No competition submission side effects."""
from pathlib import Path
import argparse,ast,json,os,subprocess,sys
os.environ.setdefault('POLARS_MAX_THREADS','4')
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from c067_collect import get_version
from final_four_remote import root,CONFIG
from c047_hard_example_study import read
from reid_probe_local import ROOT,sha,save_json,stamp
from final_four_download import fetch_required

SCRIPTS=['predict_unet_transformer.py','v1284_coordinate_refinement.py','c037_transformer_runtime.py']

def fetch(key):
    out=root(key);remote=out/'remote';action=read(remote/'push_action.json')
    assert action['status']=='pushed' and action['version']==1 and action['clean_production']
    assert read(remote/'status.json')['status']=='complete_review_required'
    dest=remote/(key+'_v1');assert not (dest/'source/provenance.json').exists(),'Already fetched; inspect/reuse exact evidence'
    source=dest/'source';output=dest/'output';source.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=True)
    api=KaggleApi();api.authenticate();response=get_version(api,action['ref'])
    actual=json.loads(response.blob.source);meta=read(out/'kernel-metadata.json');expected=read(out/meta['code_file'])
    assert len(actual['cells'])==len(expected['cells']) and all(a['cell_type']==b['cell_type'] and ''.join(a['source'])==''.join(b['source']) for a,b in zip(actual['cells'],expected['cells']))
    assert sha(out/meta['code_file'])==action['notebook_sha256']
    (source/'notebook.ipynb').write_text(response.blob.source,encoding='utf-8')
    download=fetch_required(api,action['ref'],output,['submission.csv','run_stats.csv']+['tracking_repo/scripts/'+n for n in SCRIPTS])
    get_version(api,action['ref'])
    required=[output/'submission.csv',output/'run_stats.csv']+[output/'tracking_repo/scripts'/n for n in SCRIPTS]
    assert all(p.is_file() for p in required) and len(list(output.glob('*.log')))==1
    save_json(source/'provenance.json',dict(status='passed',checked=stamp(),ref=action['ref'],version=1,private=True,
        gpu_enabled=True,internet_enabled=False,normalized_notebook_cells_exact=True,local_notebook_sha256=action['notebook_sha256'],
        source_download_sha256=sha(source/'notebook.ipynb'),version_checked_before_and_after_output_download=True,download_receipt=download))
    print(json.dumps(dict(status='fetched',candidate=key,version=1)),flush=True)

def inspect_evidence(out,key,dest):
    local=read(out/'local_verification.json');output=dest/'output';provenance=read(dest/'source/provenance.json')
    assert provenance.get('status','passed')=='passed'
    assert sha(output/'submission.csv')==local['csv_sha256']==sha(out/key/'submission.csv')
    stats=pd.read_csv(output/'run_stats.csv');expected=pd.read_csv(out/key/'run_stats.csv')
    assert len(stats)==4 and set(stats.dataset)==set(expected.dataset)
    assert (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()
    scripts={}
    for name in SCRIPTS:
        actual=(output/'tracking_repo/scripts'/name).read_text(encoding='utf-8')
        assert actual==(out/'tracking_repo/scripts'/name).read_text(encoding='utf-8'),name
        scripts[name]=sha(output/'tracking_repo/scripts'/name)
    logfiles=list(output.glob('*.log'));assert len(logfiles)==1
    log=''.join(x['data'] for x in read(logfiles[0]));hardware=[];models=[];assets=[]
    group={'c065':'pooled','c066':'pooled','c067':'44b6','c068':'6bba','c053':'fixed_both_no_routing'}[key]
    for line in log.splitlines():
        if line.startswith('LOCAL_RUNTIME '):
            row=ast.literal_eval(line.split(' ',1)[1]);assert 'T4' in row['gpu'] and row['math_sdp']
            assert not any(row[k] for k in ['cudnn_tf32','matmul_tf32','flash_sdp','mem_efficient_sdp']);hardware.append(row)
        if line.startswith('C037_PRIMARY '):
            row=ast.literal_eval(line.split(' ',1)[1]);assert row['train_embryo']==group and row['step']==600 and row['alpha']==1.
            assert row['path'].endswith('/transformer_mean600.pt');models.append(row)
        if line.startswith('FIXED_ASSET transformer_mean600.pt '):
            assert line.split()[-1]==local['model_sha256'];assets.append(line)
    assert len(hardware)==len(models)==4 and len(assets)==1
    return dict(local=local,script_hashes=scripts,hardware=hardware,model_loads=models)

def verify(key):
    out=root(key);remote=out/'remote';dest=remote/(key+'_v1');action=read(remote/'push_action.json')
    proof=inspect_evidence(out,key,dest);local=proof['local'];source=read(dest/'source/provenance.json')
    assert action['status']=='pushed' and action['version']==1 and action['clean_production']
    assert source['ref']==action['ref'] and source['version']==1 and source['normalized_notebook_cells_exact']
    assert source['local_notebook_sha256']==local['notebook_sha256']==action['notebook_sha256']
    assert sha(out/'dataset/transformer_mean600.pt')==local['model_sha256']==action['model_sha256']
    evaluation=dest/'evaluation'
    if not (evaluation/'summary.json').exists():
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'src/evaluate_local.py'),'--csv',str(dest/'output/submission.csv'),'--gt-dir',str(ROOT/'data/visible_gt/train'),'--out-dir',str(evaluation)],check=True)
    score=read(evaluation/'summary.json')['total_score'];assert abs(score-local['local_visible4'])<1e-12
    api=KaggleApi();api.authenticate();get_version(api,action['ref'])
    report=dict(status='passed',verified=stamp(),ref=action['ref'],version=1,notebook_sha256=local['notebook_sha256'],
        model_sha256=local['model_sha256'],submission_sha256=local['csv_sha256'],actual_T4_visible4=score,
        local_csv_byte_identical=True,exact_predictor_runtime_head=True,actual_source_hashes=proof['script_hashes'],
        all4_repair_fallback=0,all4_deadline_degraded=0,model_hash_in_actual_log=True,actual_model_loads=4,
        actual_gpu='Tesla T4',all4_FP32_mathSDPA=True,source_version_provenance='source/provenance.json',
        fresh_shared_quota_output_dedup_and_scientific_decision_still_required_before_submission=True)
    save_json(dest/'T4_VERIFICATION.json',report)
    paths={p for p in dest.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    paths.update([out/'local_verification.json',out/'decision.json',ROOT/'state'/CONFIG[key][1],Path(__file__),ROOT/'src/final_four_download.py',out/'kernel-metadata.json',out/read(out/'kernel-metadata.json')['code_file'],out/'dataset/transformer_mean600.pt'])
    save_json(dest/'artifact_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fetch','verify']);p.add_argument('--candidate',choices=list(CONFIG),required=True)
    a=p.parse_args();globals()[a.command](a.candidate)
