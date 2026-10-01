"""Finish C065 after preserving the old mean-model-only metadata assertion."""
from pathlib import Path
import argparse
import ast
from datetime import datetime,timedelta
import hashlib
import json
import math
import os
import sys
import time
os.environ.setdefault('POLARS_MAX_THREADS','4')
import c065_pooled_deploy as candidate
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,sha,save_json,stamp
from run_last_days_local import Queue,replay_command

DEPLOY=candidate.DEST
OUT=DEPLOY.parent/'finish'
EXPECTED_COMPLETED=['reference_smoke','infer_heldout12','exact_portable_reproduction','replay_heldout12',
    'infer_confirm10','replay_confirm10','infer_extension75','replay_extension75']

def prepare():
    assert not (OUT/'contract.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    _,derived=candidate.make_reused_module()
    function=next(n for n in ast.parse(derived).body if isinstance(n,ast.FunctionDef) and n.name=='replica_analyse')
    source=ast.get_source_segment(derived,function)
    assert source.count("'fixed_both_no_routing' in log")==1
    inputs={Path(__file__),ROOT/'src/c065_pooled_deploy.py',OUT/'launch.ps1',DEPLOY/'plan.json',
            ROOT/'src/c053_fixed_division_transformer.py',ROOT/'tools/notify_background_completion.ps1',
            DEPLOY/'mean_provenance.json',ROOT/'state/c065_fit_review.json'}
    launch=read(DEPLOY/'launch.json');eta=datetime.fromisoformat(launch['estimated_completion'])+timedelta(minutes=25)
    minutes=max(5,math.ceil((eta-datetime.now().astimezone()).total_seconds()/60))
    save_json(OUT/'contract.json',dict(created=stamp(),hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        expected_original_plan_sha256=sha(DEPLOY/'plan.json'),estimated_minutes=minutes,total_jobs=5,
        expected_failed_job='fixed97_analysis',expected_prior_complete=EXPECTED_COMPLETED,
        cause='C053 inherited analysis expects fixed_both_no_routing metadata; actual C065 directly trained model correctly says pooled',
        allowed_change='Only expected log marker after exact parsed runtime metadata and checkpoint-path checks',
        no_training=True,no_inference_rerun=True,no_original_source_or_plan_or_status_edit=True,
        no_score_gate_or_model_change=True))
    print(json.dumps(dict(prepared=True,estimated_minutes=minutes,waiting_for_original=True)),flush=True)

def assert_runtime_metadata():
    model_path=(DEPLOY/'dataset/transformer_mean600.pt').resolve();receipt=[]
    for split in ['heldout12','confirm10','extension75']:
        log=(DEPLOY/'e2e'/split/'predict.log').read_text(encoding='utf-8')
        records=[]
        for line in log.splitlines():
            if 'C037_PRIMARY ' not in line:continue
            r=ast.literal_eval(line.split('C037_PRIMARY ',1)[1].strip())
            assert r['train_embryo']=='pooled' and r['step']==600 and r['alpha']==1.
            assert Path(r['path']).resolve()==model_path
            records.append(r)
        assert records and 'fixed_both_no_routing' not in log
        receipt.append(dict(split=split,parsed_records=len(records),actual_train_embryo='pooled',step=600,
            checkpoint=str(model_path),checkpoint_sha256=sha(model_path)))
    return receipt

def repaired_module():
    module,derived=candidate.make_reused_module()
    function=next(n for n in ast.parse(derived).body if isinstance(n,ast.FunctionDef) and n.name=='replica_analyse')
    original=ast.get_source_segment(derived,function)
    assert original.count("'fixed_both_no_routing' in log")==1
    corrected=original.replace("'fixed_both_no_routing' in log","'pooled' in log")
    metadata=assert_runtime_metadata()
    exec(compile(corrected,str(Path(__file__))+'::metadata_only_adapter','exec'),module.__dict__)
    save_json(OUT/'metadata_adapter_proof.json',dict(status='passed',runtime=metadata,
        exact_string_change=["'fixed_both_no_routing' in log","'pooled' in log"],
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        corrected_function_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
        actual_checkpoint_type_verified=True,no_fake_log_flag=True,no_prediction_or_threshold_change=True))
    return module

def register_inputs():
    original=read(DEPLOY/'plan.json');verify_hashes(original['hashes'])
    hashes=dict(original['hashes']);inputs={OUT/'contract.json',DEPLOY/'status.json',DEPLOY/'smoke.json'}
    for folder in ['e2e','graphs','replay']:
        inputs.update(p for p in (DEPLOY/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    inputs.update(ROOT/p for p in read(OUT/'contract.json')['hashes'])
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
    save_json(OUT/'plan.json',dict(created=stamp(),hashes=hashes,total_jobs=5,
        original_status_preserved='failed at inherited metadata assertion',predictions_reused=True,
        model_and_gates_unchanged=True,metadata=assert_runtime_metadata()))
    return hashes

def run():
    assert not (OUT/'status.json').exists()
    contract=read(OUT/'contract.json');q=Queue(OUT,8)
    try:
        verify_hashes(contract['hashes']);q.state.update(total_jobs=5,current='waiting_for_original_compute',contract_sha256=sha(OUT/'contract.json'));q.save()
        while True:
            if datetime.now().astimezone()>=datetime.fromisoformat('2026-09-30T00:00:00+09:00'):
                raise RuntimeError('User midnight completion boundary reached')
            status=read(DEPLOY/'status.json')
            if status['status'] not in ['running','starting']:break
            time.sleep(5)
        assert status['status']=='failed',status
        jobs=status['jobs'];assert all(jobs[name]['status']=='complete' for name in EXPECTED_COMPLETED)
        assert jobs['fixed97_analysis']['status']=='failed'
        assert set(jobs)==set(EXPECTED_COMPLETED+['fixed97_analysis'])
        assert_runtime_metadata();verify_hashes(contract['hashes'])
        hashes=register_inputs();q.state.update(plan_sha256=sha(OUT/'plan.json'),current=None);q.save()
        command=[sys.executable,'-X','utf8','-u',Path(__file__)]
        q.run('actual97_analysis_metadata_corrected',command+['analyse'])
        module,_=candidate.make_reused_module();plan=read(DEPLOY/'plan.json')
        os.environ['BIOHUB_FIXED_ASSET_DIR']=str(DEPLOY/'dataset');os.environ['PYTHONUTF8']='1'
        q.run('actual_portable_replay12',replay_command(module.nbpath(DEPLOY),DEPLOY/'e2e/heldout12',plan['jobs'][0]['stems'],
            DEPLOY/'variants.json',DEPLOY/'replay/portable_heldout12.csv'))
        q.run('production_writer4',command+['write_visible'])
        q.run('actual_official_visible4',[sys.executable,'-X','utf8','-u',ROOT/'src/evaluate_local.py',
            '--csv',DEPLOY/'c065/submission.csv','--gt-dir',ROOT/'data/visible_gt/train','--out-dir',DEPLOY/'c065/evaluation'])
        q.run('verify_local',command+['verify_local'])
        verify_hashes(hashes);verify_hashes(contract['hashes'])
        files=[p for folder in ['replay','graphs','e2e','c065'] for p in (DEPLOY/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        files += [p for p in DEPLOY.iterdir() if p.is_file() and p.suffix in ['.json','.csv'] and p.name not in ['plan.json','status.json','launch.json','artifact_hashes.json','automation_confirmation.json']]
        save_json(DEPLOY/'artifact_hashes.json',{str(p.relative_to(DEPLOY)):sha(p) for p in files})
        final=files+[DEPLOY/'artifact_hashes.json',OUT/'metadata_adapter_proof.json']+list((OUT/'logs').glob('*'))
        save_json(OUT/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in final})
        q.close('complete_local_verified_review_required')
    except BaseException as e:q.close('failed',e);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyse','write_visible','verify_local'])
    action=p.parse_args().command
    if action in ['prepare','run']:globals()[action]()
    elif action=='analyse':repaired_module().analyse(DEPLOY)
    else:
        module,_=candidate.make_reused_module();getattr(module,action)(DEPLOY)
