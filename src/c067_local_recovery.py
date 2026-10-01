"""Same-device recovery preserving the original exact cache/ILP/graph gates.

Prepare is CPU-only. Run is a separate finite GPU phase, permitted only after
C066 transfer finishes and an explicit recorded scheduling decision exists.
"""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,os,shutil,sys
import numpy as np
import c067_collect as c
import c067_single_source as candidate
import c052_division_transformer as pilot
from reid_probe_local import ROOT,sha,save_json,stamp
from run_last_days_local import Queue

def prepare(key):
    out=candidate.root(key);phase=out/'local_recovery';phase.mkdir(exist_ok=True)
    assert not (phase/'plan.json').exists() and not (out/'local/plan.json').exists()
    review=c.read(ROOT/'state'/f'{key}_diagnostic_review.json')
    assert review['status']=='diagnostic_integrity_passed_not_submission_eligible'
    original=c.read(out/'plan.json');module,_=candidate.module_for(key)
    stems=original['remote_stems'];save_json(phase/'variants.json',{'as_configured':{}})
    infer=[str(x) for x in module.infer(out,'local_recovery',stems)]
    replay=[str(x) for x in pilot.replay(phase,'same_device',stems,out/'e2e/local_recovery')]
    hashes=dict(original['hashes'])
    extra=[Path(__file__),ROOT/'tools/launch_c067_recovery.ps1',phase/'same_device.ipynb',phase/'variants.json',ROOT/'state'/f'{key}_diagnostic_review.json',ROOT/'src/c067_collect.py',ROOT/'src/final_four_download.py',ROOT/'src/c066_transfer97.py',ROOT/'src/c058_localizer_baseline.py',ROOT/'src/evaluate_local.py',ROOT/'state/validation_audit_20260929/actual_per_movie.csv',ROOT/'state/c067_prepare_review.json',ROOT/'state/review_c067.py',out/'remote_audit/v1/fetch_proof.json',out/'remote_audit/v1/research_artifacts.zip']
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in extra})
    save_json(phase/'plan.json',dict(created=stamp(),candidate=key,hashes=hashes,infer=infer,replay=replay,
        stems=stems,total_jobs=3,estimated_minutes=55 if key=='c067' else 125,
        required_after_minutes=110,no_training=True,no_recipe_change=True,
        note='Reference numeric gates unchanged. Same local4070 portable inference of only missing own-domain movies plus2 historical smoke. Need separate local6jobs30min+remote60min+20margin after this phase.'))
    print(json.dumps({'prepared':key,'inputs':len(hashes),'movies':len(stems),'not_launched':True}),flush=True)

def assemble(key):
    out=candidate.root(key);phase=out/'local_recovery';original=c.read(out/'plan.json');plan=c.read(phase/'plan.json')
    local=out/'local';assert not (local/'plan.json').exists()
    c.verify_hashes(plan['hashes'])
    source=out/'e2e/local_recovery';ns=pilot.ns_for();rows=[];cases=[];extra=set()
    log=(source/'predict.log').read_text(encoding='utf-8')
    models=[ast.literal_eval(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('C037_PRIMARY ')]
    hardware=[ast.literal_eval(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('LOCAL_RUNTIME ')]
    assert models and hardware
    for value in models:
        assert value['train_embryo']==candidate.CONFIG[key] and value['step']==600 and value['alpha']==1.0
        assert Path(value['path']).resolve()==(out/'dataset/transformer_mean600.pt').resolve()
    for value in hardware:
        assert '4070' in value['gpu'] and value['math_sdp']
        assert not any(value[n] for n in ['cudnn_tf32','matmul_tf32','flash_sdp','mem_efficient_sdp'])
    for job in original['jobs']:
        for stem in job['stems']:
            old,old_final=c.historical(stem,job['split']);reuse=stem in original['reused_stems']
            new_cache=source/'edge_cache'/(stem+'.npz')
            if stem in original['remote_stems']:
                with np.load(new_cache) as a,np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as b:
                    for n in ['coords','low_coords','low_score']:assert np.array_equal(a[n],b[n],equal_nan=True),(stem,n)
            if stem in original['smoke_stems']:
                with np.load(new_cache) as a,np.load(old/'edge_cache'/(stem+'.npz')) as b:
                    assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n],equal_nan=True) for n in a.files),(stem,'full cache')
                a,b=[c.load_raw_graph(ns,p) for p in [pilot.graph_path(source,stem),pilot.graph_path(old,stem)]]
                sort=lambda e:(e['source_id'],e['target_id'])
                assert a[0]==b[0] and sorted(a[1],key=sort)==sorted(b[1],key=sort),(stem,'raw ILP')
                with np.load(phase/'graphs/same_device'/(stem+'_final.npz')) as a,np.load(old_final) as b:
                    assert all(np.array_equal(a[n],b[n]) for n in ['ids','txyz','edges']),(stem,'final')
                rows.append(dict(stem=stem,exact_all_cache=True,exact_ilp=True,exact_final=True))
            cache=old/'edge_cache'/(stem+'.npz') if reuse else new_cache
            with np.load(cache) as a,np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as b:
                for n in ['coords','low_coords','low_score']:assert np.array_equal(a[n],b[n],equal_nan=True),(stem,n)
            raw=pilot.graph_path(old if reuse else source,stem)
            graph=old_final if reuse else phase/'graphs/same_device'/(stem+'_final.npz')
            canonical=out/'e2e'/job['split'];(canonical/'edge_cache').mkdir(parents=True,exist_ok=True);(canonical/'predictions').mkdir(exist_ok=True)
            shutil.copy2(cache,canonical/'edge_cache'/(stem+'.npz'));shutil.copytree(raw,canonical/'predictions'/(stem+'.geff'))
            reference=out/'reference_graphs'/('combined_'+job['split'])/(stem+'_final.npz');reference.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(graph,reference)
            target=out/'graphs'/('combined_'+job['split'])/reference.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(reference,target)
            cases.append(dict(stem=stem,split=job['split'],embryo=stem[:4],reused=reuse,graph=str(reference.relative_to(ROOT)),raw_source=str(raw.relative_to(ROOT)),cache_source=str(cache.relative_to(ROOT))))
            extra.update([cache,graph]);extra.update(p for p in raw.rglob('*') if p.is_file())
    assert len(cases)==97 and len(rows)==2
    save_json(out/'smoke.json',dict(status='passed',rows=rows,exact_frozen_detector_movies=97,same_local_device_recovery=True,remote_numeric_cache_not_reused=True,models=models,hardware=hardware))
    inputs={Path(__file__),phase/'plan.json',out/'smoke.json',out/'plan.json',local/'launch.ps1',ROOT/'tools/launch_c067_local.ps1',ROOT/'src/c067_collect.py',ROOT/'src/final_four_download.py'}|extra
    for folder in [out/'e2e',out/'reference_graphs',phase/'graphs',c.evaluate_local.VENDOR_SRC]:
        inputs.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes=dict(plan['hashes']);hashes.update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
    save_json(local/'plan.json',dict(created=stamp(),hashes=hashes,cases=cases,total_jobs=6,estimated_minutes=30,reused_movies=len(original['reused_stems']),new_movies=len(original['new_stems']),no_new_inference=True,same_device_recovery=True,scientific_evidence='Own embryo fit-domain; opposite component transfer. All97 localdevice outputs with actual exact detector/cache/ILP controls.'))

def run(key):
    out=candidate.root(key);phase=out/'local_recovery';plan=c.read(phase/'plan.json');assert not (phase/'status.json').exists()
    decision=c.read(phase/'schedule_decision.json');assert decision['status']=='launch_same_device_recovery'
    assert c.read(ROOT/'experiments/candidates/c066_expanded_ordinary/transfer97/status.json')['status']=='complete_review_required'
    dep=ROOT/'experiments/candidates/c066_expanded_ordinary/deploy/status.json'
    assert not dep.exists() or c.read(dep)['status'] not in ['running','initializing']
    remaining=(datetime.fromisoformat('2026-09-30T00:00:00+09:00')-datetime.now().astimezone()).total_seconds()/60
    assert remaining>plan['estimated_minutes']+plan['required_after_minutes']
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset');os.environ['PYTHONUTF8']='1';q=Queue(phase,3)
    try:
        q.state.update(total_jobs=3,plan_sha256=sha(phase/'plan.json'));q.save();c.verify_hashes(plan['hashes'])
        q.run('same_device_infer',plan['infer']);q.run('same_device_replay',plan['replay'])
        q.run('exact_assemble',[sys.executable,'-X','utf8','-u',Path(__file__),'assemble','--candidate',key])
        c.verify_hashes(plan['hashes']);q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','assemble','run']);p.add_argument('--candidate',choices=list(candidate.CONFIG),required=True);a=p.parse_args();globals()[a.command](a.candidate)
