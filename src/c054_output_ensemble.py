#!/usr/bin/env python3
"""C054: bounded fixed prediction-ensemble deployment study using C053 tools."""
from __future__ import annotations
import argparse
import copy
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import c053_fixed_division_transformer as previous
import c037_transformer_study as training
import c054_transformer_runtime as runtime
from c047_hard_example_study import read, verify_hashes
from reid_probe_local import ROOT, BASE, sha, save_json, stamp
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue, replay_command

DEST = ROOT/'experiments/candidates/c054_output_ensemble'
PRIOR = previous.DEST
PILOT = previous.STUDY
SLUG = 'biohub-c054-output-ensemble'
ASSET = 'transformer_output_mean600.pt'
DATASET = 'taeyangg4/biohub-c054-output-ensemble'
IMPORT_OLD = 'from c037_transformer_runtime import capture as _c037_capture, apply_primary as _c037_apply'
IMPORT_NEW = 'from c054_transformer_runtime import capture as _c037_capture, apply_primary as _c037_apply'


def nbpath(out):
    return out/(SLUG+'.ipynb')


def reused(function, replacements=()):
    """Run the existing verification/analysis function with explicit C054 bindings."""
    source = inspect.getsource(function)
    for before, after in replacements:
        assert before in source, before
        source = source.replace(before, after)
    scope = dict(previous.__dict__, __file__=__file__, SLUG=SLUG, nbpath=nbpath)
    exec(compile(source, str(Path(__file__))+'::reuse_'+function.__name__, 'exec'), scope)
    return scope[function.__name__]


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'plan.json').exists(), 'Registered source is immutable'
    assert read(PRIOR/'REVIEW_HASH_VERIFICATION.json')['status'] == 'passed'
    members = [torch.load(PILOT/'models'/f'{g}_step600.pt', map_location='cpu', weights_only=True)
               for g in ['44b6', '6bba']]
    assert len({m['base_transformer_sha256'] for m in members}) == 1
    bundle = dict(combination='equal_raw_logit_mean', base_transformer_sha256=members[0]['base_transformer_sha256'],
                  members=members, step=600, train_embryo='fixed_both_no_routing')
    assets = out/'dataset'; assets.mkdir(exist_ok=True)
    torch.save(bundle, assets/ASSET)
    mean = torch.load(PRIOR/'dataset/transformer_mean600.pt', map_location='cpu', weights_only=True)
    torch.save(dict(bundle, members=[mean, mean]), out/'repeated_mean_control.pt')
    hashes = {ASSET:sha(assets/ASSET)}
    save_json(assets/'dataset-metadata.json', dict(title='Biohub C054 output ensemble', id=DATASET,
                                                licenses=[{'name':'CC0-1.0'}]))
    save_json(assets/'provenance.json', dict(created=stamp(), hashes=hashes,
        source_models={g:sha(PILOT/'models'/f'{g}_step600.pt') for g in ['44b6','6bba']},
        combination='Fixed equal raw-logit mean, both temporal directions, every movie',
        no_new_training=True, limitation='FIT-DOMAIN technical97; C052 opposite-embryo results supply component-transfer evidence'))
    nb = read(previous.nbpath(PRIOR)); original = read(BASE)
    nb['cells'][0]['source'] = ''.join(nb['cells'][0]['source']).replace('Biohub C053:', 'Biohub C054:', 1).splitlines(True)
    nb['cells'][2]['source'] = (''.join(original['cells'][2]['source']) +
        previous.portable.asset_source(hashes).replace('biohub-fixed-models-c041', 'biohub-c054-output-ensemble')).splitlines(True)
    code = ''.join(nb['cells'][4]['source'])
    assert code.count("_fixed_asset('transformer_mean600.pt')") == 1
    code = code.replace("_fixed_asset('transformer_mean600.pt')", f"_fixed_asset('{ASSET}')")
    extension = ("\n_c054_source = _ps.read_text(encoding='utf-8')\n"+
        f"assert _c054_source.count({IMPORT_OLD!r}) == 1\n"+
        f"_c054_source = _c054_source.replace({IMPORT_OLD!r}, {IMPORT_NEW!r}, 1)\n"+
        "compile(_c054_source, str(_ps), 'exec')\n_ps.write_text(_c054_source, encoding='utf-8')\n"+
        f"(_ps.parent/'c054_transformer_runtime.py').write_text({(ROOT/'src/c054_transformer_runtime.py').read_text(encoding='utf-8')!r}, encoding='utf-8')\n")
    assert code.count(previous.portable.END) == 1
    code = code.replace(previous.portable.END, extension+previous.portable.END, 1)
    nb['cells'][4]['source'] = code.splitlines(True)
    assert nb['cells'][5]['source'] == original['cells'][5]['source']
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            text=''.join(c['source']); compile(text,f'C054:{i}','exec')
            assert 'H:/' not in text and 'H:\\' not in text and 'from c052' not in text
            c['outputs']=[]; c['execution_count']=None
    nb['metadata']['title']=SLUG
    nbpath(out).write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
    meta=read(PRIOR/'kernel-metadata.json'); meta.update(id='taeyangg4/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    meta['dataset_sources']=[DATASET if s==previous.DATASET else s for s in meta['dataset_sources']]
    save_json(out/'kernel-metadata.json',meta)
    repo=out/'tracking_repo'
    for part in ['src','scripts']:
        shutil.copytree(previous.portable.SOURCE/part,repo/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    block=previous.portable.BEGIN+code.split(previous.portable.BEGIN,1)[1].split(previous.portable.END,1)[0]+previous.portable.END
    ps=repo/'scripts/predict_unet_transformer.py'
    exec(compile(block,'C054:actual_embedded_patch','exec'),dict(_ps=ps,os=os,_fixed_asset=lambda name:assets/name))
    expected=(PRIOR/'tracking_repo/scripts/predict_unet_transformer.py').read_text(encoding='utf-8').replace(IMPORT_OLD,IMPORT_NEW,1)
    assert ps.read_text(encoding='utf-8')==expected
    assert (repo/'scripts/c054_transformer_runtime.py').read_text(encoding='utf-8')==(ROOT/'src/c054_transformer_runtime.py').read_text(encoding='utf-8')
    save_json(out/'variants.json',{'as_configured':{}})
    jobs=read(PRIOR/'plan.json')['jobs']
    for j in jobs:
        split=j['split']; (out/(split+'.txt')).write_text('\n'.join(j['stems'])+'\n',encoding='utf-8')
        audit=copy.deepcopy(nb); text=''.join(audit['cells'][5]['source']); anchor='\nwrite_test_submission("base")\n'
        assert text.count(anchor)==1
        hook='\nfrom c052_division_transformer import install_stage_audit\n'+f'install_stage_audit(globals(),{str(out/"graphs"/("combined_"+split))!r})\n'
        audit['cells'][5]['source']=text.replace(anchor,hook+anchor,1).splitlines(True)
        (out/('audit_'+split+'.ipynb')).write_text(json.dumps(audit,indent=1),encoding='utf-8')
    pinned=dict(read(PRIOR/'plan.json')['hashes'])
    pinned.update({str((PRIOR/k).relative_to(ROOT)):v for k,v in read(PRIOR/'artifact_hashes.json').items()})
    inputs={Path(__file__),ROOT/'src/c054_transformer_runtime.py',PRIOR/'FINAL_REVIEW.md',PRIOR/'REVIEW_HASH_VERIFICATION.json',PRIOR/'REVIEW_COUNTS.json'}
    inputs.update(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in sorted(inputs):pinned[str(p.relative_to(ROOT))]=sha(p)
    save_json(out/'plan.json',dict(created=stamp(),jobs=jobs,hashes=pinned,total_jobs=14,max_start_job_hours=6,
        estimated_minutes=175,estimate_basis='C053 actual130min plus second-transformer forward overhead and cached-packet/repeated-member controls;allow45min margin',
        smoke_stems=read(PRIOR/'plan.json')['smoke_stems'],model_sha256=sha(assets/ASSET),
        policy='One fixed equal output mean;no model/threshold/weight selection or new fitting. FIT-DOMAIN97. No Kaggle writes in queue.'))
    print('Prepared C054:',len(pinned),'pinned inputs,14jobs',flush=True)


def packet_check(out):
    engine=training.engine(PILOT)
    full,_,_=engine.load_model(training.PRIMARY_WEIGHTS,torch.device('cuda'))
    base=full.transformer
    ensemble=runtime.load_members(base,torch.load(out/'dataset'/ASSET,map_location='cpu',weights_only=True))
    repeated=runtime.load_members(base,torch.load(out/'repeated_mean_control.pt',map_location='cpu',weights_only=True))
    mean=copy.deepcopy(base); mean.load_state_dict(torch.load(PRIOR/'dataset/transformer_mean600.pt',map_location='cpu',weights_only=True)['state_dict']);mean.eval()
    allowed={s for _,s in evaluation_plan()}
    packets=sorted({PILOT/'evaluation_features'/e['stem']/f"{e['t']:03d}_{e['t']+1:03d}.npz"
                    for e in read(PILOT/'event_audit.json')['events'] if e['stem'] in allowed})
    assert len(packets)==24
    rows=[]
    with torch.no_grad():
        for p in packets:
            with np.load(p) as z:packet={k:z[k].copy() for k in z.files}
            group=p.parent.name.split('_')[0]
            ref=ensemble.second if group=='44b6' else ensemble.first
            logits=training.forward(ref,packet).cpu().numpy()
            error=float(np.max(np.abs(logits[np.ix_(packet['probe_rows'],packet['probe_cols'])]-packet['probe_logits'])))
            assert error<2e-5
            delta=[]
            for reverse in [False,True]:
                data=packet if not reverse else dict(packet,feat_src=packet['feat_tgt'],feat_tgt=packet['feat_src'],coords_src=packet['coords_tgt'],coords_tgt=packet['coords_src'])
                avg=training.forward(ensemble,data)
                manual=(training.forward(ensemble.first,data)+training.forward(ensemble.second,data))*.5
                old=training.forward(mean,data);duplicate=training.forward(repeated,data)
                assert torch.equal(avg,manual) and torch.equal(old,duplicate)
                delta.append(float((avg-old).abs().mean()))
            rows.append(dict(packet=str(p.relative_to(ROOT)),saved_opposite_logit_error=error,forward_mean_abs_change=delta[0],reverse_mean_abs_change=delta[1]))
    save_json(out/'packet_check.json',dict(status='passed',packets=24,actual_saved_member_parity=True,forward_reverse_exact_manual_mean=True,repeated_mean_exact=True,rows=rows,
        interpretation='Numerical mechanism check only;not improvement evidence or inference routing.'))


def infer(out,label,stems,control=False):
    checkpoint=out/'repeated_mean_control.pt' if control else out/'dataset'/ASSET
    return [sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py','--repo',out/'tracking_repo','--notebook',nbpath(out),
        '--stems',','.join(stems),'--out',out/'e2e','--label',label,'--v1284-mode','candidate','--v1284-head',previous.pilot.old.HEAD,
        '--env',f'BIOHUB_C037_CHECKPOINT={checkpoint}','--env','BIOHUB_C037_ALPHA=1.0','--env','BIOHUB_C037_CAPTURE_DIR=',
        '--env','BIOHUB_CACHE_EDGE_THRESHOLD=0.02']


def check_smoke(out):
    reused(previous.check_smoke,[("a,b=[out/'e2e'/s for s in ['reference_smoke','heldout12']]",
                                  f"a,b=out/'e2e/repeated_mean_smoke',Path({str(PRIOR/'e2e/heldout12')!r})")])(out)


def analyse(out):
    reused(previous.analyse,[('transformer_mean600.pt',ASSET),("'C037_PRIMARY'","'C054_PRIMARY'")])(out)
    ns=previous.pilot.ns_for();a=pd.read_csv(out/'official_per_movie97.csv');b=pd.read_csv(PRIOR/'official_per_movie97.csv')
    result=read(out/'analysis.json');summary=read(out/'plan.json');groups={'all97':set(a.stem),'all22':{s for _,s in evaluation_plan()},
        **{j['split']:set(j['stems']) for j in summary['jobs']},**{g:{s for s in a.stem if s.startswith(g)} for g in ['44b6','6bba']}}
    for row in result['rows']:
        stems=groups[row['group']];x,y=[ns['aggregate_official'](d[d.stem.isin(stems)].to_dict('records')) for d in [a,b]]
        row['delta_vs_C053']=x['proxy_score']-y['proxy_score']; row['edge_delta_vs_C053']=x['adjusted_edge_jaccard']-y['adjusted_edge_jaccard']
        row['counts_delta_vs_C053']={k:int(a[a.stem.isin(stems)][k].sum()-b[b.stem.isin(stems)][k].sum()) for k in ['edge_tp','edge_fp','edge_fn','div_tp','div_fp','div_fn']}
    result['warning']='Fixed two-member prediction ensemble contains a same-embryo-trained member. FIT-DOMAIN97;C052 oppositefolds supply component evidence.'
    save_json(out/'analysis.json',result);pd.DataFrame(result['rows']).to_csv(out/'official_summary.csv',index=False)


def write_visible(out):
    reused(previous.write_visible,[('c053','c054')])(out)


def verify_local(out):
    reused(previous.verify_local,[('c053','c054'),('transformer_mean600.pt',ASSET)])(out)
    assert read(out/'packet_check.json')['status']=='passed'


def run(out):
    assert not (out/'status.json').exists(),'No blind restart'
    plan=read(out/'plan.json');os.environ.update(BIOHUB_FIXED_ASSET_DIR=str(out/'dataset'),PYTHONUTF8='1')
    q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=14,plan_sha256=sha(out/'plan.json'));q.save();verify_hashes(plan['hashes'])
        command=lambda verb:[sys.executable,'-u',Path(__file__),verb,'--out',out]
        q.run('packet_check',command('packet_check'))
        q.run('repeated_mean_smoke',infer(out,'repeated_mean_smoke',plan['smoke_stems'],True))
        q.run('exact_repeated_mean_reproduction',command('check_smoke'))
        for j in plan['jobs']:
            split=j['split'];q.run('infer_'+split,infer(out,split,j['stems']))
            q.run('replay_'+split,replay_command(out/('audit_'+split+'.ipynb'),out/'e2e'/split,j['stems'],out/'variants.json',out/'replay'/(split+'.csv')))
        q.run('fixed97_analysis',command('analyse'))
        q.run('actual_portable_replay12',replay_command(nbpath(out),out/'e2e/heldout12',plan['jobs'][0]['stems'],out/'variants.json',out/'replay/portable_heldout12.csv'))
        q.run('production_writer4',command('write_visible'))
        q.run('official_visible4',[sys.executable,'-u',ROOT/'src/evaluate_local.py','--csv',out/'c054/submission.csv','--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/'c054/evaluation'])
        q.run('verify_local',command('verify_local'));verify_hashes(plan['hashes'])
        files=[p for folder in ['replay','graphs','e2e','c054'] for p in (out/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name not in ['plan.json','status.json','launch.json','artifact_hashes.json','automation_confirmation.json']]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        assert len(q.state['jobs'])==14;q.close('complete_local_verified_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','packet_check','check_smoke','analyse','write_visible','verify_local','run']);p.add_argument('--out',type=Path,default=DEST)
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT);globals()[a.command](out)
