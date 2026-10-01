#!/usr/bin/env python3
"""One fixed C052 mean model; reuse C041 averaging and C042 portable machinery."""
from __future__ import annotations
import argparse
import contextlib
import inspect
import io
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import c041_fixed_models as fixed
import build_fixed_model_candidates as portable
import c052_division_transformer as pilot
import c038_followup_local as audit
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,BASE,CONTROL,OLD,sha,save_json,stamp
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue,replay_command,extension_stems
from eval_pp_variants_local import build_namespace,load_raw_graph,STEM_SETS

DEST=ROOT/'experiments/candidates/c053_fixed_division_transformer'
STUDY=pilot.DEST
EXT=STUDY/'extension75'
DATASET='taeyangg4/biohub-c053-division-transformer'
SLUG='biohub-c053-fixed-division-transformer'


def nbpath(out):return out/(SLUG+'.ipynb')


def average_model(out):
    # Reuse the actual original C041 averaging block; only its obsolete
    # movie-ID-disjoint assertion is removed for explicit fit-domain deployment.
    source=inspect.getsource(fixed.prepare)
    start=source.index('    checkpoints=');end=source.index("    save_json(out/'preflight.json'",start)
    block=textwrap.dedent(source[start:end])
    obsolete=';assert not training_tf & official'
    assert block.count(obsolete)==1
    block=block.replace(obsolete,'')
    scope=dict(fixed.__dict__,C037=STUDY,out=out,inputs=set())
    exec(compile(block,str(Path(__file__))+'::C041_mean_block','exec'),scope)
    result=out/'transformer_mean600.pt'
    states=scope['states'];mean=torch.load(result,map_location='cpu',weights_only=True)
    assert mean['train_embryo']=='fixed_both_no_routing'
    assert all(s.startswith('44b6') for s in states[0]['train_movies'])
    assert all(s.startswith('6bba') for s in states[1]['train_movies'])
    assert mean['train_movies']==sorted(set(states[0]['train_movies'])|set(states[1]['train_movies']))
    save_json(out/'mean_provenance.json',dict(source='unchanged C041 same-base arithmetic mean block',
        source_sha256=sha(ROOT/'src/c041_fixed_models.py'),source_models={g:sha(STUDY/'models'/f'{g}_step600.pt') for g in pilot.GROUPS},
        model_sha256=sha(result),train_movies=len(mean['train_movies']),train_embryo=mean['train_embryo'],
        local_evaluation='FIT-DOMAIN technical evidence,NOT independent validation;C052 opposite-embryo97 supplies component transfer evidence',
        change='Removed obsolete movie-ID-disjoint assertion explicitly;do not claim pooled model is held out',no_new_training=True))


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'plan.json').exists()
    assert read(EXT/'REVIEW_HASH_VERIFICATION.json')['status']=='passed'
    assert read(EXT/'decision.json')['status']=='advance_one_fixed_deployment'
    average_model(out)
    assets=out/'dataset';assets.mkdir(exist_ok=True)
    shutil.copy2(out/'transformer_mean600.pt',assets/'transformer_mean600.pt')
    asset_hashes={'transformer_mean600.pt':sha(assets/'transformer_mean600.pt')}
    save_json(assets/'dataset-metadata.json',dict(title='Biohub C053 division Transformer',id=DATASET,licenses=[{'name':'CC0-1.0'}]))
    save_json(assets/'provenance.json',dict(created=stamp(),model=read(out/'mean_provenance.json'),hashes=asset_hashes,
        policy='One fixed model for EVERY movie,no biological-prefix routing,no GT at inference. No new training.'))
    # Reuse the already verified C042 patch verbatim; C053 derives from C023.
    c042=portable.notebook('c042');old_nb=read(c042)
    old_code=''.join(old_nb['cells'][4]['source'])
    block=portable.BEGIN+old_code.split(portable.BEGIN,1)[1].split(portable.END,1)[0]+portable.END
    nb=read(BASE);cells=nb['cells']
    cells[0]['source']=''.join(cells[0]['source']).replace("'''Biohub C023:","'''Biohub C053:",1).splitlines(keepends=True)
    assets_source=portable.asset_source(asset_hashes).replace('biohub-fixed-models-c041','biohub-c053-division-transformer')
    assert 'biohub-fixed-models-c041' not in assets_source
    cells[2]['source']=(''.join(cells[2]['source'])+assets_source).splitlines(keepends=True)
    code=''.join(cells[4]['source']);anchor='_ps.write_text(_trial_source)\n';assert code.count(anchor)==1
    cells[4]['source']=code.replace(anchor,anchor+'\n'+block,1).splitlines(keepends=True)
    assert cells[5]['source']==read(BASE)['cells'][5]['source']
    for i,c in enumerate(cells):
        if c['cell_type']=='code':
            code=''.join(c['source']);compile(code,f'C053:{i}','exec')
            assert 'H:/' not in code and 'H:\\' not in code and 'from c052' not in code
            c['outputs']=[];c['execution_count']=None
    nb['metadata']['title']=SLUG;nbpath(out).write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
    meta=read(BASE.parent/'kernel-metadata.json');meta.update(id='taeyangg4/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    meta['dataset_sources']=sorted(set(meta['dataset_sources'])|{DATASET})
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
    save_json(out/'kernel-metadata.json',meta)
    repo=out/'tracking_repo'
    for part in ['src','scripts']:shutil.copytree(portable.SOURCE/part,repo/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    ps=repo/'scripts/predict_unet_transformer.py'
    exec(compile(block,'C053:actual_embedded_patch','exec'),dict(_ps=ps,os=os,_fixed_asset=lambda name:assets/name))
    # Numerical/runtime source parity with actual prior C052 inference.
    source=ps.read_text(encoding='utf-8')
    for q in ["Path('/kaggle/working')",'Path("/kaggle/working")']:
        source=source.replace(q,"Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    reference=STUDY/'e2e/division600_44b6/_work/tracking_repo/scripts/predict_unet_transformer.py'
    assert source==reference.read_text(encoding='utf-8')
    assert (repo/'scripts/c037_transformer_runtime.py').read_text(encoding='utf-8')==(ROOT/'src/c037_transformer_runtime.py').read_text(encoding='utf-8')
    save_json(out/'variants.json',{'as_configured':{}})
    extension,_=extension_stems({s for _,s in evaluation_plan()});jobs=[]
    for split in ['heldout12','confirm10','extension75']:
        stems=extension if split=='extension75' else [s for sp,s in evaluation_plan() if sp==split]
        (out/(split+'.txt')).write_text('\n'.join(stems)+'\n',encoding='utf-8')
        local=read(nbpath(out));s=''.join(local['cells'][5]['source']);anchor='\nwrite_test_submission("base")\n';assert s.count(anchor)==1
        add='\nfrom c052_division_transformer import install_stage_audit\n'+f'install_stage_audit(globals(),{str(out/"graphs"/("combined_"+split))!r})\n'
        local['cells'][5]['source']=s.replace(anchor,add+anchor,1).splitlines(keepends=True)
        (out/('audit_'+split+'.ipynb')).write_text(json.dumps(local,indent=1),encoding='utf-8')
        jobs.append(dict(split=split,stems=stems))
    hashes=dict(read(EXT/'plan.json')['hashes'])
    for name,digest in read(EXT/'artifact_hashes.json').items():hashes[str((EXT/name).relative_to(ROOT))]=digest
    inputs={Path(__file__),ROOT/'src/c041_fixed_models.py',ROOT/'src/build_fixed_model_candidates.py',ROOT/'src/evaluate_local.py',
        EXT/'REVIEW_HASH_VERIFICATION.json',EXT/'FINAL_REVIEW.md',EXT/'decision.json',EXT/'artifact_hashes.json',reference,c042}
    inputs.update(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in sorted(inputs):hashes[str(p.relative_to(ROOT))]=sha(p)
    save_json(out/'plan.json',dict(created=stamp(),jobs=jobs,hashes=hashes,total_jobs=13,max_start_job_hours=5,estimated_minutes=145,
        estimate_basis='Measured C05297 inference~102min/replay~20min plus2movie reference,portable12/writer4/evaluator and hash checks',
        model=str((assets/'transformer_mean600.pt').relative_to(ROOT)),model_sha256=asset_hashes['transformer_mean600.pt'],
        smoke_stems=[STEM_SETS['vis4'][0],STEM_SETS['vis4'][2]],candidate='c053',
        policy='One fixed equal-weight mean,unchanged C023 settings. FIT-DOMAIN97 local checks;C052 opposite folds supply scientific evidence. No Kaggle writes in queue.'))
    print('Prepared C053 fixed model+portable notebook;13jobs;',len(hashes),'pinned inputs',flush=True)


def infer(out,label,stems,reference=False):
    cmd=[sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py',
        '--repo',STUDY/'tracking_repo' if reference else out/'tracking_repo',
        '--notebook',BASE if reference else nbpath(out),'--stems',','.join(stems),
        '--out',out/'e2e','--label',label,'--v1284-mode','candidate','--v1284-head',pilot.old.HEAD,
        '--env',f'BIOHUB_C037_CHECKPOINT={out/"dataset/transformer_mean600.pt"}',
        '--env','BIOHUB_C037_ALPHA=1.0','--env','BIOHUB_C037_CAPTURE_DIR=','--env','BIOHUB_CACHE_EDGE_THRESHOLD=0.02']
    if reference:cmd+=['--t4-fp32']
    return cmd


def check_smoke(out):
    ns=pilot.ns_for();rows=[]
    for stem in read(out/'plan.json')['smoke_stems']:
        a,b=[out/'e2e'/s for s in ['reference_smoke','heldout12']]
        with np.load(a/'edge_cache'/(stem+'.npz')) as x,np.load(b/'edge_cache'/(stem+'.npz')) as y:
            assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.files)
        ga,gb=[load_raw_graph(ns,pilot.graph_path(r,stem)) for r in [a,b]]
        key=lambda e:(e['source_id'],e['target_id'])
        assert ga[0]==gb[0] and sorted(ga[1],key=key)==sorted(gb[1],key=key)
        rows.append(dict(stem=stem,exact_all_cache_arrays=True,exact_ilp=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def analyse(out):
    plan=read(out/'plan.json');ns=pilot.ns_for()
    data=pd.concat([pd.read_csv(out/'replay'/(j['split']+'.csv')) for j in plan['jobs']]);assert len(data)==97 and data.stem.is_unique
    baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{s}.csv') for s in ['heldout12','confirm10','extension75']])
    cross=pd.read_csv(EXT/'official_per_movie97.csv');assert set(data.stem)==set(baseline.stem)==set(cross.stem)
    for j in plan['jobs']:
        expected_checkpoint=str(out/'dataset/transformer_mean600.pt')
        log=(out/'e2e'/j['split']/'predict.log').read_text(encoding='utf-8')
        assert 'C037_PRIMARY' in log and 'fixed_both_no_routing' in log
        assert expected_checkpoint.replace('\\','\\\\') in log or expected_checkpoint in log
        for stem in j['stems']:
            with np.load(out/'e2e'/j['split']/'edge_cache'/(stem+'.npz')) as a,np.load(CONTROL/('control_fp32_'+j['split'])/'edge_cache'/(stem+'.npz')) as b:
                for k in ['coords','low_coords','low_score']:assert np.array_equal(a[k],b[k],equal_nan=True),('frozen detector',stem,k)
    rows=[]
    sets={'all97':set(data.stem),'all22':{s for _,s in evaluation_plan()},
        **{j['split']:set(j['stems']) for j in plan['jobs']},**{g:{s for s in data.stem if s.startswith(g)} for g in pilot.GROUPS}}
    for group,stems in sets.items():
        a,b,c=[ns['aggregate_official'](f[f.stem.isin(stems)].to_dict('records')) for f in [data,baseline,cross]]
        delta=data.set_index('stem').loc[sorted(stems),'adjusted_edge_jaccard']-baseline.set_index('stem').loc[sorted(stems),'adjusted_edge_jaccard']
        rows.append(dict(group=group,movies=len(stems),score=a['proxy_score'],delta_vs_C023=a['proxy_score']-b['proxy_score'],
            edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],delta_vs_C052_opposite_folds=a['proxy_score']-c['proxy_score'],
            div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn'],wins=int((delta>1e-10).sum()),losses=int((delta< -1e-10).sum())))
    data.to_csv(out/'official_per_movie97.csv',index=False);pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    visible=ns['aggregate_official'](data[data.stem.isin(STEM_SETS['vis4'])].to_dict('records'))
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,visible4_expected=visible['proxy_score'],exact_frozen_detector_movies=97,
        warning='Fixed mean contains a same-embryo-trained member. These97 scores are fit-domain/technical evidence,NOT independent validation. Use previous C052 whole-embryo evidence and real LB only.'))


def write_visible(out):
    # Existing production notebook writer and exact saved-graph checks.
    portable.STUDY=out;portable.ARMS={'c053':('division-transformer','final')};portable.notebook=lambda key:nbpath(out)
    portable.write_visible(out,'c053')


def verify_local(out):
    original=pd.read_csv(out/'replay/heldout12.csv');actual=pd.read_csv(out/'replay/portable_heldout12.csv')
    audit.verify(actual,original,{'as_configured':'as_configured'})
    evaluation=read(out/'c053/evaluation/summary.json');expected=read(out/'analysis.json')['visible4_expected']
    assert abs(evaluation['total_score']-expected)<1e-12
    assert read(out/'smoke.json')['status']=='passed' and read(out/'c053/writer_parity.json')['status']=='passed'
    save_json(out/'local_verification.json',dict(status='passed',created=stamp(),candidate='c053',
        notebook_sha256=sha(nbpath(out)),model_sha256=sha(out/'dataset/transformer_mean600.pt'),
        csv_sha256=sha(out/'c053/submission.csv'),local_visible4=expected,portable12='exact',writer4='exact',
        repair_fallback=0,deadline_degraded=0,actual_t4_verified=False,
        limitation='Actual T4 verification and scientific review of fixed97 required before competition submission.'))


def run(out):
    assert not (out/'status.json').exists(),'No blind restart'
    plan=read(out/'plan.json');os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset')
    os.environ['PYTHONUTF8']='1'  # The reused portable notebook writes UTF-8 source.
    q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(out/'plan.json'));q.save();verify_hashes(plan['hashes'])
        q.run('reference_smoke',infer(out,'reference_smoke',plan['smoke_stems'],True))
        for j in plan['jobs']:
            split=j['split'];q.run('infer_'+split,infer(out,split,j['stems']))
            if split=='heldout12':q.run('exact_portable_reproduction',[sys.executable,'-u',Path(__file__),'check_smoke','--out',out])
            q.run('replay_'+split,replay_command(out/('audit_'+split+'.ipynb'),out/'e2e'/split,j['stems'],out/'variants.json',out/'replay'/(split+'.csv')))
        q.run('fixed97_analysis',[sys.executable,'-u',Path(__file__),'analyse','--out',out])
        q.run('actual_portable_replay12',replay_command(nbpath(out),out/'e2e/heldout12',plan['jobs'][0]['stems'],out/'variants.json',out/'replay/portable_heldout12.csv'))
        q.run('production_writer4',[sys.executable,'-u',Path(__file__),'write_visible','--out',out])
        q.run('official_visible4',[sys.executable,'-u',ROOT/'src/evaluate_local.py','--csv',out/'c053/submission.csv','--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/'c053/evaluation'])
        q.run('verify_local',[sys.executable,'-u',Path(__file__),'verify_local','--out',out]);verify_hashes(plan['hashes'])
        files=[p for folder in ['replay','graphs','e2e','c053'] for p in (out/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name not in ['plan.json','status.json','launch.json','artifact_hashes.json','automation_confirmation.json']]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        assert len(q.state['jobs'])==plan['total_jobs'];q.close('complete_local_verified_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','check_smoke','analyse','write_visible','verify_local']);p.add_argument('--out',type=Path,default=DEST)
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT);globals()[a.command](out)
