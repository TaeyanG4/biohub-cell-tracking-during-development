#!/usr/bin/env python3
"""Bounded C024 + fixed C041 models: existing inference, replay and Queue only."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from build_fixed_model_candidates import ROOT, DEST as PORTABLE, STUDY, BEGIN, END, notebook
from c041_fixed_models import sha, save_json, stamp
from c037_transformer_study import HEAD, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS
from c038_followup_local import verify, graph, edge_audit
from eval_pp_variants_local import build_namespace, load_raw_graph, STEM_SETS
from run_last_days_local import Queue, replay_command
from v1284_capture_local import embedded_v1284_module

DEST=ROOT/'experiments/candidates/c044_c024_fixed_study'
BASE=ROOT/'experiments/candidates/c024_head_ensemble/biohub-c024-head-ensemble.ipynb'
SOURCE=ROOT/'tmp/c024_output/tracking_repo'
HEAD_V1=ROOT/'experiments/candidates/c012_v1284_head/heads/head_v1.pt'
ARMS={'c044':'transformer','c045':'transformer-appearance'}


def candidate(key):
    return ROOT/'experiments/candidates'/f'{key}_fixed_ensemble_{ARMS[key].replace("-","_")}'


def nbpath(key):
    return candidate(key)/f'biohub-{key}-ensemble-{ARMS[key]}.ipynb'


def prepare(out):
    if (out/'plan.json').exists():raise RuntimeError('Already registered; do not overwrite')
    out.mkdir(parents=True,exist_ok=True)
    base=json.loads(BASE.read_text(encoding='utf-8'))
    refs={k:json.loads(notebook(k).read_text(encoding='utf-8')) for k in ['c042','c043']}
    source_model=embedded_v1284_module(BASE)
    assert source_model==(SOURCE/'scripts/v1284_coordinate_refinement.py').read_text(encoding='utf-8')
    assert sha(HEAD_V1)=='9d3484f794b48c379b657714878ff3d7bee6042dd932ef257fced992b34edda6'
    assert sha(HEAD)=='625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c'
    assert ''.join(base['cells'][5]['source'])==''.join(refs['c042']['cells'][5]['source'])
    portable=json.loads((PORTABLE/'plan.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in portable['hashes'].items()),'portable source drift'
    block=BEGIN+''.join(refs['c042']['cells'][4]['source']).split(BEGIN,1)[1].split(END,1)[0]+END
    repo=out/'tracking_repo'
    for part in ['scripts','src']:
        shutil.copytree(PORTABLE/'tracking_repo'/part,repo/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    (repo/'scripts/v1284_coordinate_refinement.py').write_text(source_model,encoding='utf-8')
    files={ROOT/p for p in portable['hashes'] if p.endswith(('.py','.pt','.ipynb','.json'))}
    files.update([Path(__file__),BASE,HEAD_V1,HEAD,PRIMARY_WEIGHTS,SECONDARY_WEIGHTS,
                  ROOT/'src/c038_followup_local.py',ROOT/'src/evaluate_local.py'])
    for key,ref in [('c044','c042'),('c045','c043')]:
        dest=candidate(key);dest.mkdir(parents=True,exist_ok=True)
        nb=json.loads(BASE.read_text(encoding='utf-8'))
        nb['cells'][0]['source']=''.join(nb['cells'][0]['source']).replace('Biohub C024:',f'Biohub {key.upper()}:',1).splitlines(keepends=True)
        nb['cells'][2]['source']=refs[ref]['cells'][2]['source']
        code=''.join(nb['cells'][4]['source']);anchor='_ps.write_text(_trial_source)\n';assert code.count(anchor)==1
        nb['cells'][4]['source']=code.replace(anchor,anchor+'\n'+block,1).splitlines(keepends=True)
        nb['cells'][5]['source']=refs[ref]['cells'][5]['source']
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'{key}:{i}','exec');c['outputs']=[];c['execution_count']=None
        path=nbpath(key);nb['metadata']['title']=path.stem
        path.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
        meta=json.loads((BASE.parent/'kernel-metadata.json').read_text());meta.update(id='taeyangg4/'+path.stem,title=path.stem,code_file=path.name)
        meta['dataset_sources']=sorted(set(meta['dataset_sources'])|{'taeyangg4/biohub-fixed-models-c041'})
        save_json(dest/'kernel-metadata.json',meta);files.update([path,dest/'kernel-metadata.json'])
        (dest/'README.md').write_text(f'# {key.upper()} C024 plus fixed {ARMS[key]}\n\nPrepared for finite local study c044_c024_fixed_study. Unvalidated; no push/submission. Scored C024 head ensemble retained, C041 fixed models reused. New candidate must improve its own matching C024 control across validation groups and be compared with C042/C043 before spending reserved capacity. Actual notebook replay/T4 parity required.\n',encoding='utf-8')
        # Execute the same embedded inference patch with C024's original predictor.
        temp=out/f'patch_check_{key}';temp.mkdir(exist_ok=True)
        ps=temp/'predict_unet_transformer.py';ps.write_text((SOURCE/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8'),encoding='utf-8')
        exec(compile(block,f'{key}:patch','exec'),dict(_ps=ps,os=os,_fixed_asset=lambda name:PORTABLE/'dataset'/name))
        assert ps.read_text(encoding='utf-8')==(repo/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
        assert (temp/'c037_transformer_runtime.py').read_text(encoding='utf-8')==(repo/'scripts/c037_transformer_runtime.py').read_text(encoding='utf-8')
    save_json(out/'off.json',{'off':{'C038_MODE':'off'}})
    save_json(out/'variants.json',{'off':{'C038_MODE':'off'},'appearance':{'C038_MODE':'appearance'}})
    save_json(out/'as_configured.json',{'as_configured':{}})
    # Separate, visibly local passive-audit notebook. Never uploaded.
    nb=json.loads(nbpath('c045').read_text());code=''.join(nb['cells'][5]['source'])
    anchor='\nwrite_test_submission("base")\n'
    payload="\nfrom c038_followup_local import install_graph_audit\ninstall_graph_audit(globals(), os.environ['BIOHUB_C044_GRAPH_DIR'])\n"
    nb['cells'][5]['source']=code.replace(anchor,payload+anchor,1).splitlines(keepends=True)
    (out/'local_graph_audit.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
    for split in ['heldout12','confirm10']:
        shutil.copy2(STUDY/(split+'.txt'),out/(split+'.txt'))
        files.add(STUDY/'replay'/f'combined_{split}.csv')
    # Pin exact production assets, references and GT (raw movies remain read-only).
    for part in [repo,SOURCE]:files.update(p for p in part.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for split in ['heldout12','confirm10']:
        for stem in (out/(split+'.txt')).read_text().split():
            files.update(p for p in (ROOT/'data/train'/f'{stem}.geff').rglob('*') if p.is_file())
    files.update(p for p in out.iterdir() if p.is_file())
    save_json(out/'plan.json',dict(created=stamp(),hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        arms=['c024_control','c044_transformer','c045_combined'],smoke_stems=[STEM_SETS['vis4'][0],STEM_SETS['vis4'][2]],
        policy='Fixed C024 heads and C041 models/thresholds; no training, sweeps or routing. Compare actual22 own C024 and existing C042/C043. Reserved slots not automatically consumed.',
        continuation='Advance only supported positive own-control split/embryo edge/total and complementary advantage; if unclear resolve with fixed extension rather than submit to fill two slots.'))
    print('Prepared C044/C045 C024 fixed-model study; portable candidates unvalidated',flush=True)


def inference(out,label,stems,reference=False,learned=False):
    checkpoint=str(PORTABLE/'dataset/transformer_mean600.pt') if learned else ''
    command=[sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py','--repo',SOURCE if reference else out/'tracking_repo',
        '--notebook',BASE if reference else nbpath('c044'),'--stems',','.join(stems),'--out',out/'e2e','--label',label,
        '--v1284-mode','candidate','--v1284-head',str(HEAD_V1)+';'+str(HEAD),
        '--env','BIOHUB_C037_CHECKPOINT='+checkpoint]
    command+=['--env','BIOHUB_C037_ALPHA=1.0','--env','BIOHUB_C037_CAPTURE_DIR=']
    if reference:command+=['--t4-fp32']
    return command


def check_smoke(out):
    plan=json.loads((out/'plan.json').read_text());rows=[]
    with contextlib.redirect_stdout(io.StringIO()):ns=build_namespace(BASE,{},out/'e2e/control_heldout12/edge_cache')
    for stem in plan['smoke_stems']:
        runs=[out/'e2e'/s for s in ['reference_smoke','control_heldout12']]
        with np.load(runs[0]/'edge_cache'/f'{stem}.npz') as a,np.load(runs[1]/'edge_cache'/f'{stem}.npz') as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files),('off cache drift',stem)
        a,b=[load_raw_graph(ns,next((p/'predictions').rglob(stem+'.geff'))) for p in runs]
        key=lambda e:(e['source_id'],e['target_id'])
        assert a[0]==b[0] and sorted(a[1],key=key)==sorted(b[1],key=key),('off ILP drift',stem)
        rows.append(dict(stem=stem,exact_cache=True,exact_ilp=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def analyse(out):
    control=pd.concat([pd.read_csv(out/'replay'/f'control_{s}.csv') for s in ['heldout12','confirm10']])
    off=pd.concat([pd.read_csv(out/'replay'/f'control_wrapped_{s}.csv') for s in ['heldout12','confirm10']])
    verify(off,control,{'off':'as_configured'})
    data=pd.concat([pd.read_csv(out/'replay'/f'learned_{s}.csv') for s in ['heldout12','confirm10']])
    peer=pd.concat([pd.read_csv(STUDY/'replay'/f'combined_{s}.csv') for s in ['heldout12','confirm10']])
    with contextlib.redirect_stdout(io.StringIO()):ns=build_namespace(BASE,{},out/'e2e/control_heldout12/edge_cache')
    groups={'all22':set(control.stem),**{s:set((out/(s+'.txt')).read_text().split()) for s in ['heldout12','confirm10']},
        **{g:{s for s in control.stem if s.startswith(g)} for g in ['44b6','6bba']}}
    rows=[]
    for name,mode in [('c044','off'),('c045','appearance')]:
        for group,stems in groups.items():
            a=data[(data.config==mode)&data.stem.isin(stems)];b=control[control.stem.isin(stems)];c=peer[(peer.config==mode)&peer.stem.isin(stems)]
            assert len(a)==len(b)==len(c)==len(stems)
            va,vb,vc=[ns['aggregate_official'](f.to_dict('records')) for f in [a,b,c]]
            delta=a.set_index('stem').adjusted_edge_jaccard-b.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(arm=name,group=group,movies=len(stems),score=va['proxy_score'],control=vb['proxy_score'],
                delta=va['proxy_score']-vb['proxy_score'],edge_delta=va['adjusted_edge_jaccard']-vb['adjusted_edge_jaccard'],
                delta_vs_c042_or_c043=va['proxy_score']-vc['proxy_score'],edge_delta_vs_c042_or_c043=va['adjusted_edge_jaccard']-vc['adjusted_edge_jaccard'],
                wins=int((delta>1e-10).sum()),losses=int((delta< -1e-10).sum()),div_tp=va['div_tp'],div_fp=va['div_fp'],div_fn=va['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    import c038_followup_local as audit
    prior=audit.MODES
    try:
        audit.MODES=['off','appearance']
        audit.edge_audit(out,dict(jobs=[dict(name='learned_'+s,kind='learned',stems=sorted(groups[s])) for s in ['heldout12','confirm10']]))
    finally:audit.MODES=prior
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,exact_control_rows=22,
        note='22 repeatedly used train movies; source C024 head includes pretrained supervision. Not hidden-test proof. No reserved-slot submission from local-only result.'))


def run(out):
    plan=json.loads((out/'plan.json').read_text());assert all(sha(ROOT/p)==h for p,h in plan['hashes'].items()),'input drift'
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(PORTABLE/'dataset');q=Queue(out,6)
    try:
        q.state['plan_sha256']=sha(out/'plan.json');q.save()
        q.run('reference_smoke',inference(out,'reference_smoke',plan['smoke_stems'],reference=True))
        for split in ['heldout12','confirm10']:
            stems=(out/(split+'.txt')).read_text().split();label='control_'+split
            q.run('infer_'+label,inference(out,label,stems))
            if split=='heldout12':q.run('check_smoke',[sys.executable,'-u',Path(__file__),'check_smoke','--out',out])
            q.run('direct_'+label,replay_command(BASE,out/'e2e'/label,stems,out/'as_configured.json',out/'replay'/(label+'.csv')))
            os.environ['BIOHUB_C044_GRAPH_DIR']=str(out/'graphs'/label)
            q.run('wrapped_'+label,replay_command(out/'local_graph_audit.ipynb',out/'e2e'/label,stems,out/'off.json',out/'replay'/('control_wrapped_'+split+'.csv')))
        for split in ['heldout12','confirm10']:
            stems=(out/(split+'.txt')).read_text().split();label='learned_'+split
            q.run('infer_'+label,inference(out,label,stems,learned=True))
            os.environ['BIOHUB_C044_GRAPH_DIR']=str(out/'graphs'/label)
            q.run('replay_'+label,replay_command(out/'local_graph_audit.ipynb',out/'e2e'/label,stems,out/'variants.json',out/'replay'/(label+'.csv')))
        q.run('analyse',[sys.executable,'-u',Path(__file__),'analyse','--out',out])
        assert all(sha(ROOT/p)==h for p,h in plan['hashes'].items()),'input drift during run'
        files=[p for part in ['graphs','replay'] for p in (out/part).rglob('*') if p.is_file()]
        files += [p for folder in (out/'e2e').iterdir() for part in ['edge_cache','predictions'] for p in (folder/part).rglob('*') if p.is_file()]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files});q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','check_smoke','analyse'])
    p.add_argument('--out',type=Path,default=DEST);a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT);globals()[a.command](out)
