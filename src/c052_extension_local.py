#!/usr/bin/env python3
"""Fixed C05275 extension: existing inference, replay, capture and Queue only."""
from __future__ import annotations
import argparse
import collections
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import c052_division_transformer as pilot
import c037_transformer_study as training
import c038_followup_local as audit
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,BASE,CONTROL,OLD,sha,save_json,stamp
from run_last_days_local import Queue,extension_stems
from reid_augmented_local import evaluation_plan
from eval_pp_variants_local import load_raw_graph

PILOT=pilot.DEST
DEST=PILOT/'extension75'
PREVIOUS=ROOT/'experiments/candidates/c040_late600_combo/extension75'


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'plan.json').exists()
    assert read(PILOT/'REVIEW_HASH_VERIFICATION.json')['status']=='passed'
    assert read(PILOT/'status.json')['status']=='complete_review_required'
    source=read(PILOT/'plan.json');verify_hashes(source['hashes'])
    hashes=dict(source['hashes'])
    # Bind ALL completed pilot artifacts, including folds, packets and exact graphs.
    for name,digest in read(PILOT/'artifact_hashes.json').items():
        hashes[str((PILOT/name).relative_to(ROOT))]=digest
    stems,lists=extension_stems({s for _,s in evaluation_plan()});hashes.update(lists)
    inputs={Path(__file__),ROOT/'src/c052_division_survival.py',out/'README.md',
        PILOT/'plan.json',PILOT/'status.json',PILOT/'artifact_hashes.json',PILOT/'PILOT_REVIEW.md',
        PILOT/'REVIEW_HASH_VERIFICATION.json',PILOT/'decision.json',
        audit.DEST/'artifact_hashes.json',PREVIOUS/'artifact_hashes.json'}
    jobs=[]
    for group in pilot.GROUPS:
        other='6bba' if group=='44b6' else '44b6'
        members=[s for s in stems if s.startswith(group)]
        checkpoint=PILOT/'models'/f'{other}_step600.pt'
        meta=read(checkpoint.with_suffix('.json'))
        assert sha(checkpoint)==meta['model_sha256'] and meta['train_embryo']==other
        assert all(s.startswith(other) for s in meta['train_movies'])
        assert not set(members)&set(meta['train_movies'])
        smoke=next(s for _,s in evaluation_plan() if s.startswith(group))
        for name,values in [(f'extension_{group}',members),(f'smoke_{group}',[smoke])]:
            path=out/(name+'.txt');path.write_text('\n'.join(values)+'\n',encoding='utf-8');inputs.add(path)
        jobs.append(dict(group=group,train_embryo=other,stems=members,smoke=smoke,checkpoint=str(checkpoint)))
        previous=PREVIOUS/'replay'/f'base_{group}.csv'
        assert sha(previous)==read(PREVIOUS/'artifact_hashes.json')[str(previous.relative_to(PREVIOUS))]
        inputs.add(previous)
        # Build passive capture notebooks before pinning the extension inputs.
        pilot.replay(out,'off_'+group,members,CONTROL/'control_fp32_extension75')
        pilot.replay(out,'extension_'+group,members,out/'e2e'/('extension_'+group))
    for stem in stems:
        for folder in [ROOT/'data/train'/(stem+'.zarr'),ROOT/'data/train'/(stem+'.geff'),pilot.graph_path(CONTROL/'control_fp32_extension75',stem)]:
            inputs.update(p for p in folder.rglob('*') if p.is_file())
        inputs.add(CONTROL/'control_fp32_extension75/edge_cache'/(stem+'.npz'))
        p=audit.DEST/'graphs'/('extension75_'+stem[:4])/(stem+'_off.npz')
        assert sha(p)==read(audit.DEST/'artifact_hashes.json')[str(p.relative_to(audit.DEST))]
        inputs.add(p)
    save_json(out/'variants.json',{'as_configured':{}})
    inputs.update(p for p in out.iterdir() if p.is_file())
    for p in sorted(inputs):hashes[str(p.relative_to(ROOT))]=sha(p)
    verify_hashes(hashes)
    save_json(out/'plan.json',dict(created=stamp(),jobs=jobs,hashes=hashes,total_jobs=11,
        max_start_job_hours=5,estimated_minutes=135,
        estimate_basis='Measured C05222 inference1455sec,off/on replay623sec;scale75/22 plus2 reproduction movies and audit/hash margin',
        recipe='UNCHANGED C052 trained step600 checkpoints and C023 inference/postprocess; no fits or new knobs',
        validation='opposite whole embryo ONLY; no prefix deployment router; frozen detector/head not held out'))
    print('Prepared C05275 extension;',len(hashes),'pinned inputs;11 jobs',flush=True)


def infer(out,name,checkpoint,capture=False):
    cmd=training.inference(PILOT,name,out/(name+'.txt'),[
        f'BIOHUB_C037_CHECKPOINT={checkpoint}','BIOHUB_C037_ALPHA=1.0',
        f'BIOHUB_C037_CAPTURE_DIR={out/"evaluation_features"}' if capture else 'BIOHUB_C037_CAPTURE_DIR=',
        'BIOHUB_CACHE_EDGE_THRESHOLD=0.02'])
    cmd[cmd.index('--out')+1]=out/'e2e';return cmd


def smoke_check(out):
    ns=pilot.ns_for();rows=[]
    for j in read(out/'plan.json')['jobs']:
        stem=j['smoke'];new=out/'e2e'/('smoke_'+j['group']);old=PILOT/'e2e'/('division600_'+j['group'])
        with np.load(new/'edge_cache'/(stem+'.npz')) as a,np.load(old/'edge_cache'/(stem+'.npz')) as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files),('full cache parity',stem)
        a,b=[load_raw_graph(ns,pilot.graph_path(r,stem)) for r in [new,old]]
        assert a[0]==b[0]
        key=lambda e:(e['source_id'],e['target_id'])
        assert sorted(a[1],key=key)==sorted(b[1],key=key),('full ILP parity',stem)
        rows.append(dict(stem=stem,exact_all_cache_arrays=True,exact_ilp=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def analyse(out):
    ns=pilot.ns_for();jobs=read(out/'plan.json')['jobs']
    off=pd.concat([pd.read_csv(out/'replay'/('off_'+j['group']+'.csv')) for j in jobs])
    data=pd.concat([pd.read_csv(out/'replay'/('extension_'+j['group']+'.csv')) for j in jobs])
    original=pd.read_csv(OLD/'control_fp32_extension75.csv')
    assert len(data)==len(off)==75 and set(data.stem)==set(original.stem)
    audit.verify(off,original,{'as_configured':'as_configured'})
    for j in jobs:
        for stem in j['stems']:
            assert audit.graph(out/'graphs'/('off_'+j['group']),stem,'final')==audit.graph(audit.DEST/'graphs'/('extension75_'+j['group']),stem,'off')
    combined=pd.concat([pd.read_csv(PILOT/'official_per_movie.csv'),data]);assert len(combined)==97
    baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{sp}.csv') for sp in ['heldout12','confirm10','extension75']])
    previous=pd.concat([pd.read_csv(PILOT.parent/'c037_transformer_finetune/replay'/f'late600_{g}.csv') for g in pilot.GROUPS]+[
        pd.read_csv(PREVIOUS/'replay'/f'base_{g}.csv') for g in pilot.GROUPS])
    assert len(previous)==97 and set(previous.stem)==set(combined.stem)
    rows=[]
    for kind,frame in [('extension75',data),('all97',combined)]:
        for group in ['all',*pilot.GROUPS]:
            members={s for s in frame.stem if group=='all' or s.startswith(group)}
            a,b,c=[ns['aggregate_official'](f[f.stem.isin(members)].to_dict('records')) for f in [frame,baseline,previous]]
            delta=frame.set_index('stem').loc[sorted(members),'adjusted_edge_jaccard']-baseline.set_index('stem').loc[sorted(members),'adjusted_edge_jaccard']
            rows.append(dict(kind=kind,group=group,movies=len(members),score=a['proxy_score'],delta_vs_C023=a['proxy_score']-b['proxy_score'],
                delta_vs_C037_late600=a['proxy_score']-c['proxy_score'],edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
                edge_delta_vs_C037=a['adjusted_edge_jaccard']-c['adjusted_edge_jaccard'],div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn'],
                edge_wins=int((delta>1e-10).sum()),edge_losses=int((delta< -1e-10).sum())))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False);combined.to_csv(out/'official_per_movie97.csv',index=False)
    changes=[]
    for j in jobs:
        for stem in j['stems']:
            for stage in ['ilp','motion','pre_restore','final']:
                a,ae=audit.graph(out/'graphs'/('off_'+j['group']),stem,stage)
                b,be=audit.graph(out/'graphs'/('extension_'+j['group']),stem,stage)
                ac=collections.Counter((a[s],a[t]) for s,t in ae);bc=collections.Counter((b[s],b[t]) for s,t in be)
                changes.append(dict(stem=stem,stage=stage,removed=sum((ac-bc).values()),added=sum((bc-ac).values()),before_nodes=len(a),after_nodes=len(b)))
    pd.DataFrame(changes).to_csv(out/'stage_graph_changes.csv',index=False)
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,exact_off_metrics=75,exact_off_graphs=75,
        note='Combined97 retain opposite-embryo component folds. Only two biological embryos,correlated crops,frozen public models exposed. No submission qualification without fixed deployment/portable/T4 verification.'))


def run(out):
    assert not (out/'status.json').exists(),'No blind restart'
    plan=read(out/'plan.json');q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(out/'plan.json'));q.save();verify_hashes(plan['hashes'])
        for j in plan['jobs']:q.run('smoke_infer_'+j['group'],infer(out,'smoke_'+j['group'],j['checkpoint']))
        q.run('exact_full_movie_reproductions',[sys.executable,'-u',Path(__file__),'smoke_check','--out',out])
        for j in plan['jobs']:
            g=j['group'];meta=read(Path(j['checkpoint']).with_suffix('.json'))
            assert all(s.startswith(j['train_embryo']) for s in meta['train_movies']) and j['train_embryo']!=g
            q.run('off_'+g,pilot.replay_command(out/('off_'+g+'.ipynb'),CONTROL/'control_fp32_extension75',j['stems'],out/'variants.json',out/'replay'/('off_'+g+'.csv')))
            q.run('infer_'+g,infer(out,'extension_'+g,j['checkpoint'],True))
            q.run('replay_'+g,pilot.replay_command(out/('extension_'+g+'.ipynb'),out/'e2e'/('extension_'+g),j['stems'],out/'variants.json',out/'replay'/('extension_'+g+'.csv')))
        q.run('aggregate97',[sys.executable,'-u',Path(__file__),'analyse','--out',out])
        q.run('division_survival97',[sys.executable,'-u',ROOT/'src/c052_division_survival.py','--extension',out])
        verify_hashes(plan['hashes'])
        files=[p for folder in ['graphs','replay','e2e','evaluation_features'] for p in (out/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        files += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.ipynb'] and p.name not in ['plan.json','status.json','launch.json','artifact_hashes.json','automation_confirmation.json']]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        assert len(q.state['jobs'])==plan['total_jobs'];q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','smoke_check','analyse']);p.add_argument('--out',type=Path,default=DEST)
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT);globals()[a.command](out)
