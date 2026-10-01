#!/usr/bin/env python3
"""Finite fixed C040 appearance extension using existing inference/replay/Queue."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from c040_late600_combo import (ROOT, C037, C038, BASE, CONTROL, OLD, Queue,
    sha, save_json, stamp, setup_ns, evaluation_plan, verify, edge_audit)
from c037_transformer_study import HEAD, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, inference
from run_last_days_local import extension_stems, replay_command
from eval_pp_variants_local import load_raw_graph

PILOT = ROOT/'experiments/candidates/c040_late600_combo'
DEST = PILOT/'extension75'


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise RuntimeError('Registered extension exists; do not overwrite')
    old = json.loads((PILOT/'plan.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in old['hashes'].items()), 'pilot input drift'
    inputs = {ROOT/p for p in old['hashes']}
    source_hashes = json.loads((C037/'status.json').read_text())['source_hashes']
    assert all(sha(ROOT/p)==v for p,v in source_hashes.items()), 'C037 source drift'
    inputs.update(ROOT/p for p in source_hashes)
    inputs.update([Path(__file__), ROOT/'src/c037_transformer_study.py',
        HEAD, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, OLD/'control_fp32_extension75.csv',
        ROOT/'experiments/candidates/c035_augmented_reid/plan.json'])
    inputs.update((C037/'tracking_repo').rglob('*.py'))
    stems, lists = extension_stems({s for _,s in evaluation_plan()})
    inputs.update(ROOT/p for p in lists)
    save_json(out/'variants.json', {'off':{'C038_MODE':'off'}, 'appearance':{'C038_MODE':'appearance'}})
    save_json(out/'base_variants.json', {'as_configured':{}})
    inputs.update([out/'variants.json', out/'base_variants.json'])
    jobs=[]
    for group in ['44b6','6bba']:
        train='6bba' if group=='44b6' else '44b6'
        checkpoint=C037/'models'/f'{train}_step600.pt'
        metadata=json.loads(checkpoint.with_suffix('.json').read_text())
        assert sha(checkpoint)==metadata['model_sha256']
        assert not set(metadata['train_movies']) & (set(stems)|{s for _,s in evaluation_plan()})
        inputs.update([checkpoint,checkpoint.with_suffix('.json')])
        members=[s for s in stems if s.startswith(group)]
        smoke=next(s for _,s in evaluation_plan() if s.startswith(group))
        for label, values in [('extension_'+group,members),('smoke_'+group,[smoke])]:
            path=out/(label+'.txt'); path.write_text('\n'.join(values)+'\n'); inputs.add(path)
        name='extension_'+group
        nb=json.loads((C038/f'local_replay_{group}.ipynb').read_text())
        code=''.join(nb['cells'][5]['source']); anchor='\nwrite_test_submission("base")\n'
        assert code.count(anchor)==1
        addition=('\nfrom c038_followup_local import install_graph_audit\n'
                  f'install_graph_audit(globals(), {str(out/"graphs"/name)!r})\n')
        nb['cells'][5]['source']=code.replace(anchor,addition+anchor).splitlines(keepends=True)
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'{name}:cell{i}','exec')
        path=out/(name+'.ipynb'); path.write_text(json.dumps(nb,indent=1),encoding='utf-8'); inputs.add(path)
        jobs.append(dict(name=name,kind='extension',group=group,stems=members,smoke=smoke,checkpoint=str(checkpoint)))
        inputs.add(PILOT/'replay'/f'combo_late600_{group}.csv')
    save_json(out/'plan.json',dict(created=stamp(),jobs=jobs,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs) if p.is_file()},
        policy='Retrospective fixed appearance extension;22 totals positive on both splits but confirm edge/6bba negative. Resolve generalization, not submission approval.',
        deployment=False))
    print('Prepared fixed75 extension and two full-movie reproduction checks',flush=True)


def infer_command(out, label, checkpoint):
    # Use the C037 command builder without writing into the original C037 run.
    cmd=inference(C037,label,out/(label+'.txt'),[
        f'BIOHUB_C037_CHECKPOINT={checkpoint}','BIOHUB_C037_ALPHA=1.0','BIOHUB_C037_CAPTURE_DIR='])
    cmd[cmd.index('--out')+1]=out/'e2e'
    return cmd


def smoke_check(out):
    plan=json.loads((out/'plan.json').read_text()); rows=[]
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    for job in plan['jobs']:
        stem=job['smoke']; new=out/'e2e'/('smoke_'+job['group']); old=C037/'e2e'/('late600_'+job['group'])
        with np.load(new/'edge_cache'/f'{stem}.npz') as a,np.load(old/'edge_cache'/f'{stem}.npz') as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files), ('cached arrays drift',stem)
        a,b=[load_raw_graph(ns,next((folder/'predictions').rglob(stem+'.geff'))) for folder in [new,old]]
        assert a[0]==b[0],('ILP nodes drift',stem)
        key=lambda e:(e['source_id'],e['target_id'])
        assert sorted(a[1],key=key)==sorted(b[1],key=key),('ILP edges drift',stem)
        rows.append(dict(stem=stem,exact_cache=True,exact_ilp=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def analyse(out):
    plan=json.loads((out/'plan.json').read_text())
    ext=pd.concat([pd.read_csv(out/'replay'/(j['name']+'.csv')) for j in plan['jobs']])
    direct=pd.concat([pd.read_csv(out/'replay'/('base_'+j['group']+'.csv')) for j in plan['jobs']])
    assert len(direct)==75
    verify(ext,direct,{'off':'as_configured'})
    pilot=pd.concat([pd.read_csv(PILOT/'replay'/f'combo_late600_{g}.csv') for g in ['44b6','6bba']])
    baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{s}.csv') for s in ['heldout12','confirm10','extension75']])
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    rows=[]
    for kind,data in [('extension75',ext),('aggregate97',pd.concat([pilot,ext]))]:
        for mode in ['off','appearance']:
            for group in ['all','44b6','6bba']:
                stems={s for s in data.stem if group=='all' or s.startswith(group)}
                current=data[(data.config==mode)&data.stem.isin(stems)]
                own=data[(data.config=='off')&data.stem.isin(stems)]
                ref=baseline[baseline.stem.isin(stems)]
                assert len(current)==len(own)==len(ref)==len(stems)
                a,b,c=[ns['aggregate_official'](x.to_dict('records')) for x in [current,ref,own]]
                delta=current.set_index('stem').adjusted_edge_jaccard-ref.set_index('stem').adjusted_edge_jaccard
                rows.append(dict(kind=kind,mode=mode,group=group,movies=len(stems),score=a['proxy_score'],
                    delta=a['proxy_score']-b['proxy_score'],edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
                    delta_vs_own_off=a['proxy_score']-c['proxy_score'],
                    edge_delta_vs_own_off=a['adjusted_edge_jaccard']-c['adjusted_edge_jaccard'],
                    wins=int((delta>1e-10).sum()),losses=int((delta< -1e-10).sum()),
                    div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    # Reuse existing passive graph invariant audit, limited to the fixed registered arm.
    import c038_followup_local as audit
    prior=audit.MODES
    try:
        audit.MODES=['off','appearance']; edge_audit(out,plan)
    finally:audit.MODES=prior
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,
        parity=dict(full_movie_reproductions=2,direct_late600_off_controls=75),
        baseline='Pinned previously verified C023 FP32 controls; not newly inferred.',
        note='Retrospective fixed extension, cross-embryo diagnostic folds; no hidden deployment router.'))


def run(out):
    assert json.loads((PILOT/'status.json').read_text())['status']!='running'
    plan=json.loads((out/'plan.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift'
    q=Queue(out,8)
    try:
        q.state['plan_sha256']=sha(out/'plan.json');q.save()
        for j in plan['jobs']:q.run('infer_smoke_'+j['group'],infer_command(out,'smoke_'+j['group'],j['checkpoint']))
        q.run('exact_reproduction',[sys.executable,'-u',Path(__file__),'smoke_check','--out',out])
        for j in plan['jobs']:
            q.run('infer_'+j['group'],infer_command(out,j['name'],j['checkpoint']))
            folder=out/'e2e'/j['name']
            q.run('base_'+j['group'],replay_command(BASE,folder,j['stems'],out/'base_variants.json',out/'replay'/('base_'+j['group']+'.csv')))
            q.run('combo_'+j['group'],replay_command(out/(j['name']+'.ipynb'),folder,j['stems'],out/'variants.json',out/'replay'/(j['name']+'.csv')))
        q.run('analyse',[sys.executable,'-u',Path(__file__),'analyse','--out',out])
        assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift during run'
        files=[p for folder in ['graphs','replay'] for p in (out/folder).rglob('*') if p.is_file()]
        files += [p for folder in (out/'e2e').iterdir() for sub in ['edge_cache','predictions'] for p in (folder/sub).rglob('*') if p.is_file()]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','smoke_check','analyse'])
    parser.add_argument('--out',type=Path,default=DEST)
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    globals()[args.command](out)
