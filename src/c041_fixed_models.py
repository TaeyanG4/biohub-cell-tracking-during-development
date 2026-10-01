#!/usr/bin/env python3
"""Fixed deployment-model pilot; extract existing stage, reuse inference/replay/Queue."""
from __future__ import annotations
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from c040_late600_combo import (ROOT,C037,C038,BASE,CONTROL,OLD,Queue,sha,save_json,
    stamp,setup_ns,evaluation_plan,verify)
from c037_transformer_study import HEAD,PRIMARY_WEIGHTS,SECONDARY_WEIGHTS,inference
from reid_augmented_local import RECIPE
from run_last_days_local import replay_command

DEST=ROOT/'experiments/candidates/c041_fixed_models'
C035=ROOT/'experiments/candidates/c035_augmented_reid'


def extract(path,name):
    text=path.read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(text,node)


def portable_source():
    probe=ROOT/'src/reid_probe_local.py';stage=ROOT/'src/c038_complementary_stage.py'
    aug=ROOT/'src/reid_augmented_local.py'
    notebook=json.loads(BASE.read_text(encoding='utf-8'))
    motion=next(ast.get_source_segment(s,n) for c in notebook['cells'] if c['cell_type']=='code'
        for s in [''.join(c['source'])] for n in ast.parse(s).body
        if isinstance(n,ast.FunctionDef) and n.name=='motion_relink_edges' and 'def assign_pass' in ast.get_source_segment(s,n))
    recorder=ast.parse(extract(probe,'Recorder')).body[0]
    recorder.body=[n for n in recorder.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','capture']]
    install=extract(probe,'install_recorder')
    start=install.index('    source = next(');end=install.index('    patches = [',start)
    install=install[:start]+'    source = MOTION_SOURCE\n'+install[end:]
    patches=extract(aug,'patches').replace("ROOT/'data/train'/(stem+'.zarr')","Path(ns['TEST_DIR'])/(stem+'.zarr')")
    original=extract(stage,'install_complementary')
    begin=original.index('            state=torch.load(');end=original.index("            cache['model']=model",begin)
    original=original[:begin]+'            model=load_appearance(checkpoint)\n'+original[end:]
    original=original.replace("['off','appearance','agreement']","['off','appearance']")
    begin=original.index("            if mode=='agreement':");end=original.index('            proposals[sid]=target;',begin)
    original=original[:begin]+original[end:]
    blocks=[
        'import collections, copy, math, os\nfrom pathlib import Path\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport zarr',
        f'RECIPE = {RECIPE!r}\nCROP=(8,32,32)\nFRAMES=(-1,0,1)\nMOTION_SOURCE={motion!r}',
        extract(ROOT/'src/division_crops_extract.py','crop_at'),patches,
        extract(ROOT/'src/division_cnn_train.py','DivisionCNN'),extract(aug,'AppearanceEncoder'),
        '''class MeanCosineEncoder(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.models=nn.ModuleList([AppearanceEncoder() for _ in range(count)])
    def forward(self,x):
        return torch.cat([model(x) for model in self.models],dim=1)/math.sqrt(len(self.models))

def load_appearance(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['recipe']==RECIPE and state['arm']=='weak_aug'
    model=MeanCosineEncoder(state['n_models']) if 'n_models' in state else AppearanceEncoder()
    model.load_state_dict(state['state_dict'],strict=True)
    return model.cpu().eval()''',
        ast.unparse(recorder),extract(stage,'AllNodeRecorder'),install,
        '@torch.no_grad()\n'+extract(stage,'appearance_vectors'),original]
    source='\n\n'.join(blocks)+'\n'
    compile(source,'c041_portable_appearance','exec')
    assert 'data/train' not in source and 'from reid' not in source
    return source


def make_notebook(out,label,checkpoint,source):
    nb=json.loads(BASE.read_text(encoding='utf-8'));code=''.join(nb['cells'][5]['source'])
    anchor='\nwrite_test_submission("base")\n';assert code.count(anchor)==1
    payload=(f'\nexec(compile({source!r}, "c041_portable_appearance", "exec"), globals())\n'
        f'install_complementary(globals(), {str(checkpoint)!r})\n'
        'from c038_followup_local import install_graph_audit\n'
        f'install_graph_audit(globals(), {str(out/"graphs"/label)!r})\n')
    nb['cells'][5]['source']=code.replace(anchor,payload+anchor).splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'{label}:{i}','exec')
    path=out/(label+'.ipynb');path.write_text(json.dumps(nb,indent=1),encoding='utf-8');return path


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise RuntimeError('Registered plan exists')
    old=json.loads((ROOT/'experiments/candidates/c040_late600_combo/extension75/plan.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in old['hashes'].items()),'prior input drift'
    inputs={ROOT/p for p in old['hashes']}
    inputs.update([Path(__file__),PRIMARY_WEIGHTS,SECONDARY_WEIGHTS,HEAD])
    source=portable_source();runtime=out/'portable_appearance.py';runtime.write_text(source,encoding='utf-8');inputs.add(runtime)
    spec=importlib.util.spec_from_file_location('c041_portable',runtime);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    models=[];training=set();state_dict={}
    for i,group in enumerate(['44b6','6bba']):
        path=C035/'models'/f'{group}_weak_aug.pt';inputs.add(path);state=torch.load(path,map_location='cpu',weights_only=True)
        training.update(state['train_movies']);models.append(module.load_appearance(path))
        for k,v in state['state_dict'].items():state_dict[f'models.{i}.{k}']=v
    official=set(json.loads((C035/'plan.json').read_text())['excluded_official97'])
    assert len(official)==97 and not training & official
    appearance=out/'appearance_mean_cosine.pt'
    torch.save(dict(state_dict=state_dict,n_models=2,recipe=RECIPE,arm='weak_aug',train_movies=sorted(training)),appearance)
    inputs.add(appearance)
    torch.set_num_threads(4);torch.manual_seed(4101);x=torch.randn(4,1,8,32,32)
    with torch.no_grad():
        vectors=module.load_appearance(appearance)(x)
        expected=sum(m(x)@m(x).T for m in models)/2
        error=float((vectors@vectors.T-expected).abs().max());assert error<1e-6
    checkpoints=[C037/'models'/f'{g}_step600.pt' for g in ['44b6','6bba']]
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in checkpoints]
    for p in checkpoints:
        assert sha(p)==json.loads(p.with_suffix('.json').read_text())['model_sha256'];inputs.add(p)
    assert states[0]['base_transformer_sha256']==states[1]['base_transformer_sha256']
    assert set(states[0]['state_dict'])==set(states[1]['state_dict'])
    training_tf=set(states[0]['train_movies'])|set(states[1]['train_movies']);assert not training_tf & official
    mean={}
    for k,a in states[0]['state_dict'].items():
        b=states[1]['state_dict'][k]
        if torch.is_floating_point(a):mean[k]=(a+b)/2
        else:assert torch.equal(a,b);mean[k]=a
    transformer=out/'transformer_mean600.pt'
    torch.save(dict(state_dict=mean,base_transformer_sha256=states[0]['base_transformer_sha256'],
        train_embryo='fixed_both_no_routing',step=600,train_movies=sorted(training_tf)),transformer);inputs.add(transformer)
    save_json(out/'preflight.json',dict(status='passed',mean_cosine_max_error=error,training_movies=len(training),
        no_training_overlap_official97=True,note='CPU algebra/extraction only; real pipeline parity still pending.'))
    variants=out/'variants.json';save_json(variants,{'off':{'C038_MODE':'off'},'appearance':{'C038_MODE':'appearance'}});inputs.add(variants)
    jobs=[]
    for group in ['44b6','6bba']:
        stem=next(s for _,s in evaluation_plan() if s.startswith(group));train='6bba' if group=='44b6' else '44b6'
        label='smoke_'+group;inputs.add(make_notebook(out,label,C035/'models'/f'{train}_weak_aug.pt',source))
        jobs.append(dict(name=label,kind='smoke',stems=[stem],run=str(CONTROL/'control_fp32_heldout12'),group=group))
    for split in ['heldout12','confirm10']:
        stems=[s for sp,s in evaluation_plan() if sp==split]
        path=out/(split+'.txt');path.write_text('\n'.join(stems)+'\n');inputs.add(path)
        for family in ['appearance','combined']:
            label=family+'_'+split;inputs.add(make_notebook(out,label,appearance,source))
            jobs.append(dict(name=label,kind=family,stems=stems,split=split,
                run=str(CONTROL/('control_fp32_'+split) if family=='appearance' else out/'e2e'/split)))
    # Inputs used by local replay: fixed model, all required baseline tables/caches and pilot smoke graphs.
    for job in jobs:
        if job['kind']=='combined':continue
        for stem in job['stems']:
            folder=Path(job['run']);inputs.update((folder/'edge_cache').glob(stem+'*'))
            graph=next((folder/'predictions').rglob(stem+'.geff'));inputs.update(p for p in graph.rglob('*') if p.is_file())
    for split in ['heldout12','confirm10']:inputs.add(OLD/f'control_fp32_{split}.csv')
    for job in jobs[:2]:
        inputs.add(C038/'replay'/f'heldout12_{job["group"]}.csv')
    inputs={p for p in inputs if p.is_file()}
    save_json(out/'plan.json',dict(created=stamp(),jobs=jobs,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)},
        policy='Fixed mean of two same-base late600 Transformer weights; mean cosine of two separate appearance models. No prefix routing, fitting or threshold sweep.',
        candidates=['appearance_only','transformer_only','combined'],deployment=False))
    print('Prepared fixed deployment-model pilot; no GPU launch; pinned',len(inputs),'inputs',flush=True)


def check_smoke(out):
    plan=json.loads((out/'plan.json').read_text());rows=[]
    from c038_followup_local import graph
    for job in [j for j in plan['jobs'] if j['kind']=='smoke']:
        data=pd.read_csv(out/'replay'/(job['name']+'.csv'))
        ref=pd.read_csv(C038/'replay'/f'heldout12_{job["group"]}.csv');ref=ref[ref.stem.isin(job['stems'])]
        verify(data,ref,{'off':'off','appearance':'appearance'})
        for mode in ['off','appearance']:
            before=graph(C038/'followup_extension/graphs'/f'pilot_heldout12_{job["group"]}',job['stems'][0],mode)
            after=graph(out/'graphs'/job['name'],job['stems'][0],mode)
            assert before==after,('portable graph mismatch',job['group'],mode)
        rows.append(dict(stem=job['stems'][0],exact_graphs=True,exact_metric=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def analyse(out):
    plan=json.loads((out/'plan.json').read_text())
    frames={family:pd.concat([pd.read_csv(out/'replay'/(j['name']+'.csv')) for j in plan['jobs'] if j['kind']==family]) for family in ['appearance','combined']}
    baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{s}.csv') for s in ['heldout12','confirm10']])
    verify(frames['appearance'],baseline,{'off':'as_configured'})
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    rows=[]
    for name,family,mode in [('appearance_only','appearance','appearance'),('transformer_only','combined','off'),('combined','combined','appearance')]:
        data=frames[family];data=data[data.config==mode]
        for group in ['all22','heldout12','confirm10','44b6','6bba']:
            stems={s for split,s in evaluation_plan() if group=='all22' or split==group or s.startswith(group)}
            current=data[data.stem.isin(stems)];ref=baseline[baseline.stem.isin(stems)]
            assert len(current)==len(ref)==len(stems)
            a,b=[ns['aggregate_official'](x.to_dict('records')) for x in [current,ref]]
            delta=current.set_index('stem').adjusted_edge_jaccard-ref.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(arm=name,group=group,score=a['proxy_score'],delta=a['proxy_score']-b['proxy_score'],
                edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],wins=int((delta>1e-10).sum()),losses=int((delta< -1e-10).sum()),
                div_tp=a['div_tp'],div_fp=a['div_fp'],div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    import c038_followup_local as audit
    jobs=[dict(j,kind=j['kind']) for j in plan['jobs'] if j['kind']!='smoke']
    prior=audit.MODES
    try:audit.MODES=['off','appearance'];audit.edge_audit(out,dict(jobs=jobs))
    finally:audit.MODES=prior
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=rows,appearance_off_controls=22,
        note='Fixed global models; current notebooks still local paths/audit imports. Build portable submission and verify actual T4 before submission.'))


def run(out):
    extension=ROOT/'experiments/candidates/c040_late600_combo/extension75/status.json'
    assert json.loads(extension.read_text())['status']!='running','C040 GPU queue still active; defer without waiting'
    plan=json.loads((out/'plan.json').read_text());assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift'
    q=Queue(out,8)
    try:
        q.state['plan_sha256']=sha(out/'plan.json');q.save()
        for j in [x for x in plan['jobs'] if x['kind']=='smoke']:
            q.run(j['name'],replay_command(out/(j['name']+'.ipynb'),Path(j['run']),j['stems'],out/'variants.json',out/'replay'/(j['name']+'.csv')))
        q.run('verify_portable_single',[sys.executable,'-u',Path(__file__),'check_smoke','--out',out])
        for family in ['appearance','combined']:
            for j in [x for x in plan['jobs'] if x['kind']==family]:
                if family=='combined':
                    cmd=inference(C037,j['split'],out/(j['split']+'.txt'),[f'BIOHUB_C037_CHECKPOINT={out/"transformer_mean600.pt"}','BIOHUB_C037_ALPHA=1.0','BIOHUB_C037_CAPTURE_DIR='])
                    cmd[cmd.index('--out')+1]=out/'e2e';q.run('infer_'+j['split'],cmd)
                q.run(j['name'],replay_command(out/(j['name']+'.ipynb'),Path(j['run']),j['stems'],out/'variants.json',out/'replay'/(j['name']+'.csv')))
        q.run('analyse',[sys.executable,'-u',Path(__file__),'analyse','--out',out])
        assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift'
        files=[p for folder in ['graphs','replay'] for p in (out/folder).rglob('*') if p.is_file()]
        files += [p for run_dir in (out/'e2e').iterdir() for sub in ['edge_cache','predictions'] for p in (run_dir/sub).rglob('*') if p.is_file()]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files});q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','check_smoke','analyse'])
    parser.add_argument('--out',type=Path,default=DEST);args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT);globals()[args.command](out)
