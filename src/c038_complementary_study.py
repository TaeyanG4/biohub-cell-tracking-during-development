#!/usr/bin/env python3
"""C038 small-signal combination review using the existing official replay tool.

Prepared alongside C037; run only when the one-GPU C037 queue is idle/complete.
Local replay notebooks explicitly contain local imports and must never be pushed.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reid_probe_local import BASE,CONTROL,OLD,sha,save_json,stamp,setup_ns
from reid_augmented_local import AppearanceEncoder,RECIPE,evaluation_plan
from run_last_days_local import Queue

DEST=ROOT/'experiments/candidates/c038_complementary_fusion'
C035=ROOT/'experiments/candidates/c035_augmented_reid'
C036=ROOT/'experiments/candidates/c036_local_registration'


def audit(out):
    out.mkdir(parents=True,exist_ok=True)
    counts={g:{k:dict(proposals=0,fixes=0,harms=0) for k in ['appearance','registration','union','agreement']} for g in ['44b6','6bba']}
    overlap=[]
    for _,stem in evaluation_plan():
        with np.load(C035/'scores'/f'{stem}.npz') as z:d={k:z[k].copy() for k in z.files}
        registration=pd.read_csv(C036/'groups'/f'{stem}.csv').set_index('source')
        for sid in np.unique(d['source']):
            ids=np.flatnonzero(d['source']==sid);old=np.flatnonzero(d['selected'][ids])
            if len(old)!=1:continue
            sim=d['weak_aug'][ids];geo=d['geometry'][ids];labels=d['label'][ids];targets=d['target'][ids]
            top,second=np.argsort(-sim,kind='stable')[:2];r=registration.loc[sid]
            ap=int(targets[top]) if top!=old[0] and sim[top]-sim[old[0]]>=.1 and sim[top]-sim[second]>=.05 and geo[top,3]<=geo[old[0],3]+1 else None
            rp=int(r.candidate) if r.proposal else None
            if ap is not None and rp is not None:overlap.append(dict(stem=stem,source=int(sid),same_target=ap==rp))
            choices=dict(appearance=ap,registration=rp,union=ap if rp is None else rp if ap is None or ap==rp else None,
                         agreement=ap if ap is not None and r.trusted and int(r.candidate)==ap else None)
            was=bool(labels[old[0]]==1)
            for mode,target in choices.items():
                if target is None:continue
                correct=bool(labels[np.flatnonzero(targets==target)[0]]==1)
                row=counts[stem[:4]][mode];row['proposals']+=1;row['fixes']+=int(correct and not was);row['harms']+=int(was and not correct)
    save_json(out/'overlap_diagnostic.json',dict(reviewed=stamp(),counts=counts,overlap=overlap,
        conclusion='Union adds no unique recovery; motion agreement removes appearance harms in the annotated-source diagnostic, leaving1 fix. User requested testing small effects, so proceed to GT-free actual graph replay.',
        warning='Diagnostic source whitelist must never be used as a graph-edit whitelist.'))


def build(out):
    audit(out)
    for group in ['44b6','6bba']:
        train='6bba' if group=='44b6' else '44b6'
        nb=json.loads(BASE.read_text(encoding='utf-8'))
        code=''.join(nb['cells'][5]['source'])
        anchor='\nwrite_test_submission("base")\n';assert code.count(anchor)==1
        addition=("\n# C038 LOCAL REPLAY ONLY; NOT A KAGGLE SUBMISSION NOTEBOOK.\nimport sys\n"
                  f"sys.path.insert(0, {str(ROOT/'src')!r})\nfrom c038_complementary_stage import install_complementary\n"
                  f"install_complementary(globals(), {str(C035/'models'/f'{train}_weak_aug.pt')!r})\n")
        nb['cells'][5]['source']=code.replace(anchor,addition+anchor,1).splitlines(keepends=True)
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'C038 cell{i}','exec')
        (out/f'local_replay_{group}.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
    (out/'variants.json').write_text(json.dumps({m:{'C038_MODE':m} for m in ['off','appearance','agreement']},indent=2))


def cpu_parity(out):
    torch.set_num_threads(4);rows=[]
    for group in ['44b6','6bba']:
        stem=next(s for _,s in evaluation_plan() if s.startswith(group))
        train='6bba' if group=='44b6' else '44b6'
        model=AppearanceEncoder().eval();checkpoint=torch.load(C035/'models'/f'{train}_weak_aug.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        with np.load(C035/'eval'/f'{stem}.npz') as z:
            si,ti=z['source_ix'][:64],z['target_ix'][:64];ids=np.unique(np.r_[si,ti]);x=z['crops'][ids].astype(np.float32)
        with torch.no_grad():v=model(torch.from_numpy(x)[:,None]).numpy()
        sim=(v[np.searchsorted(ids,si)]*v[np.searchsorted(ids,ti)]).sum(1)
        with np.load(C035/'scores'/f'{stem}.npz') as z:ref=z['weak_aug'][:64]
        err=float(np.max(np.abs(sim-ref)));assert err<1e-4,err
        rows.append(dict(stem=stem,cpu_vs_cached_gpu_cosine_max_error=err))
    save_json(out/'cpu_parity.json',dict(status='passed',rows=rows))


def analyse(out):
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    rows=[];baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{s}.csv') for s in ['heldout12','confirm10']])
    data=pd.concat([pd.read_csv(p) for p in sorted((out/'replay').glob('*.csv')) if not p.stem.endswith('_summary')])
    control=data[data.config=='off'].set_index('stem');ref=baseline.set_index('stem')
    assert len(control)==22 and set(control.index)==set(ref.index)
    for key in ['nodes','edges','edge_tp','edge_fp','edge_fn','div_tp','div_fp','div_fn','adjusted_edge_jaccard']:
        assert np.allclose(control[key].sort_index(),ref[key].sort_index(),rtol=0,atol=1e-10),('off_control',key)
    sets={'all22':set(ref.index),'44b6':{s for s in ref.index if s.startswith('44b6')},'6bba':{s for s in ref.index if s.startswith('6bba')}}
    sets.update({split:{s for sp,s in evaluation_plan() if sp==split} for split in ['heldout12','confirm10']})
    for mode in ['appearance','agreement']:
        for name,stems in sets.items():
            frame=data[(data.config==mode)&data.stem.isin(stems)]
            cur=ns['aggregate_official'](frame.to_dict('records'));base=ns['aggregate_official'](baseline[baseline.stem.isin(stems)].to_dict('records'))
            rows.append(dict(mode=mode,group=name,score=cur['proxy_score'],delta=cur['proxy_score']-base['proxy_score'],
                             edge_delta=cur['adjusted_edge_jaccard']-base['adjusted_edge_jaccard'],div_tp=cur['div_tp'],div_fp=cur['div_fp'],div_fn=cur['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    save_json(out/'analysis.json',dict(status='official_replay_complete_review_required',off_controls_verified=22,rows=rows,
        instruction='Do not discard solely for small effect. Inspect changed-edge counts, complementary errors and later-stage suppression; extend supported tiny gains before final judgment.'))


def run(out):
    # Prevent unintended concurrent GPU replay while C037 runs.
    p=ROOT/'experiments/candidates/c037_transformer_finetune/status.json'
    if p.exists() and json.loads(p.read_text())['status']=='running':raise RuntimeError('C037 still running; defer C038 until its GPU queue completes')
    q=Queue(out,8)
    try:
        build(out);cpu_parity(out)
        tracked=[Path(__file__),ROOT/'src/c038_complementary_stage.py',ROOT/'src/reid_probe_local.py',
                 ROOT/'src/reid_augmented_local.py',ROOT/'src/local_registration_probe.py',ROOT/'src/eval_pp_variants_local.py',BASE]
        hashes={str(p.relative_to(ROOT)):sha(p) for p in tracked}
        if q.state.get('source_hashes'):assert q.state['source_hashes']==hashes,'C038 source drift'
        q.state['source_hashes']=hashes;q.save()
        for split in ['heldout12','confirm10']:
            for group in ['44b6','6bba']:
                stems=[s for sp,s in evaluation_plan() if sp==split and s.startswith(group)]
                if not stems:continue
                run_dir=CONTROL/('control_fp32_'+split)
                q.run(f'replay_{split}_{group}',[sys.executable,'-u',ROOT/'src/eval_pp_variants_local.py',
                    '--notebook',out/f'local_replay_{group}.ipynb','--variants',out/'variants.json',
                    '--stems',','.join(stems),'--pred-root',run_dir/'predictions','--lowdet-dir',run_dir/'edge_cache',
                    '--round-coords','--out',out/'replay'/f'{split}_{group}.csv'])
        analyse(out)
        assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in tracked}
        q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','analyse'])
    ap.add_argument('--out',type=Path,default=DEST);a=ap.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':build(out);cpu_parity(out)
    else:globals()[a.command](out)


if __name__=='__main__':main()
