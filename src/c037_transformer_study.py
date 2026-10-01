#!/usr/bin/env python3
"""C037 frozen-detector, production-input transformer fine-tuning pilot.

Orchestrates the existing inference/official replay tools. Two fits/four checkpoints
and three fixed inference arms receive graph evaluation irrespective of a small
or zero diagnostic top-1 effect. No Kaggle operations in this queue.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from c037_transformer_runtime import state_digest
from reid_probe_local import BASE, CONTROL, OLD, setup_ns, sha, save_json, stamp
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue
from v1284_capture_local import PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, notebook_env

DEST = ROOT/'experiments/candidates/c037_transformer_finetune'
SOURCE = ROOT/'tmp/c023_output/tracking_repo'
HEAD = ROOT/'artifacts/anvithpothula_v1284_head_s075/v1284_head.pt'
RECIPE = dict(seed=3701, steps=600, checkpoints=[150,600], lr=1e-5, weight_decay=1e-4,
              teacher_kl=.1, division_weight=2., match_um=7., dropout=False,
              arms={'early150': [150,1.], 'late600': [600,1.], 'late600_blend25': [600,.25]})


def policy():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    assert torch.cuda.is_available()
    return dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), dtype='float32',
                matmul_tf32=False, cudnn_tf32=False, flash_sdp=False, memory_efficient_sdp=False,
                note='Local4070 policy, actual T4 still required')


def build(out):
    dest = out/'tracking_repo'; dest.mkdir(parents=True, exist_ok=True)
    for part in ['src', 'scripts']:
        shutil.copytree(SOURCE/part, dest/part, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    p = dest/'scripts/predict_unet_transformer.py'
    code = p.read_text(encoding='utf-8')
    anchor = 'import tracksdata as td\n'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor+'from c037_transformer_runtime import capture as _c037_capture, apply_primary as _c037_apply\n',1)
    anchor = '    model, window_size, downsample = load_model(weights_path, device)\n'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor+'    _c037_apply(model)\n',1)
    anchor = '            _bidirectional_weight = float(\n'
    assert code.count(anchor) == 1
    hook = '''            _c037_capture(ds_path, t_src, t_tgt, unet_feat_src, unet_feat_tgt,
                          p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                          p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt, edge_logits_pair)

'''
    code = code.replace(anchor,hook+anchor,1)
    compile(code,str(p),'exec'); p.write_text(code,encoding='utf-8')
    shutil.copy2(ROOT/'src/c037_transformer_runtime.py',dest/'scripts/c037_transformer_runtime.py')
    plan035 = json.loads((ROOT/'experiments/candidates/c035_augmented_reid/plan.json').read_text())
    train = [r['stem'] for r in plan035['selected']]
    excluded = plan035['excluded_official97']
    assert len(train)==24 and not set(train).intersection(excluded)
    plan = dict(recipe=RECIPE, train_movies=train, excluded_official97=excluded,
                test_movies=[s for _,s in evaluation_plan()], train_selection='Reuse C035 fixed24 GT-eligible movies, no score selection',
                primary_sha256=sha(PRIMARY_WEIGHTS), secondary_sha256=sha(SECONDARY_WEIGHTS), head_sha256=sha(HEAD))
    plan_path=out/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'C037 plan drift'
    else:
        save_json(plan_path,plan)
    (out/'train_stems.txt').write_text('\n'.join(train)+'\n')
    for g in ['44b6','6bba']:
        (out/f'test_{g}.txt').write_text('\n'.join(s for _,s in evaluation_plan() if s.startswith(g))+'\n')
    (out/'variants.json').write_text(json.dumps({'as_configured':{}}))
    return plan


def engine(out):
    policy()
    os.environ.update(notebook_env(BASE))
    os.environ.update(V1284_MODE='candidate', V1284_HEAD=str(HEAD), BIOHUB_CACHE_DIR='',
                      BIOHUB_LOCAL_WORKING=str(out), BIOHUB_C037_CHECKPOINT='')
    sys.path[:0] = [str(out/'tracking_repo/scripts'),str(out/'tracking_repo/src')]
    import predict_unet_transformer as pu
    code = (out/'tracking_repo/scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    code = code.replace('Path("/kaggle/working")', "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    exec(compile(code,pu.__file__,'exec'),pu.__dict__)
    pu.UNetNodeTransformer._index_features = pu._v1284_index
    return pu


def smoke(out):
    pu = engine(out)
    original=(SOURCE/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    original=original.replace('Path("/kaggle/working")',"Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    spec=importlib.util.spec_from_loader('c037_original_control',loader=None)
    base=importlib.util.module_from_spec(spec);base.__file__=str(SOURCE/'scripts/predict_unet_transformer.py')
    sys.modules[spec.name]=base;exec(compile(original,base.__file__,'exec'),base.__dict__)
    device=torch.device('cuda')
    model,w,down=pu.load_model(PRIMARY_WEIGHTS,device)
    secondary,w2,d2=pu.load_model(SECONDARY_WEIGHTS,device)
    assert w==w2==2 and down==d2
    cfg=pu.PredictConfig(det_threshold=float(os.environ['BIOHUB_DET_THRESHOLD']))
    cfg.threshold=float(os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'])
    kw=dict(cfg=cfg,window_size=w,max_frames=4,downsample=down,secondary_model=secondary,
            secondary_edge_weight=float(os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT']),
            secondary_detection_weight=float(os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT']),
            secondary_link_mode=os.environ['BIOHUB_SECONDARY_LINK_MODE'],
            secondary_mix_temperature=float(os.environ.get('BIOHUB_SECONDARY_MIX_TEMPERATURE','1')),
            secondary_low_margin_max=float(os.environ.get('BIOHUB_SECONDARY_LOW_MARGIN_MAX','.35')))
    rows=[]
    with (out/'smoke.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        for stem in ['44b6_12dfb391','6bba_05db0fb1']:
            result=[]
            for capture_on,eng in [(False,base),(True,pu)]:
                os.environ['BIOHUB_C037_CAPTURE_DIR']=str(out/'smoke_capture') if capture_on else ''
                eng._LOWDET.clear();eng._CACHE_EDGES.clear()
                coords,edges=eng.predict_video(model,ROOT/'data/train'/stem,device,**kw)
                result.append((np.asarray(coords).copy(),np.asarray(edges).copy(),[(a.copy(),b.copy()) for a,b in eng._LOWDET]))
            assert np.array_equal(result[0][0],result[1][0]) and np.array_equal(result[0][1],result[1][1])
            assert len(result[0][2])==len(result[1][2]) and all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(result[0][2],result[1][2]))
            max_error=0.
            for p in sorted((out/'smoke_capture'/stem).glob('*.npz')):
                with np.load(p) as z:
                    packet={k:z[k].copy() for k in z.files}
                with torch.no_grad():
                    logits=forward(model.transformer,packet)
                reference=packet['probe_logits'];check=logits.detach().cpu().numpy()[np.ix_(packet['probe_rows'],packet['probe_cols'])]
                error=float(np.max(np.abs(reference-check)));max_error=max(max_error,error)
                assert error < 2e-5, ('cached_forward',error)
            rows.append(dict(stem=stem,nodes=len(result[0][0]),edges=len(result[0][1]),exact_output=True,exact_lowdet=True,max_cached_logit_error=max_error))
    save_json(out/'smoke.json',dict(status='passed',rows=rows,runtime=policy(),note='Four-frame actual C023 inference, no modified-graph score'))


def forward(model,packet):
    keys=['feat_src','feat_tgt','coords_src','coords_tgt']
    a,b,c,d=[torch.from_numpy(packet[k]).to('cuda',dtype=torch.float32)[None] for k in keys]
    return model(a,b,c,d,torch.ones(a.shape[:2],device='cuda',dtype=torch.bool),
                 torch.ones(b.shape[:2],device='cuda',dtype=torch.bool))[0]


def label(out):
    plan=json.loads((out/'plan.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    records=[]
    for stem in plan['train_movies']:
        gt,edges=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(stem+'.geff')))
        parents={};children={}
        for a,b in edges:
            parents.setdefault(b,[]).append(a);children.setdefault(a,[]).append(b)
        for p in sorted((out/'features'/stem).glob('*.npz')):
            with np.load(p) as z:
                a,b=z['coords_src'],z['coords_tgt'];ta,tb=int(z['t_src']),int(z['t_tgt'])
            plain={i:(ta,*(max(0,int(round(float(v)))) for v in xyz)) for i,xyz in enumerate(a)}
            plain.update({len(a)+i:(tb,*(max(0,int(round(float(v)))) for v in xyz)) for i,xyz in enumerate(b)})
            mapping,_=ns['match_nodes_bipartite'](plain,gt,max_dist=RECIPE['match_um'])
            known=np.array(sorted(i for i in mapping if i<len(a)),np.int64)
            parent_to_local={mapping[int(i)]:j for j,i in enumerate(known)}
            targets=[];labels=[];weights=[]
            for j in range(len(b)):
                gid=mapping.get(len(a)+j)
                par=parents.get(gid,[])
                if len(par)==1 and par[0] in parent_to_local:
                    targets.append(j);labels.append(parent_to_local[par[0]])
                    weights.append(RECIPE['division_weight'] if len(children.get(par[0],[]))==2 else 1.)
            if len(known)<2 or not targets:
                continue
            # Unmatched source nodes stay in attention context, but never become negative labels.
            dest=out/'labels'/stem;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/p.name,known_src=known,targets=np.array(targets,np.int64),
                                parent_index=np.array(labels,np.int64),weight=np.array(weights,np.float32))
            records.append(dict(stem=stem,file=str(p.relative_to(out)),labels=str((dest/p.name).relative_to(out)),
                                packet_sha256=sha(p),label_sha256=sha(dest/p.name),source_nodes=len(a),target_nodes=len(b),
                                known_sources=len(known),positive_targets=len(targets),division_targets=sum(w>1 for w in weights)))
        print(stamp(),'labels',stem,'windows',sum(r['stem']==stem for r in records),flush=True)
    for g in ['44b6','6bba']:
        rows=[r for r in records if r['stem'].startswith(g)]
        if len(rows)<30 or len({r['stem'] for r in rows})<6:
            raise RuntimeError('C037 insufficient identifiable supervised windows: '+g)
    save_json(out/'training_manifest.json',dict(recipe=RECIPE,records=records,unknown_sources_are_negatives=False,
        inference_geometry_units='original full-resolution voxel coordinates, exactly as production',
        label_match='existing official one-to-one matcher,7um, integer emitted coordinate convention'))


def train(out,group):
    pu=engine(out);os.environ['BIOHUB_C037_CAPTURE_DIR']=''
    full,_,_=pu.load_model(PRIMARY_WEIGHTS,torch.device('cuda'))
    teacher=full.transformer
    student=copy.deepcopy(teacher)
    del full
    for p in teacher.parameters():p.requires_grad_(False)
    teacher.eval();student.eval()  # Keep dropout disabled while gradients update the transformer.
    base_hash=state_digest(teacher.state_dict())
    frozen_before=state_digest(teacher.state_dict())
    torch.manual_seed(RECIPE['seed']);rng=np.random.default_rng(RECIPE['seed'])
    records=json.loads((out/'training_manifest.json').read_text())['records']
    records=[r for r in records if group=='pooled' or r['stem'].startswith(group)]
    by_movie={s:[r for r in records if r['stem']==s] for s in sorted({r['stem'] for r in records})}
    for r in records:
        assert sha(out/r['file'])==r['packet_sha256'] and sha(out/r['labels'])==r['label_sha256']
    opt=torch.optim.AdamW(student.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,RECIPE['steps'])
    logs=[];checked=set();start=time.time()
    (out/'models').mkdir(exist_ok=True)
    for step in range(1,RECIPE['steps']+1):
        movies=list(by_movie);movie=movies[int(rng.integers(len(movies)))];pool=by_movie[movie]
        rec=pool[int(rng.integers(len(pool)))]
        with np.load(out/rec['file']) as z:packet={k:z[k].copy() for k in z.files}
        with np.load(out/rec['labels']) as z:lab={k:torch.from_numpy(z[k]).to('cuda') for k in z.files}
        with torch.no_grad():base=forward(teacher,packet)
        if rec['file'] not in checked:
            check=base.detach().cpu().numpy()[np.ix_(packet['probe_rows'],packet['probe_cols'])]
            assert np.max(np.abs(check-packet['probe_logits']))<2e-5, 'captured transformer parity failure'
            checked.add(rec['file'])
        current=forward(student,packet)
        known=current[lab['known_src']][:,lab['targets']].T
        ce=F.cross_entropy(known,lab['parent_index'],reduction='none')
        supervised=(ce*lab['weight']).sum()/lab['weight'].sum()
        # Distillation is a soft preservation regularizer, never a GT-negative assignment.
        kl=F.kl_div(F.log_softmax(current,dim=0),F.softmax(base,dim=0),reduction='sum')/current.shape[1]
        loss=supervised+RECIPE['teacher_kl']*kl
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite C037 training loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        opt.step();schedule.step()
        if step==1 or step%25==0:
            row=dict(step=step,loss=float(loss),supervised_ce=float(supervised),teacher_kl=float(kl),
                     movie=movie,known_sources=len(lab['known_src']),targets=len(lab['targets']),seconds=time.time()-start)
            logs.append(row);save_json(out/'models'/f'{group}.training.json',logs);print(stamp(),group,json.dumps(row),flush=True)
        if step in RECIPE['checkpoints']:
            assert state_digest(teacher.state_dict())==frozen_before
            learned={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            assert state_digest(learned)!=base_hash, 'no parameter update'
            destination=out/'models'/f'{group}_step{step}.pt'
            torch.save(dict(state_dict=learned,base_transformer_sha256=base_hash,train_embryo=group,
                            step=step,recipe=RECIPE,train_movies=movies),destination)
            save_json(destination.with_suffix('.json'),dict(model_sha256=sha(destination),step=step,
                train_embryo=group,train_movies=movies,base_transformer_unchanged=True,cached_forward_windows_verified=len(checked)))


def inference(out,label_name,stems_file,extras=()):
    cmd=[sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py','--repo',out/'tracking_repo',
         '--notebook',BASE,'--stems-file',stems_file,'--out',out/'e2e','--label',label_name,
         '--v1284-mode','candidate','--v1284-head',HEAD,'--t4-fp32']
    for item in extras:cmd+=['--env',item]
    return cmd


def analyse(out):
    # Use original official aggregation on actual end-to-end per-movie replay rows.
    with contextlib.redirect_stdout(io.StringIO()):ns,_,_=setup_ns(CONTROL/'control_fp32_heldout12')
    baseline=pd.concat([pd.read_csv(OLD/f'control_fp32_{split}.csv') for split in ['heldout12','confirm10']])
    split_sets={split:{s for p,s in evaluation_plan() if p==split} for split in ['heldout12','confirm10']}
    groups={'all22':set(baseline.stem),**split_sets,**{g:{s for s in baseline.stem if s.startswith(g)} for g in ['44b6','6bba']}}
    rows=[]
    for arm in RECIPE['arms']:
        frame=pd.concat([pd.read_csv(out/'replay'/f'{arm}_{g}.csv') for g in ['44b6','6bba']])
        assert set(frame.stem)==set(baseline.stem) and len(frame)==len(baseline)==22
        frame.to_csv(out/'replay'/f'{arm}_all22.csv',index=False)
        for name,members in groups.items():
            current=ns['aggregate_official'](frame[frame.stem.isin(members)].to_dict('records'))
            control=ns['aggregate_official'](baseline[baseline.stem.isin(members)].to_dict('records'))
            a=frame[frame.stem.isin(members)].set_index('stem')['adjusted_edge_jaccard']
            b=baseline[baseline.stem.isin(members)].set_index('stem')['adjusted_edge_jaccard']
            delta=a-b
            rows.append(dict(arm=arm,group=name,movies=len(members),score=current['proxy_score'],
                control=control['proxy_score'],delta=current['proxy_score']-control['proxy_score'],
                edge_delta=current['adjusted_edge_jaccard']-control['adjusted_edge_jaccard'],
                edge_wins=int((delta>1e-10).sum()),edge_losses=int((delta < -1e-10).sum()),
                div_tp=current['div_tp'],div_fp=current['div_fp'],div_fn=current['div_fn']))
    pd.DataFrame(rows).to_csv(out/'official_summary.csv',index=False)
    result=dict(status='pilot_complete_review_required',rows=rows,
                user_instruction='Small effects are not automatic rejection. Review complementarity, fixed combinations and extension evidence; do not reuse old diagnostic >=5 cutoff.',
                limitations=['Both base detectors were pretrained on these embryos.',
                             'Cross-embryo fine-tunes are diagnostic folds; hidden inference cannot route by embryo ID.',
                             'A pooled or fixed ensemble deployment model needs its own actual replay and T4 verification.'])
    save_json(out/'analysis.json',result)
    lines=['# C037 official end-to-end pilot','',
           '| Arm | Group | Score | Delta | Edge delta | Edge wins / losses |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['group']} | {r['score']:.6f} | {r['delta']:+.6f} | {r['edge_delta']:+.6f} | {r['edge_wins']} / {r['edge_losses']} |")
    lines+=['','Review small consistent effects and combinations before closure. No Kaggle operation performed.',*['- '+s for s in result['limitations']]]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    q=Queue(out,18)
    try:
        sources=[Path(__file__),ROOT/'src/c037_transformer_runtime.py',ROOT/'src/run_kaggle_predict_local.py',
                 ROOT/'src/v1284_capture_local.py',ROOT/'src/eval_pp_variants_local.py',ROOT/'src/run_last_days_local.py',BASE]
        sources += [SOURCE/'scripts'/name for name in ['predict_unet_transformer.py', 'train_unet_transformer.py', 'v1284_coordinate_refinement.py']]
        sources += sorted((SOURCE/'src/biohub_tracking/models').glob('*.py'))
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
        if q.state.get('source_hashes'):assert q.state['source_hashes']==hashes,'C037 source drift'
        q.state['source_hashes']=hashes;q.save()
        plan=build(out)
        q.state['recipe']=RECIPE;q.state['plan_sha256']=sha(out/'plan.json');q.save()
        script=Path(__file__)
        q.run('real_equivalence_smoke',[sys.executable,'-u',script,'smoke','--out',out])
        assert json.loads((out/'smoke.json').read_text())['status']=='passed'
        q.run('capture_train24',inference(out,'capture_train24',out/'train_stems.txt',
            [f'BIOHUB_C037_CAPTURE_DIR={out / "features"}','BIOHUB_C037_CHECKPOINT=']))
        q.run('label_production_inputs',[sys.executable,'-u',script,'label','--out',out])
        for group in ['44b6','6bba']:
            q.run('train_'+group,[sys.executable,'-u',script,'train','--group',group,'--out',out])
        for arm,(step,alpha) in RECIPE['arms'].items():
            for test in ['44b6','6bba']:
                train_group='6bba' if test=='44b6' else '44b6'
                label_name=arm+'_'+test
                checkpoint=out/'models'/f'{train_group}_step{step}.pt'
                q.run('infer_'+label_name,inference(out,label_name,out/f'test_{test}.txt',
                    [f'BIOHUB_C037_CHECKPOINT={checkpoint}',f'BIOHUB_C037_ALPHA={alpha}','BIOHUB_C037_CAPTURE_DIR=']))
                stems=[s for _,s in evaluation_plan() if s.startswith(test)]
                run_dir=out/'e2e'/label_name
                q.run('replay_'+label_name,[sys.executable,'-u',ROOT/'src/eval_pp_variants_local.py',
                    '--notebook',BASE,'--variants',out/'variants.json','--stems',','.join(stems),
                    '--pred-root',run_dir/'predictions','--lowdet-dir',run_dir/'edge_cache',
                    '--round-coords','--out',out/'replay'/f'{label_name}.csv'])
        q.run('aggregate_official',[sys.executable,'-u',script,'analyse','--out',out])
        assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in sources},'source changed during C037'
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed',exc);raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['run','smoke','label','train','analyse'])
    ap.add_argument('--out',type=Path,default=DEST)
    ap.add_argument('--group',choices=['44b6','6bba','pooled'])
    args=ap.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    if args.command=='train':
        assert args.group;train(out,args.group)
    else:globals()[args.command](out)


if __name__=='__main__':main()
