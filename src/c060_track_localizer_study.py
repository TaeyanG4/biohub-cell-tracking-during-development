"""C060 finite spatial-track localization study using existing official tools.

Known-only opposite-embryo component experiment. The C023 final graph is frozen;
this driver never uploads, rewrites a scorer, or selects evaluation thresholds.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import c058_raw_localizer_study as old
import c058_localizer_baseline as baseline
import c060_track_localizer_data as data_code
import c060_spatial_localizer_model as model_code
from c055_guarded_readmit import read,save_json,sha,split_stems
from run_last_days_local import Queue,stamp
from frame_motion_audit import VOX
from local_registration_probe import bounded_crop

DEST=ROOT/'experiments/candidates/c060_spatial_track_localizer'
SOURCE=ROOT/'experiments/candidates/c058_raw_localizer'
BENCH=old.BENCH_STEMS
CROP=(13,49,49)
EXPANDED=(15,57,57)
RECIPE=dict(seed=6001,steps=1200,batch_size=32,lr=.0003,weight_decay=.0001,
    crop_zyx=list(CROP),expanded_crop_zyx=list(EXPANDED),jitter_vox_zyx=[1,4,4],
    training='uniform source movie then stable-known three-node GT lineage',
    inference='unique consecutive nonbranching real predicted triple; no GT input',
    symmetry='all eight inverse-reflected logit maps; unique integer modes must agree',
    ownership='agreed proposal must stay strictly inside original all-prediction Voronoi cell',
    optimizer='AdamW;CosineAnnealingLR;FP32;final1200step checkpoint',
    topology='C023 final IDs,times,node counts,edges frozen',
    folds='opposite biological embryo only; public detector already saw both',
    no_photometric_augmentation=True,model=model_code.RECIPE)


def policy():
    old.numeric_policy();torch.manual_seed(RECIPE['seed']);np.random.seed(RECIPE['seed'])


def movies(out):
    if (out/'plan.json').exists():return read(out/'plan.json')['test_movies']
    return sum([s for k,s in split_stems().items() if k!='extension75'],[])


def controls(out):
    """Reuse immutable exact outputs, then execute the actual official evaluator."""
    folder=out/'baseline';folder.mkdir(parents=True,exist_ok=True);rows=[]
    recorded=read(SOURCE/'output_hashes.json')
    for stem in movies(out):
        for suffix in ['npz','json']:
            rel=f'baseline/{stem}.{suffix}';source=SOURCE/rel
            digest=recorded.get(rel,recorded.get(rel.replace('/','\\')))
            assert digest and sha(source)==digest,(stem,'C058 baseline input drift')
            shutil.copy2(source,folder/source.name)
        g=old.load_baseline(out,stem)
        actual=baseline.official_score(g['ids'],g['txyz'],g['edges'],ROOT/'data/train'/f'{stem}.geff')
        expected=read(folder/f'{stem}.json')['official']
        for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes']:
            assert actual[k]==expected[k],(stem,k)
        assert abs(actual['adj_edge_jaccard']-expected['adj_edge_jaccard'])<1e-12
        rows.append(dict(stem=stem,status='passed',sha256=sha(folder/f'{stem}.npz')))
    save_json(out/'baseline_controls.json',dict(status='passed',rows=rows))


def extract_all(out):
    for stem in movies(out):data_code.extract_one(out,stem)


def load_training(out,embryo):
    result={}
    for stem in movies(out):
        if not stem.startswith(embryo):continue
        with np.load(out/'train'/f'{stem}.npz') as z:targets=z['targets'].copy()
        crops=np.load(out/'train'/f'{stem}.npy',mmap_mode='r')
        assert len(crops)==len(targets)
        if len(targets):result[stem]=(crops,targets)
    assert result and all(s.startswith(embryo) for s in result)
    return result


def sample(data,rng):
    stem=sorted(data)[int(rng.integers(len(data)))];crops,targets=data[stem]
    ix=rng.integers(len(crops),size=RECIPE['batch_size'])
    x=torch.from_numpy(np.asarray(crops[ix],np.float32)).cuda()
    y=torch.from_numpy(targets[ix].copy()).cuda()
    x,y,_=model_code.jitter_batch(x,y)
    x,y,_=model_code.reflect_batch(x,y)
    return x,y,stem


def train(out,embryo):
    policy();data=load_training(out,embryo);model=model_code.SpatialLocalizer().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=RECIPE['steps'])
    rng=np.random.default_rng(RECIPE['seed']);counts={s:0 for s in data};history=[];start=time.perf_counter()
    for step in range(1,RECIPE['steps']+1):
        x,y,stem=sample(data,rng);counts[stem]+=len(x);opt.zero_grad(set_to_none=True)
        loss=model_code.conditional_loss(model(x),y);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();scheduler.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            history.append(row);print(json.dumps(row),flush=True)
    folder=out/'models';folder.mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),recipe=RECIPE,train_embryo=embryo,
        training_stems=sorted(data),steps=RECIPE['steps']),folder/f'{embryo}.pt')
    save_json(folder/f'{embryo}.json',dict(embryo=embryo,recipe=RECIPE,training_stems=sorted(data),
        sample_counts=counts,history=history,seconds=time.perf_counter()-start))


def infer_one(out,stem,model,destination):
    """Runtime inputs are images and predicted graph only; no labels/GT read."""
    policy();g=old.load_baseline(SOURCE,stem);meta=old.metadata(stem)
    prev,nxt,mask=data_code.context_indices(g,meta[1]['shape'],CROP)
    corrected=g['txyz'].copy();shifts=np.zeros((len(mask),3),np.int64)
    agreed=np.zeros(len(mask),bool);owned=agreed.copy();start=time.perf_counter();cache={}
    for t in np.unique(g['txyz'][mask,0]):
        for tt in [int(t)-1,int(t),int(t)+1]:
            if tt not in cache:cache[tt]=old.get_frame(stem,tt,meta)
        for tt in list(cache):
            if abs(tt-int(t))>1:del cache[tt]
        ix=np.flatnonzero(mask&(g['txyz'][:,0]==t))
        all_ix=np.flatnonzero(g['txyz'][:,0]==t)
        tree=cKDTree(g['txyz'][all_ix,1:]*VOX)
        for begin in range(0,len(ix),32):
            part=ix[begin:begin+32];triples=np.stack([prev[part],part,nxt[part]],axis=1)
            blocks=[np.stack([bounded_crop(cache[int(g['txyz'][j,0])],g['txyz'][j,1:],CROP)[0]
                               for j in rows]) for rows in triples]
            x=torch.from_numpy(np.stack(blocks)).cuda()
            with torch.inference_mode():pred=model.predict(x)
            proposal=pred['proposal_vox'].cpu().numpy().astype(np.int64)
            accept=pred['accepted'].cpu().numpy().astype(bool)
            proposed=(g['txyz'][part,1:]+proposal)*VOX
            nearest_distance,nearest_local=tree.query(proposed,k=min(2,len(all_ix)))
            if len(all_ix)>1:
                is_own=(all_ix[nearest_local[:,0]]==part)&(nearest_distance[:,1]>nearest_distance[:,0]+1e-9)
            else:is_own=np.ones(len(part),bool)
            keep=accept&is_own;shifts[part[keep]]=proposal[keep]
            agreed[part]=accept;owned[part]=is_own
    corrected[:,1:]+=shifts
    assert (corrected[:,1:]>=0).all() and np.array_equal(corrected[~mask],g['txyz'][~mask])
    assert (np.linalg.norm(shifts*VOX,axis=1)<=7.+1e-8).all()
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/f'{stem}.npz',ids=g['ids'],txyz=corrected,edges=g['edges'],
        shifts_um=shifts*VOX,eligible=mask,mode_agreement=agreed,ownership=owned,gap_synthetic=g['gap_synthetic'])
    row=dict(stem=stem,nodes=len(mask),eligible=int(mask.sum()),agreed=int(agreed.sum()),
        ownership_rejected=int((agreed&~owned).sum()),changed=int(np.any(shifts!=0,axis=1).sum()),
        seconds=time.perf_counter()-start)
    save_json(destination/f'{stem}.json',row);print(json.dumps(row),flush=True);return row


def infer(out,embryo):
    policy();source='6bba' if embryo=='44b6' else '44b6'
    saved=torch.load(out/'models'/f'{source}.pt',map_location='cpu',weights_only=True)
    assert saved['train_embryo']==source and saved['recipe']==RECIPE and saved['steps']==RECIPE['steps']
    assert saved['training_stems']==sorted(load_training(out,source))
    model=model_code.SpatialLocalizer().cuda().eval();model.load_state_dict(saved['state_dict'])
    for stem in movies(out):
        if stem.startswith(embryo):infer_one(out,stem,model,out/'predictions')


def evaluate(out):
    with contextlib.redirect_stdout(io.StringIO()):ns=baseline.namespace()
    rows=[];residuals=[];transitions=[]
    continuity=pd.read_csv(ROOT/'state/localization_diagnosis_20260929/continuity.csv')
    for stem in movies(out):
        original=old.load_baseline(out,stem)
        with np.load(out/'predictions'/f'{stem}.npz') as z:new={k:z[k].copy() for k in z.files}
        assert all(np.array_equal(original[k],new[k]) for k in ['ids','edges'])
        assert np.array_equal(original['txyz'][:,0],new['txyz'][:,0])
        _,_,mask=data_code.context_indices(original,old.metadata(stem)[1]['shape'],CROP)
        assert np.array_equal(mask,new['eligible'])
        assert np.array_equal(original['txyz'][~mask],new['txyz'][~mask])
        gtpath=ROOT/'data/train'/f'{stem}.geff';gt,_=ns['graph_to_plain'](ns['graph_from_geff'](gtpath))
        matches={}
        for arm,g in [('zero',original),('localizer',new)]:
            row=baseline.official_score(g['ids'],g['txyz'],g['edges'],gtpath)
            rows.append(dict(row,stem=stem,embryo=stem[:4],arm=arm))
            matches[arm]=baseline.official_match(g['ids'],g['txyz'],g['edges'],gtpath)
        before,after=matches['zero'],matches['localizer'];lookup={int(n):i for i,n in enumerate(original['ids'])}
        for n,gid in before.items():
            i=lookup[n];target=np.array(gt[gid][1:]);old_error=np.linalg.norm((original['txyz'][i,1:]-target)*VOX)
            residuals.append(dict(stem=stem,embryo=stem[:4],node_id=n,gt_id=gid,eligible=bool(mask[i]),
                before_um=old_error,after_um=np.linalg.norm((new['txyz'][i,1:]-target)*VOX),
                before_group='le2_5' if old_error<=2.5 else ('le3_5' if old_error<=3.5 else 'gt3_5'),
                same_identity=after.get(n)==gid,unmatched_after=n not in after,
                changed_identity=n in after and after[n]!=gid,changed=bool(np.any(new['txyz'][i]!=original['txyz'][i]))))
        transitions.append(dict(stem=stem,retained=sum(after.get(n)==g for n,g in before.items()),
            lost=sum(n not in after for n in before),remapped=sum(n in after and after[n]!=g for n,g in before.items()),
            previously_unmatched_now_matched=sum(n not in before for n in after)))
        print(stem,'actual official evaluated',flush=True)
    pd.DataFrame(rows).to_csv(out/'official_rows.csv',index=False)
    res=pd.DataFrame(residuals).merge(continuity.rename(columns={'p':'node_id'})[
        ['stem','node_id','all_known_links_correct','any_known_link_wrong']],on=['stem','node_id'],validate='one_to_one')
    res.to_csv(out/'paired_residuals.csv',index=False)
    save_json(out/'identity_transitions.json',transitions)


def analyse(out):
    rows=pd.read_csv(out/'official_rows.csv');res=pd.read_csv(out/'paired_residuals.csv');summary=[]
    for group in ['all22','44b6','6bba']:
        d=rows if group=='all22' else rows[rows.embryo==group]
        r=res if group=='all22' else res[res.embryo==group]
        arms={a:d[d.arm==a] for a in ['zero','localizer']}
        scores={a:baseline.official_summary(v.to_dict('records')) for a,v in arms.items()}
        item=dict(group=group,scores=scores,delta_total=scores['localizer']['score']-scores['zero']['score'],
            delta_edge=scores['localizer']['adj_edge_jaccard']-scores['zero']['adj_edge_jaccard'])
        for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']:
            item['delta_'+k]=int(arms['localizer'][k].sum()-arms['zero'][k].sum())
        delta=arms['localizer'].set_index('stem').adj_edge_jaccard-arms['zero'].set_index('stem').adj_edge_jaccard
        item.update(movie_wins=int((delta>1e-12).sum()),movie_losses=int((delta< -1e-12).sum()))
        for label,mask in [('eligible',r.eligible),('tail',r.eligible&(r.before_group=='gt3_5')),
                           ('good',r.eligible&(r.before_group=='le2_5')),
                           ('good_all',r.before_group=='le2_5'),('all_original',np.ones(len(r),bool)),
                           ('persistent_tail',r.eligible&(r.before_group=='gt3_5')&r.all_known_links_correct),
                           ('broken_tail',r.eligible&(r.before_group=='gt3_5')&r.any_known_link_wrong)]:
            sub=r[mask];item[label]=dict(n=len(sub),before=float(sub.before_um.mean()),after=float(sub.after_um.mean()),
                changed=int(sub.changed.sum()),lost=int(sub.unmatched_after.sum()),remapped=int(sub.changed_identity.sum()))
        summary.append(item)
    by={r['group']:r for r in summary}
    component=all(by[g]['eligible']['after']<by[g]['eligible']['before'] and
        by[g]['tail']['after']<by[g]['tail']['before'] and
        by[g]['persistent_tail']['after']<by[g]['persistent_tail']['before'] and
        by[g]['good']['after']<=by[g]['good']['before']+1e-9 and
        by[g]['good_all']['lost']==0 and by[g]['good_all']['remapped']==0 and
        by[g]['delta_total']>=-1e-12 and by[g]['delta_edge']>=-1e-12 for g in ['44b6','6bba'])
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=summary))
    save_json(out/'decision.json',dict(component_gate_passed=component,
        reason='Both embryos overall/tail/persistent-tail paired localization improve; good-point mean/identity and actual official graph do not regress.',
        further_work='Only manual review may justify an unchanged97 component extension or separate exact-control pre-association integration.',
        no_hidden_gain_claim=True,no_auto_submission=True))
    print(json.dumps(summary,indent=2),flush=True)


def benchmark(out):
    out.mkdir(parents=True,exist_ok=True);policy();rows=[]
    for stem in BENCH:
        data_code.extract_one(out,stem)
        model=model_code.SpatialLocalizer().cuda().eval()
        row=infer_one(out,stem,model,out/'zero_benchmark')
        g=old.load_baseline(SOURCE,stem)
        with np.load(out/'zero_benchmark'/f'{stem}.npz') as z:
            assert all(np.array_equal(z[k],g[k]) for k in ['ids','txyz','edges'])
        rows.append(row)
    train_data=load_training_benchmark(out)
    model=model_code.SpatialLocalizer().cuda();opt=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'])
    rng=np.random.default_rng(RECIPE['seed']);torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(30):
        x,y,_=sample(train_data,rng);opt.zero_grad(set_to_none=True)
        loss=model_code.conditional_loss(model(x),y);loss.backward();opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    save_json(out/'benchmark.json',dict(status='passed',recipe=RECIPE,zero_graphs_exact=2,
        full_movie_zero_rows=rows,training30seconds=elapsed,checkpoint_saved=False))


def benchmark_queue(out):
    folder=out/'preflight_run';q=Queue(folder,1)
    try:
        q.run('two_movie_benchmark',[sys.executable,'-u',Path(__file__),'benchmark','--out',out])
        q.close('complete_review_required')
    except Exception as exc:q.close('failed',exc);raise


def load_training_benchmark(out):
    stem=BENCH[0]
    with np.load(out/'train'/f'{stem}.npz') as z:target=z['targets'].copy()
    return {stem:(np.load(out/'train'/f'{stem}.npy',mmap_mode='r'),target)}


def audit(out):
    policy();results=[]
    for stem in movies(out):
        graph=data_code.load_graph(stem)
        context=data_code.context_indices(graph,old.metadata(stem)[1]['shape'],EXPANDED)
        expected=data_code.source_labels(stem,graph,context)
        receipt=read(out/'train'/f'{stem}.json')
        assert sha(out/'train'/f'{stem}.npy')==receipt['crop_sha256']
        assert sha(out/'train'/f'{stem}.npz')==receipt['targets_sha256']
        with np.load(out/'train'/f'{stem}.npz') as z:
            targets=z['targets'];assert len(targets)>0 and np.isfinite(targets).all()
            assert (np.linalg.norm(targets,axis=1)<=7.+1e-6).all()
            assert np.array_equal(z['targets'],expected[['dz_um','dy_um','dx_um']].to_numpy(np.float32))
            assert np.array_equal(z['node_ids'],expected.node_id.to_numpy(np.int64))
            assert np.array_equal(z['gt_ids'],expected.gt_id.to_numpy(np.int64))
            assert np.array_equal(z['context_rows'],expected[['previous_row','row','next_row']].to_numpy(np.int64))
            results.append(dict(stem=stem,embryo=stem[:4],labels=len(targets),tails=int((np.linalg.norm(targets,axis=1)>3.5).sum())))
    assert all(sum(r['tails'] for r in results if r['embryo']==e)>0 for e in ['44b6','6bba'])
    save_json(out/'label_audit.json',dict(status='passed',rows=results))


def prepare(out):
    assert not (out/'plan.json').exists(),'Registered recipe immutable'
    bench=read(out/'benchmark.json');assert bench['status']=='passed' and bench['recipe']==RECIPE
    assert read(out/'preflight.json')['status']=='passed'
    splits={k:v for k,v in split_stems().items() if k!='extension75'};stems=sum(splits.values(),[])
    inputs={Path(__file__),out/'README.md',out/'benchmark.json',out/'preflight.json',
        ROOT/'state/localization_diagnosis_20260929/continuity.csv'}
    inputs.update(p for p in (ROOT/'src').glob('*.py'))
    inputs.update(p for p in (out/'zero_benchmark').rglob('*') if p.is_file())
    inputs.update(p for p in (out/'writer_preflight').rglob('*') if p.is_file())
    inputs.update(ROOT/name for name in read(out/'preflight.json')['proof_hashes'])
    for name in ['c060_spatial_localizer_model.py','c060_track_localizer_data.py','c058_raw_localizer_study.py',
                 'c058_raw_localizer_model.py','c058_localizer_baseline.py','c055_guarded_readmit.py',
                 'run_last_days_local.py','frame_motion_audit.py','local_registration_probe.py',
                 'eval_pp_variants_local.py','evaluate_local.py']:
        inputs.add(ROOT/'src'/name)
    inputs.add(old.BASE);inputs.add(SOURCE/'output_hashes.json')
    for stem in stems:
        inputs.update([SOURCE/'baseline'/f'{stem}.npz',SOURCE/'baseline'/f'{stem}.json',SOURCE/'labels'/f'{stem}.csv'])
        for folder in [ROOT/'data/train'/f'{stem}.zarr',ROOT/'data/train'/f'{stem}.geff']:
            inputs.update(p for p in folder.rglob('*') if p.is_file())
    import c055_guarded_readmit as replay
    import eval_pp_variants_local as harness
    inputs.update([harness.DEEPCENTER_CHECKPOINT,harness.DEEPCENTER_MANIFEST])
    for split,split_movies in splits.items():
        for stem in split_movies:inputs.update(p for p in replay.pred_path(split,stem).rglob('*') if p.is_file())
    import evaluate_local
    inputs.update(p for p in evaluate_local.VENDOR_SRC.parent.rglob('*.py') if p.is_file())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}
    count=sum(len(np.load(SOURCE/'baseline'/f'{s}.npz')['ids']) for s in stems)
    rate=sum(r['seconds'] for r in bench['full_movie_zero_rows'])/sum(r['nodes'] for r in bench['full_movie_zero_rows'])
    estimate=int(np.ceil((count*rate+2*RECIPE['steps']*bench['training30seconds']/30)/60+20))
    save_json(out/'plan.json',dict(created=stamp(),recipe=RECIPE,splits=splits,test_movies=stems,hashes=hashes,
        total_jobs=10,max_start_job_hours=4,estimated_minutes=estimate,estimated_from_actual_nodes=count,
        no_kaggle_writes=True,gate='Both embryos overall/tail/persistent-tail residual improve; no good mean/identity or official regression'))
    print('Prepared',len(hashes),'inputs;estimated minutes',estimate,flush=True)


def run(out):
    assert not (out/'status.json').exists(),'No blind resume'
    p=read(out/'plan.json');q=Queue(out,p['max_start_job_hours'])
    try:
        q.state.update(total_jobs=p['total_jobs'],plan_sha256=sha(out/'plan.json'));q.save()
        for name,digest in p['hashes'].items():assert sha(ROOT/name)==digest,('input drift',name)
        for verb in ['controls','extract_all','audit']:
            q.run(verb,[sys.executable,'-u',Path(__file__),verb,'--out',out])
        for verb in ['train','infer']:
            for group in ['44b6','6bba']:
                q.run(verb+'_'+group,[sys.executable,'-u',Path(__file__),verb,'--out',out,'--group',group])
        for verb in ['evaluate','writer_verify','analyse']:
            q.run(verb,[sys.executable,'-u',Path(__file__),verb,'--out',out])
        for name,digest in p['hashes'].items():assert sha(ROOT/name)==digest,('input drift',name)
        hashes={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and
            f.name not in ['queue.lock','status.json','launcher.stdout.log','launcher.stderr.log','output_hashes.json']}
        save_json(out/'output_hashes.json',hashes);q.close('complete_review_required')
    except Exception as exc:q.close('failed',exc);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('verb',choices=['benchmark','benchmark_queue','controls','extract_all','audit','train','infer','evaluate','writer_verify','analyse','prepare','run'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--group',choices=['44b6','6bba']);a=p.parse_args()
    if a.verb in ['train','infer']:globals()[a.verb](a.out,a.group)
    elif a.verb=='writer_verify':old.writer_verify(a.out)
    else:globals()[a.verb](a.out)


if __name__=='__main__':main()
