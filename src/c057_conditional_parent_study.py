"""C057: conditional parent competition on reused C052 production packets.

Reuses the original learner, sampler, inference, Queue and official replay.
Only the supervised parent competition changes; no inference rules change.
"""
from __future__ import annotations
import argparse
import collections
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import c037_transformer_study as old
import c052_division_transformer as pilot
import c038_followup_local as audit
from c047_hard_example_study import read, verify_hashes
from reid_probe_local import ROOT, BASE, CONTROL, OLD, sha, save_json, stamp
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue

DEST = ROOT/'experiments/candidates/c057_conditional_parent'
RECIPE = dict(pilot.RECIPE, arms={'conditional600': [600, 1.]},
              supervision='conditional parent likelihood with explicit identity-ambiguity exclusion',
              biological_unknown_nodes_are_negative=False)
GROUPS = pilot.GROUPS


def prepare(out):
    assert not (out/'plan.json').exists(), 'Registered sources are immutable'
    out.mkdir(parents=True, exist_ok=True)
    assert read(pilot.DEST/'status.json')['status'] == 'complete_review_required'
    source = read(pilot.DEST/'plan.json')
    verify_hashes(source['hashes'])
    for part in ['src', 'scripts']:
        shutil.copytree(pilot.DEST/'tracking_repo'/part, out/'tracking_repo'/part,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for group in GROUPS:
        shutil.copy2(pilot.DEST/f'test_{group}.txt', out/f'test_{group}.txt')
    save_json(out/'variants.json', {'as_configured': {}})
    manifest = read(pilot.DEST/'training_manifest.json')
    inputs = {Path(__file__), ROOT/'src/c057_conditional_parent_loss.py', out/'README.md',
              pilot.DEST/'training_manifest.json', pilot.DEST/'label_audit.json',
              pilot.DEST/'event_audit.json', pilot.DEST/'plan.json',
              ROOT/'src/c052_division_transformer.py', ROOT/'src/c037_transformer_study.py',
              ROOT/'src/c037_transformer_runtime.py', ROOT/'src/run_last_days_local.py',
              ROOT/'src/run_kaggle_predict_local.py', ROOT/'src/eval_pp_variants_local.py',
              BASE, old.HEAD, old.PRIMARY_WEIGHTS, old.SECONDARY_WEIGHTS}
    hashes = dict(source['hashes'])
    for rec in manifest['records']:
        for key, digest in [('file', 'packet_sha256'), ('labels', 'label_sha256')]:
            path = pilot.DEST/rec[key]
            assert sha(path) == rec[digest], str(path)
            inputs.add(path)
    for split, stem in evaluation_plan():
        inputs.update(p for p in (ROOT/'data/train'/(stem+'.geff')).rglob('*') if p.is_file())
        for stage in ['ilp', 'motion', 'pre_restore', 'final']:
            inputs.add(pilot.DEST/'graphs'/('off_'+split)/(stem+'_'+stage+'.npz'))
        inputs.add(pilot.control_run(stem)/'edge_cache'/(stem+'.npz'))
    for split in ['heldout12', 'confirm10']:
        inputs.add(OLD/f'control_fp32_{split}.csv')
    for g in GROUPS:
        inputs.add(pilot.DEST/'replay'/f'division600_{g}.csv')
    for stem in {r['stem'] for r in manifest['records']}:
        inputs.update(p for p in (ROOT/'data/train'/(stem+'.geff')).rglob('*') if p.is_file())
    inputs.update((out/'tracking_repo').rglob('*.py'))
    inputs.update(p for p in out.iterdir() if p.is_file())
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)})
    save_json(out/'plan.json', dict(created=stamp(), recipe=RECIPE, hashes=hashes,
        test_movies=[s for _, s in evaluation_plan()], total_jobs=12,
        estimated_minutes=65, max_start_job_hours=4,
        estimate_basis='Reuse all capture packets; two600-step fits, measured22 inference~25min, replay~10min, labels/controls/hash margin.',
        validation='opposite whole embryos; frozen public detector/head saw both',
        deployment=False, kaggle_writes=False))
    print('Prepared C057', len(hashes), 'pinned inputs; 12 finite jobs', flush=True)


def label(out):
    from c057_conditional_parent_loss import build_labels
    ns=pilot.ns_for();source=read(pilot.DEST/'training_manifest.json')
    records=[];rows=[];graphs={}
    for rec in source['records']:
        stem=rec['stem']
        if stem not in graphs:
            graphs[stem]=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(stem+'.geff')))
        feature=pilot.DEST/rec['file'];original=pilot.DEST/rec['labels']
        assert sha(feature)==rec['packet_sha256'] and sha(original)==rec['label_sha256']
        with np.load(feature) as z:packet={k:z[k].copy() for k in z.files}
        with np.load(original) as z:lab={k:z[k].copy() for k in z.files}
        labels,metadata=build_labels(packet,lab,*graphs[stem],ns)
        path=out/'labels'/stem/feature.name;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**labels)
        records.append(dict(rec,file=str(feature),labels=str(path),label_sha256=sha(path),
            original_label_path=str(original),original_label_sha256=rec['label_sha256']))
        rows.append(dict(stem=stem,pool=rec['pool'],file=str(feature),**metadata))
    assert len(records)==len(source['records'])
    save_json(out/'training_manifest.json',dict(source,recipe=RECIPE,records=records,
        conditional_unmatched_source_edge_negatives=True,
        unknown_sources_are_negative_cells=False,
        supervision='Unique annotated target-parent edge, original7um identity gate; ambiguous unmatched alternatives excluded'))
    pd.DataFrame(rows).to_csv(out/'label_provenance.csv',index=False)
    groups=[]
    for g in GROUPS:
        subset=[r for r in rows if r['stem'].startswith(g)]
        groups.append(dict(embryo=g,packets=len(subset),
            **{key:sum(r[key] for r in subset) for key in ['targets','conditional_edge_negatives','masked_identity_ambiguous','targets_with_conditional_negatives']},
            original_division_targets=sum(r['division_targets'] for r in records if r['stem'].startswith(g))))
    save_json(out/'label_audit.json',dict(status='passed',rows=groups,original_packet_and_target_counts_unchanged=True,
        biological_unknown_cell_negatives=False,unlabelled_target_supervision=False,identity_exclusion_um=7.,
        records=len(records),manifest_sha256=sha(out/'training_manifest.json')))
    print(json.dumps(groups),flush=True)


def smoke(out):
    import torch
    from c057_conditional_parent_loss import numerical_smoke
    old.smoke(out)
    engine=old.engine(out);full,_,_=engine.load_model(old.PRIMARY_WEIGHTS,torch.device('cuda'))
    model=full.transformer;model.eval();records=read(out/'training_manifest.json')['records'];rows=[]
    for g in GROUPS:
        for pool in ['ordinary','division']:
            rec=next(r for r in records if r['stem'].startswith(g) and r['pool']==pool)
            with np.load(rec['file']) as z:packet={k:z[k].copy() for k in z.files}
            with np.load(rec['labels']) as z:lab={k:z[k].copy() for k in z.files}
            with torch.no_grad():current=old.forward(model,packet)
            error=float(np.max(np.abs(current.cpu().numpy()[np.ix_(packet['probe_rows'],packet['probe_cols'])]-packet['probe_logits'])))
            assert error<2e-5,('teacher packet',g,pool,error)
            rows.append(dict(embryo=g,pool=pool,teacher_probe_error=error,**numerical_smoke(current,lab)))
    save_json(out/'loss_smoke.json',dict(status='passed',rows=rows))


def train(out, group):
    from c057_conditional_parent_loss import supervised_loss
    old.RECIPE = RECIPE
    code = inspect.getsource(old.train)
    anchor = "    opt=torch.optim.AdamW(student.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])"
    assert code.count(anchor) == 1
    code = code.replace(anchor, '    sampler=FixedSampler(out,group,records,rng)\n'+anchor)
    before = "        movies=list(by_movie);movie=movies[int(rng.integers(len(movies)))];pool=by_movie[movie]\n        rec=pool[int(rng.integers(len(pool)))]"
    assert code.count(before) == 1
    code = code.replace(before, "        movies=list(by_movie);rec=sampler(step);movie=rec['stem']")
    before = "        known=current[lab['known_src']][:,lab['targets']].T\n        ce=F.cross_entropy(known,lab['parent_index'],reduction='none')\n        supervised=(ce*lab['weight']).sum()/lab['weight'].sum()"
    assert code.count(before) == 1
    code = code.replace(before, '        supervised=supervised_loss(current,lab)')
    code += '\n    sampler.finish()\n'
    scope = dict(old.__dict__, FixedSampler=pilot.FixedSampler, supervised_loss=supervised_loss)
    exec(compile(code, str(Path(__file__))+'::conditional_parent_loss', 'exec'), scope)
    scope['train'](out, group)


def verify_folds(out):
    pilot.verify_folds(out)


def analyse(out):
    ns = pilot.ns_for()
    baseline = pd.concat([pd.read_csv(OLD/f'control_fp32_{sp}.csv') for sp in ['heldout12', 'confirm10']])
    controls = pd.concat([pd.read_csv(out/'replay'/f'off_{sp}.csv') for sp in ['heldout12', 'confirm10']])
    audit.verify(controls, baseline, {'as_configured':'as_configured'})
    data = pd.concat([pd.read_csv(out/'replay'/f'conditional600_{g}.csv') for g in GROUPS])
    previous = pd.concat([pd.read_csv(pilot.DEST/'replay'/f'division600_{g}.csv') for g in GROUPS])
    assert len(data) == 22 and set(data.stem) == set(baseline.stem)
    rows = []
    groups = [('all22', set(data.stem))]
    groups += [(sp, {s for p, s in evaluation_plan() if p == sp}) for sp in ['heldout12','confirm10']]
    groups += [(g, {s for s in data.stem if s.startswith(g)}) for g in GROUPS]
    for name, members in groups:
        a,b,c = [ns['aggregate_official'](f[f.stem.isin(members)].to_dict('records')) for f in [data,baseline,previous]]
        d = data.set_index('stem').loc[sorted(members), 'adjusted_edge_jaccard']-baseline.set_index('stem').loc[sorted(members), 'adjusted_edge_jaccard']
        rows.append(dict(group=name, movies=len(members), score=a['proxy_score'],
            delta_vs_C023=a['proxy_score']-b['proxy_score'], delta_vs_C052=a['proxy_score']-c['proxy_score'],
            edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
            edge_delta_vs_C052=a['adjusted_edge_jaccard']-c['adjusted_edge_jaccard'],
            div_tp=a['div_tp'], div_fp=a['div_fp'], div_fn=a['div_fn'],
            **{f'{label}_{key}':int(frame[frame.stem.isin(members)][key].sum())
               for label,frame in [('c057',data),('c023',baseline),('c052',previous)]
               for key in ['edge_tp','edge_fp','edge_fn','div_tp','div_fp','div_fn']},
            edge_wins=int((d>1e-10).sum()), edge_losses=int((d < -1e-10).sum())))
    changes = []
    for split,stem in evaluation_plan():
        for stage in ['ilp','motion','pre_restore','final']:
            before = audit.graph(out/'graphs'/('off_'+split),stem,stage)
            assert before == audit.graph(pilot.DEST/'graphs'/('off_'+split),stem,stage), ('off graph', stem,stage)
            after = audit.graph(out/'graphs'/('conditional600_'+stem[:4]),stem,stage)
            an,ae=before;bn,be=after
            ac=collections.Counter((an[s],an[t]) for s,t in ae)
            bc=collections.Counter((bn[s],bn[t]) for s,t in be)
            changes.append(dict(stem=stem,stage=stage,removed=sum((ac-bc).values()),added=sum((bc-ac).values()),
                                before_nodes=len(an),after_nodes=len(bn)))
        with np.load(out/'e2e'/('conditional600_'+stem[:4])/'edge_cache'/(stem+'.npz')) as a, \
             np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as b:
            for key in ['coords','low_coords','low_score']:
                assert np.array_equal(a[key],b[key],equal_nan=True), ('frozen detector',stem,key)
    pd.DataFrame(rows).to_csv(out/'official_summary.csv', index=False)
    data.to_csv(out/'official_per_movie.csv', index=False)
    pd.DataFrame(changes).to_csv(out/'stage_graph_changes.csv', index=False)
    save_json(out/'analysis.json', dict(status='complete_review_required', rows=rows, exact_off_controls=22,
        exact_off_stage_graphs=88, frozen_detector_movies=22,
        label_audit=read(out/'label_audit.json'), fold_proof=read(out/'fold_proof.json'),
        note='No probability-only promotion. Review vs BOTH C023 and C052 before unchanged75 extension; no hidden score claim.'))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run(out):
    assert not (out/'status.json').exists(), 'No blind restart'
    plan=read(out/'plan.json');q=Queue(out,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(out/'plan.json'));q.save()
        verify_hashes(plan['hashes'])
        def task(name,command,*args):q.run(name,[sys.executable,'-u',Path(__file__),command,'--out',out,*args])
        task('conditional_label_provenance','label')
        task('real_off_and_loss_smoke','smoke')
        for g in GROUPS:task('train_'+g,'train','--group',g)
        task('whole_embryo_fold_proof','verify_folds')
        for split in ['heldout12','confirm10']:
            stems=[s for p,s in evaluation_plan() if p==split]
            q.run('off_'+split,pilot.replay(out,'off_'+split,stems,CONTROL/('control_fp32_'+split)))
        for test in GROUPS:
            training='6bba' if test=='44b6' else '44b6';name='conditional600_'+test
            verify_folds(out)
            q.run('infer_'+test,old.inference(out,name,out/f'test_{test}.txt',[
                f'BIOHUB_C037_CHECKPOINT={out/"models"/(training+"_step600.pt")}',
                'BIOHUB_C037_ALPHA=1.0',f'BIOHUB_C037_CAPTURE_DIR={out/"evaluation_features"}',
                'BIOHUB_CACHE_EDGE_THRESHOLD=0.02']))
            q.run('replay_'+test,pilot.replay(out,name,[s for _,s in evaluation_plan() if s.startswith(test)],out/'e2e'/name))
        task('analyse','analyse');verify_hashes(plan['hashes'])
        paths=[p for f in ['labels','models','graphs','replay','evaluation_features','e2e'] for p in (out/f).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        paths += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.ipynb'] and p.name not in ['plan.json','status.json','launch.json','artifact_hashes.json']]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in paths})
        assert len(q.state['jobs'])==plan['total_jobs'];q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','label','smoke','train','verify_folds','analyse'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--group',choices=GROUPS)
    a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    if a.command=='train':train(out,a.group)
    else:globals()[a.command](out)
