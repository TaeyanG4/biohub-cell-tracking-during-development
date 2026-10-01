"""C066 deployment: reuse C053 portable execution, use direct fit and real metric."""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timedelta
import collections
import inspect
import json
import os
import shutil
import sys
import types
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np
import pandas as pd
import torch
import c053_fixed_division_transformer as original
import c058_localizer_baseline as official
import evaluate_local
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,sha,save_json,stamp

DEST=ROOT/'experiments/candidates/c066_expanded_ordinary/deploy'
FIT=DEST.parent/'fit'
HISTORY=ROOT/'state/validation_audit_20260929'

def install_pooled_model(out):
    assert read(ROOT/'state/c066_fit_review.json')['status']=='passed'
    assert read(FIT/'status.json')['status']=='complete_review_required'
    verify_hashes(read(FIT/'plan.json')['hashes']);verify_hashes(read(FIT/'output_hashes.json'))
    source=FIT/'models/pooled_step600.pt';ckpt=torch.load(source,map_location='cpu',weights_only=True)
    assert ckpt['train_embryo']=='pooled' and ckpt['step']==600
    target=out/'transformer_mean600.pt'  # Existing portable asset name; contents are NOT an average.
    shutil.copy2(source,target);assert sha(source)==sha(target)
    save_json(out/'mean_provenance.json',dict(source='C066 directly fitted pooled checkpoint; compatibility asset filename only',
        model_sha256=sha(target),source_checkpoint=str(source.relative_to(ROOT)),
        train_movies=len(ckpt['train_movies']),train_embryo='pooled',step=600,parameter_averaging=False,
        fit_review=read(ROOT/'state/c066_fit_review.json'),
        local_evaluation='Pooled fit-domain technical evidence; C066 opposite-embryo97 transfer supplies scientific support; C052 is its comparator',
        no_new_training=False,training_completed_in_prior_phase=True))

def make_reused_module():
    source=Path(original.__file__).read_text(encoding='utf-8')
    source=source.replace('C053','C066').replace('c053','c066')
    start=source.index('def average_model(out):');end=source.index('\n\ndef prepare(out):',start)
    source=source[:start]+'def average_model(out):\n    return _install_pooled_model(out)\n'+source[end:]
    assert source.count('def analyse(out):')==1
    source=source.replace('def analyse(out):','def replica_analyse(out):',1)
    source=source.replace('No new training.','Training completed in the pinned C066 fit phase; no fitting in this deployment queue.')
    assert source.count("'fixed_both_no_routing' in log")==1
    source=source.replace("'fixed_both_no_routing' in log", "'pooled' in log")
    module=types.ModuleType('c066_reused_portable_driver')
    module.__file__=str(Path(__file__).resolve());module.__dict__['_install_pooled_model']=install_pooled_model
    exec(compile(source,str(Path(__file__))+'::reused_C053','exec'),module.__dict__)
    module.DEST=DEST;module.SLUG='biohub-c066-expanded-ordinary-transformer'
    module.analyse=lambda out:analyse_actual(module,out)
    return module,source

def prepare(module,derived):
    assert not (DEST/'plan.json').exists(),'Registered deployment is immutable'
    transfer=DEST.parent/'transfer97'
    assert read(transfer/'status.json')['status']=='complete_review_required'
    assert read(transfer/'decision.json')['status']=='advance_pooled_deployment'
    assert read(ROOT/'state/c066_transfer_review.json')['status']=='passed'
    verify_hashes(read(transfer/'plan.json')['hashes'])
    verify_hashes(read(transfer/'output_hashes.json'))
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'derived_portable_driver.py').write_text(derived,encoding='utf-8')
    save_json(DEST/'reuse_contract.json',dict(original_source='src/c053_fixed_division_transformer.py',
        original_sha256=sha(Path(original.__file__)),derived_sha256=sha(DEST/'derived_portable_driver.py'),
        adaptations=['self identifiers C053->C066','replace parameter average with byte-exact directly fitted checkpoint',
          'retain legacy analysis as explicitly named replica_analyse for frozen detector controls',
          'actual organizer97 aggregation replaces promotion metric and visible4 expectation'],
        unchanged_execution=['portable predictor patch','inference command','FP32 mathSDPA','C023postprocessing',
          'twofullmovie cacheILP equality','portable12replay','actual productionwriter4','queue and exact verification'],
        no_prefix_router=True))
    module.prepare(DEST)
    # Finalize registration before any child job starts; original C053 files are untouched.
    plan=read(DEST/'plan.json');inputs={ROOT/'src/c053_fixed_division_transformer.py',ROOT/'src/c058_localizer_baseline.py',
        ROOT/'state/c066_fit_review.json',FIT/'plan.json',FIT/'output_hashes.json',FIT/'verification_pooled.json',
        HISTORY/'actual_per_movie.csv',HISTORY/'independent_verification.json',DEST.parent/'README.md'}
    inputs.update(ROOT/p for p in read(FIT/'output_hashes.json'))
    inputs.add(ROOT/'experiments/candidates/c065_pooled_transformer/deploy/official_per_movie97.csv')
    inputs.update([transfer/'plan.json',transfer/'output_hashes.json',transfer/'decision.json',
                   transfer/'official_per_movie97.csv',transfer/'analysis.json',ROOT/'state/c066_transfer_review.json'])
    inputs.update(ROOT/p for p in read(transfer/'output_hashes.json'))
    inputs.update(evaluate_local.VENDOR_SRC.rglob('*.py'))
    plan['hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
    plan.update(candidate='c066',production_refit=True,parameter_averaging=False,
        estimated_minutes=145,completion_target='2026-09-30T00:00:00+09:00',
        policy='One directly fitted checkpoint, fixed C023 pipeline. Pooled fit-domain97; actual organizer promotion metric. No Kaggle writes in queue.',
        official_metric_commit=evaluate_local.METRIC_COMMIT)
    save_json(DEST/'plan.json',plan);print(json.dumps(dict(inputs=len(plan['hashes']),jobs=plan['total_jobs'],estimated_minutes=145)),flush=True)

def analyse_actual(module,out):
    # Retain proven all97 frozen-detector and historical graph/replay checks.
    metadata=[]
    for split in ['heldout12','confirm10','extension75']:
        records=[]
        for line in (out/'e2e'/split/'predict.log').read_text(encoding='utf-8').splitlines():
            if 'C037_PRIMARY ' not in line: continue
            value=ast.literal_eval(line.split('C037_PRIMARY ',1)[1].strip())
            assert value['train_embryo']=='pooled' and value['step']==600 and value['alpha']==1.
            assert Path(value['path']).resolve()==(out/'dataset/transformer_mean600.pt').resolve()
            records.append(value)
        assert records
        metadata.append(dict(split=split,records=records))
    save_json(out/'runtime_metadata_proof.json',dict(status='passed',rows=metadata))
    module.replica_analyse(out)
    for before,after in [('official_per_movie97.csv','replica_per_movie97.csv'),
                         ('official_summary.csv','replica_summary.csv'),('analysis.json','replica_analysis.json')]:
        assert not (out/after).exists();(out/before).replace(out/after)
    plan=read(out/'plan.json');rows=[]
    for job in plan['jobs']:
        for stem in job['stems']:
            path=out/'graphs'/('combined_'+job['split'])/(stem+'_final.npz')
            with np.load(path) as g:
                assert np.issubdtype(g['txyz'].dtype,np.integer)
                row=official.official_score(g['ids'],g['txyz'],g['edges'],ROOT/'data/train'/(stem+'.geff'))
            rows.append(dict(candidate='C066',stem=stem,split=job['split'],embryo=stem[:4],**row))
    data=pd.DataFrame(rows);assert len(data)==97 and data.stem.is_unique
    historical=pd.concat([pd.read_csv(HISTORY/'actual_per_movie.csv'),
        pd.read_csv(ROOT/'experiments/candidates/c065_pooled_transformer/deploy/official_per_movie97.csv')],ignore_index=True)
    summarise=evaluate_local._load_official()[-1]
    result=[]
    groups=[('all97',data),('all22',data[data.split!='extension75'])]
    groups += [(s,data[data.split==s]) for s in ['heldout12','confirm10','extension75']]
    groups += [(e,data[data.embryo==e]) for e in ['44b6','6bba']]
    for group,part in groups:
        current=summarise(part.to_dict('records'));entry=dict(group=group,movies=len(part),score=current['score'],
            adjusted_edge=current['adj_edge_jaccard'],division_jaccard=current['division_jaccard'])
        for column in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']:
            entry[column]=int(part[column].sum())
        for name in ['C023','C052','C053','C054','C065']:
            baseline=historical[(historical.candidate==name)&historical.stem.isin(part.stem)]
            assert set(baseline.stem)==set(part.stem)
            summary=summarise(baseline.to_dict('records'));entry['delta_vs_'+name]=current['score']-summary['score']
            if name=='C023':
                delta=part.set_index('stem').adj_edge_jaccard-baseline.set_index('stem').adj_edge_jaccard
                entry.update(edge_wins=int((delta>1e-12).sum()),edge_losses=int((delta< -1e-12).sum()))
        result.append(entry)
    data.to_csv(out/'official_per_movie97.csv',index=False);pd.DataFrame(result).to_csv(out/'official_summary.csv',index=False)
    visible=data[data.stem.isin(module.STEM_SETS['vis4'])];assert len(visible)==4
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=result,
        visible4_expected=summarise(visible.to_dict('records'))['score'],exact_frozen_detector_movies=97,
        actual_organizer_metric=True,metric_commit=evaluate_local.METRIC_COMMIT,
        warning='Direct pooled production refit;97 is fit-domain, NOT independent validation. C066 whole-embryo97 transfer is reviewed separately.'))
    print(json.dumps(dict(actual_official97=result,visible4_expected=read(out/'analysis.json')['visible4_expected'])),flush=True)

def run_notified(module):
    # Observer-only alias for the existing notifier's documented terminal names.
    # Keep the scientific queue status/receipts and all execution untouched.
    path=ROOT/'state/background_notifications/c066_deploy_notifier_status.json'
    save_json(path,dict(status='running',pid=os.getpid(),started=stamp(),source_status=str((DEST/'status.json').relative_to(ROOT))))
    try:
        module.run(DEST)
        assert read(DEST/'status.json')['status']=='complete_local_verified_review_required'
    except BaseException as exc:
        save_json(path,dict(status='failed',ended=stamp(),error=str(exc),source_status=str((DEST/'status.json').relative_to(ROOT))))
        raise
    save_json(path,dict(status='complete',ended=stamp(),source_status=str((DEST/'status.json').relative_to(ROOT)),
        source_status_sha256=sha(DEST/'status.json'),original_terminal_status='complete_local_verified_review_required',notification_alias_only=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','check_smoke','analyse','write_visible','verify_local'])
    parser.add_argument('--out',type=Path,default=DEST);args=parser.parse_args()
    assert args.out.resolve()==DEST.resolve()
    module,derived=make_reused_module()
    if args.command=='prepare':prepare(module,derived)
    else:
        if args.command=='run':
            assert datetime.now().astimezone()+timedelta(minutes=145+60+20)<datetime.fromisoformat('2026-09-30T00:00:00+09:00'), 'Full remaining deployment path no longer fits midnight'
            run_notified(module)
        else:getattr(module,args.command)(DEST)
