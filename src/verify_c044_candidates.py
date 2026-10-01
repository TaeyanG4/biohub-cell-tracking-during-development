#!/usr/bin/env python3
"""C044/C045 portable checks using existing Queue, notebook writer and scorers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
from c044_c024_fixed_study import ROOT, DEST as STUDY, PORTABLE, nbpath, BEGIN, END
from c041_fixed_models import sha, save_json, stamp
from c038_followup_local import verify, graph
from eval_pp_variants_local import build_namespace, STEM_SETS
from run_last_days_local import Queue, replay_command
from v1284_capture_local import embedded_v1284_module

DEST=ROOT/'state/c044_c045_portable'
MODES={'c044':'off','c045':'appearance'}
RUN=STUDY/'e2e/learned_heldout12'


def prepare(out):
    assert not (out/'plan.json').exists(),'registered verification exists'
    out.mkdir(parents=True,exist_ok=True)
    assert json.loads((STUDY/'status.json').read_text())['status']=='complete_review_required'
    inputs=json.loads((STUDY/'plan.json').read_text())['hashes']
    outputs=json.loads((STUDY/'artifact_hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in inputs.items()),'study input drift'
    assert all(sha(STUDY/p)==h for p,h in outputs.items()),'study artifact drift'
    files=set(ROOT/p for p in inputs)|set(STUDY/p for p in outputs)
    files.update([Path(__file__),STUDY/'status.json',STUDY/'analysis.json',STUDY/'smoke.json',STUDY/'official_summary.csv'])
    expected={};summary=pd.read_csv(STUDY/'replay/learned_heldout12_summary.csv')
    for key,mode in MODES.items():
        path=nbpath(key);nb=json.loads(path.read_text(encoding='utf-8'))
        meta=json.loads((path.parent/'kernel-metadata.json').read_text())
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':
                s=''.join(c['source']);compile(s,f'{key}:{i}','exec')
                assert 'from c038_followup_local' not in s and 'H:/' not in s and 'H:\\' not in s
        temp=out/('patch_'+key);temp.mkdir(exist_ok=True)
        ps=temp/'predict_unet_transformer.py'
        source=ROOT/'tmp/c024_output/tracking_repo/scripts/predict_unet_transformer.py'
        ps.write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
        block=''.join(nb['cells'][4]['source']).split(BEGIN,1)[1].split(END,1)[0]
        exec(compile(block,f'{key}:embedded_patch','exec'),dict(_ps=ps,os=os,_fixed_asset=lambda name:PORTABLE/'dataset'/name))
        predictor=ps.read_text(encoding='utf-8')
        for quoted in ["Path('/kaggle/working')",'Path("/kaggle/working")']:
            predictor=predictor.replace(quoted,"Path(os.environ['BIOHUB_LOCAL_WORKING'])")
        actual=RUN/'_work/tracking_repo/scripts'
        assert predictor==(actual/'predict_unet_transformer.py').read_text(encoding='utf-8'),'embedded predictor drift'
        assert embedded_v1284_module(path)==(actual/'v1284_coordinate_refinement.py').read_text(encoding='utf-8'),'head module drift'
        assert (temp/'c037_transformer_runtime.py').read_text(encoding='utf-8')==(actual/'c037_transformer_runtime.py').read_text(encoding='utf-8')
        expected[key]=float(summary[(summary.config==mode)&(summary.group=='vis4')].iloc[0].proxy_score)
        files.update(actual/name for name in ['predict_unet_transformer.py','v1284_coordinate_refinement.py','c037_transformer_runtime.py'])
        files.update([path,path.parent/'kernel-metadata.json'])
    save_json(out/'as_configured.json',{'as_configured':{}});files.add(out/'as_configured.json')
    save_json(out/'plan.json',dict(created=stamp(),hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        expected_visible4=expected,policy='Seven jobs; exact existing portable notebook execution only. No new inference/training or Kaggle writes.'))
    print('C044/C045 embedded predictor/head/runtime == actual study; prepared7 verification jobs',flush=True)


def write_visible(out,key):
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(PORTABLE/'dataset')
    work=out/key;work.mkdir(exist_ok=True)
    ns=build_namespace(nbpath(key),{},RUN/'edge_cache')
    ns.update(TEST_DIR=ROOT/'data/train',REPO_DIR=work/'writer_repo',test_stems=STEM_SETS['vis4'],
        SUBMISSION_PATH=work/'submission.csv',RUN_STATS_PATH=work/'run_stats.csv',predict_seconds=0)
    if key=='c045':assert ns['C038_MODE']=='appearance','production appearance must be enabled'
    for stem in STEM_SETS['vis4']:
        src=RUN/'predictions'/f'{stem}.geff'
        shutil.copytree(src,work/'writer_repo/predictions/portable'/ns['METHOD']/'split_0'/src.name,dirs_exist_ok=True)
    ns['write_test_submission']('portable_visible4')
    stats=pd.read_csv(work/'run_stats.csv')
    assert len(stats)==4 and (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()
    data=pd.read_csv(work/'submission.csv')
    for stem in STEM_SETS['vis4']:
        movie=data[data.dataset==stem];nodes=movie[movie.row_type=='node'];edges=movie[movie.row_type=='edge']
        actual_nodes={int(r.node_id):tuple(int(getattr(r,k)) for k in ['t','z','y','x']) for r in nodes.itertuples()}
        actual_edges={(int(r.source_id),int(r.target_id)) for r in edges.itertuples()}
        assert (actual_nodes,actual_edges)==graph(STUDY/'graphs/learned_heldout12',stem,MODES[key]),('writer graph drift',key,stem)
    save_json(work/'writer_parity.json',dict(status='passed',exact_final_graphs=4,repair_fallback=0,deadline_degraded=0,
        submission_sha256=sha(work/'submission.csv')))


def verify_local(out):
    reference=pd.read_csv(STUDY/'replay/learned_heldout12.csv');plan=json.loads((out/'plan.json').read_text());results={};edges={}
    for key,mode in MODES.items():
        verify(pd.read_csv(out/'replay'/f'{key}_heldout12.csv'),reference,{'as_configured':mode})
        result=json.loads((out/key/'evaluation/summary.json').read_text())
        assert abs(result['total_score']-plan['expected_visible4'][key])<1e-12
        assert json.loads((out/key/'writer_parity.json').read_text())['status']=='passed'
        results[key]=dict(notebook_sha256=sha(nbpath(key)),replay12='exact',visible4=result['total_score'],
            repair_fallback=0,deadline_degraded=0,mode=mode,submission_sha256=sha(out/key/'submission.csv'))
        frame=pd.read_csv(out/key/'submission.csv');edges[key]={(r.dataset,int(r.source_id),int(r.target_id)) for r in frame[frame.row_type=='edge'].itertuples()}
    save_json(out/'local_verification.json',dict(status='passed',created=stamp(),candidates=results,
        visible4_edge_symmetric_difference=len(edges['c044']^edges['c045']),
        next='Actual T4 exact-version verification before user-authorized submission; no LB generalization claim.'))


def run(out):
    plan=json.loads((out/'plan.json').read_text());assert all(sha(ROOT/p)==h for p,h in plan['hashes'].items()),'input drift'
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(PORTABLE/'dataset');q=Queue(out,3)
    try:
        q.state['plan_sha256']=sha(out/'plan.json');q.save()
        stems=(STUDY/'heldout12.txt').read_text().split()
        for key in MODES:
            q.run(key+'_replay12',replay_command(nbpath(key),RUN,stems,out/'as_configured.json',out/'replay'/f'{key}_heldout12.csv'))
            q.run(key+'_writer4',[sys.executable,'-u',Path(__file__),'write_visible','--out',out,'--key',key])
            q.run(key+'_official4',[sys.executable,'-u',ROOT/'src/evaluate_local.py','--csv',out/key/'submission.csv',
                '--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/key/'evaluation'])
        q.run('verify_local',[sys.executable,'-u',Path(__file__),'verify_local','--out',out])
        assert all(sha(ROOT/p)==h for p,h in plan['hashes'].items()),'input drift during verification'
        files=[p for part in ['replay','c044','c045'] for p in (out/part).rglob('*') if p.is_file()]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files});q.close('complete_local_verified')
    except BaseException as exc:q.close('failed',exc);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','write_visible','verify_local'])
    p.add_argument('--out',type=Path,default=DEST);p.add_argument('--key',choices=list(MODES));a=p.parse_args();out=a.out.resolve();out.relative_to(ROOT)
    if a.command=='write_visible':write_visible(out,a.key)
    else:globals()[a.command](out)
