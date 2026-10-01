#!/usr/bin/env python3
"""Package C041 fixed models as C042/C043; reuse existing inference/replay/Queue.

No Kaggle writes. prepare is immutable once registered; run is background work.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from c041_fixed_models import ROOT, BASE, DEST as STUDY, sha, save_json, stamp
from c037_transformer_study import SOURCE, DEST as C037, HEAD
from c038_followup_local import verify, graph
from eval_pp_variants_local import build_namespace, load_raw_graph, STEM_SETS
from run_last_days_local import Queue, replay_command

DEST = ROOT/'state/c042_c043_portable'
DATASET = 'taeyangg4/biohub-fixed-models-c041'
ARMS = {'c042': ('transformer', 'off'), 'c043': ('transformer-appearance', 'appearance')}
BEGIN = '# BEGIN C041 FIXED MODEL PATCH\n'
END = '# END C041 FIXED MODEL PATCH\n'


def candidate(key):
    return ROOT/'experiments/candidates'/f'{key}_fixed_{ARMS[key][0].replace("-", "_")}'


def notebook(key):
    return candidate(key)/f'biohub-{key}-fixed-{ARMS[key][0]}.ipynb'


def numerical_policy():
    tree = ast.parse((ROOT/'src/run_kaggle_predict_local.py').read_text(encoding='utf-8'))
    return next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'block' for t in n.targets)
                and isinstance(n.value, ast.Constant) and str(n.value.value).startswith('import torch\n'))


def predictor_patches():
    return [
        ('import tracksdata as td\n', 'import tracksdata as td\nfrom c037_transformer_runtime import capture as _c037_capture, apply_primary as _c037_apply\n'),
        ('    model, window_size, downsample = load_model(weights_path, device)\n',
         '    model, window_size, downsample = load_model(weights_path, device)\n    _c037_apply(model)\n'),
        ('            _bidirectional_weight = float(\n',
         '            _c037_capture(ds_path, t_src, t_tgt, unet_feat_src, unet_feat_tgt,\n'
         '                          p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n'
         '                          p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt, edge_logits_pair)\n\n'
         '            _bidirectional_weight = float(\n'),
    ]


def asset_source(hashes):
    return '''
# Fixed model assets: identical files for every movie, checked before inference.
import hashlib as _fixed_hashlib
_FIXED_HASHES = ''' + repr(hashes) + '''
def _fixed_asset(name):
    local = os.environ.get('BIOHUB_FIXED_ASSET_DIR', '').strip()
    if local:
        candidates = [Path(local)/name]
    else:
        candidates = [Path('/kaggle/input/biohub-fixed-models-c041')/name,
                      Path('/kaggle/input/datasets/taeyangg4/biohub-fixed-models-c041')/name]
        if not any(p.is_file() for p in candidates):
            candidates += list(Path('/kaggle/input').rglob('biohub-fixed-models-c041/'+name))
    found = next((p for p in candidates if p.is_file()), None)
    if found is None or _fixed_hashlib.sha256(found.read_bytes()).hexdigest() != _FIXED_HASHES[name]:
        raise RuntimeError('Missing or changed fixed model asset: '+name)
    return found
for _fixed_name in _FIXED_HASHES:
    print('FIXED_ASSET', _fixed_name, str(_fixed_asset(_fixed_name)), _FIXED_HASHES[_fixed_name])
'''


def prepare(out):
    if (out/'plan.json').exists():
        raise RuntimeError('Registered build exists; do not overwrite')
    out.mkdir(parents=True, exist_ok=True)
    assert json.loads((STUDY/'status.json').read_text())['status'] == 'complete_review_required'
    assert json.loads((STUDY/'review_verification.json').read_text())['status'] == 'passed'
    # Check original study sources/results again before deriving deployable assets.
    prior = json.loads((STUDY/'plan.json').read_text())['hashes']
    outputs = json.loads((STUDY/'artifact_hashes.json').read_text())
    assert all(sha(ROOT/p) == h for p,h in prior.items()), 'C041 input drift'
    assert all(sha(STUDY/p) == h for p,h in outputs.items()), 'C041 output drift'
    assets = out/'dataset'; assets.mkdir(exist_ok=True)
    names = ['transformer_mean600.pt', 'appearance_mean_cosine.pt']
    for name in names:
        shutil.copy2(STUDY/name, assets/name)
    hashes = {name:sha(assets/name) for name in names}
    save_json(assets/'dataset-metadata.json', dict(title='Biohub fixed models C041', id=DATASET,
        licenses=[{'name':'CC0-1.0'}]))
    save_json(assets/'provenance.json', dict(created=stamp(), hashes=hashes,
        source_study=str(STUDY.relative_to(ROOT)),
        transformer='Equal parameter mean of two C037 same-base step600 Transformers',
        appearance='Concatenated normalized C035 weak_aug CNNs divided by sqrt(2); mean cosine',
        training='24 movies disjoint from correction-evaluation97; public base trained on199',
        policy='Every movie uses identical fixed models. No GT, prefix routing or new training.'))
    runtime = (ROOT/'src/c037_transformer_runtime.py').read_text(encoding='utf-8')
    appearance = (STUDY/'portable_appearance.py').read_text(encoding='utf-8')
    patches = predictor_patches()
    original = (SOURCE/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    patched = original
    for old,new in patches:
        assert patched.count(old) == 1
        patched = patched.replace(old,new,1)
    assert patched == (C037/'tracking_repo/scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    numeric = numerical_policy()
    assert patched.count('import torch\n') == 1
    patched = patched.replace('import torch\n',numeric,1)
    block = BEGIN + '_fixed_source = _ps.read_text()\n_fixed_patches = ' + repr(patches + [('import torch\n', numeric)]) + '\n'
    block += '''for _fixed_old, _fixed_new in _fixed_patches:
    if _fixed_source.count(_fixed_old) != 1:
        raise RuntimeError('Fixed Transformer predictor anchor mismatch')
    _fixed_source = _fixed_source.replace(_fixed_old, _fixed_new, 1)
compile(_fixed_source, str(_ps), 'exec')
_ps.write_text(_fixed_source)
'''
    block += f"(_ps.parent/'c037_transformer_runtime.py').write_text({runtime!r})\n"
    block += "os.environ['BIOHUB_C037_CHECKPOINT'] = str(_fixed_asset('transformer_mean600.pt'))\n"
    block += "os.environ['BIOHUB_C037_ALPHA'] = '1.0'\nos.environ['BIOHUB_C037_CAPTURE_DIR'] = ''\n" + END
    repo = out/'tracking_repo'
    for part in ['src','scripts']:
        shutil.copytree(SOURCE/part,repo/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    paths = []
    for key, (_, mode) in ARMS.items():
        dest = candidate(key); dest.mkdir(parents=True,exist_ok=True)
        nb = json.loads(BASE.read_text(encoding='utf-8'))
        cells = nb['cells']
        cells[0]['source'] = ''.join(cells[0]['source']).replace("'''Biohub C023:",f"'''Biohub {key.upper()}:",1).splitlines(keepends=True)
        cells[2]['source'] = (''.join(cells[2]['source']) + asset_source(hashes)).splitlines(keepends=True)
        code = ''.join(cells[4]['source']);anchor = '_ps.write_text(_trial_source)\n'
        assert code.count(anchor) == 1
        cells[4]['source'] = code.replace(anchor,anchor+'\n'+block,1).splitlines(keepends=True)
        if mode == 'appearance':
            code = ''.join(cells[5]['source']);anchor = '\nwrite_test_submission("base")\n'
            assert code.count(anchor) == 1
            payload = f'\nexec(compile({appearance!r}, "c041_portable_appearance", "exec"), globals())\n'
            payload += "install_complementary(globals(), str(_fixed_asset('appearance_mean_cosine.pt')))\nC038_MODE = 'appearance'\n"
            cells[5]['source'] = code.replace(anchor,payload+anchor,1).splitlines(keepends=True)
        for i,c in enumerate(cells):
            if c['cell_type']=='code':
                text=''.join(c['source']);compile(text,f'{key}:cell{i}','exec')
                assert 'H:\\' not in text and 'H:/' not in text and 'from c038_followup_local' not in text
                c['outputs']=[];c['execution_count']=None
        path = notebook(key);nb['metadata']['title']=path.stem
        path.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
        meta = json.loads((BASE.parent/'kernel-metadata.json').read_text())
        meta.update(id='taeyangg4/'+path.stem,title=path.stem,code_file=path.name)
        meta['dataset_sources'] = sorted(meta['dataset_sources']+[DATASET])
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
        save_json(dest/'kernel-metadata.json',meta)
        # Execute each notebook's actual embedded patch against a fresh C023 predictor.
        ps=repo/'scripts/predict_unet_transformer.py';ps.write_text(original,encoding='utf-8')
        embedded=''.join(cells[4]['source']).split(BEGIN,1)[1].split(END,1)[0]
        exec(compile(embedded,str(path)+':fixed_patch','exec'),dict(_ps=ps,os=os,_fixed_asset=lambda name:assets/name))
        assert ps.read_text(encoding='utf-8') == patched
        assert (repo/'scripts/c037_transformer_runtime.py').read_text(encoding='utf-8') == runtime
        paths += [path,dest/'kernel-metadata.json']
        (dest/'README.md').write_text(f'# {key.upper()} fixed {ARMS[key][0]}\n\nDerived from C023 and reviewed C041. Uniform models for every movie; no embryo routing. Exact SHA-pinned private assets: {DATASET}.\n\nLocal verification queue: state/c042_c043_portable/status.json. Do not push before local_verification.json passes. Actual T4 visible4 and repair_fallback=0 required before exact-version submission. Final picks C023/C024 unchanged.\n',encoding='utf-8')
    # Compare the produced predictor with the actual C041 FP32 inference source,
    # normalizing only the existing local-working-directory adaptation.
    local=patched
    for quoted in ["Path('/kaggle/working')", 'Path("/kaggle/working")']:
        local=local.replace(quoted,"Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    reference=STUDY/'e2e/heldout12/_work/tracking_repo/scripts/predict_unet_transformer.py'
    assert local == reference.read_text(encoding='utf-8'), 'actual C041 predictor differs'
    save_json(out/'as_configured.json',{'as_configured':{}})
    inputs=set(ROOT/p for p in prior)|set(STUDY/p for p in outputs)
    inputs.update(paths+[Path(__file__),BASE,reference,ROOT/'src/c037_transformer_runtime.py'])
    inputs.update(p for p in assets.rglob('*') if p.is_file())
    inputs.update(p for p in repo.rglob('*') if p.is_file())
    inputs.add(out/'as_configured.json')
    save_json(out/'plan.json',dict(created=stamp(),hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)},
        candidates=list(ARMS),smoke_stems=[STEM_SETS['vis4'][0],STEM_SETS['vis4'][2]],
        local_visible4_expected=.9315252396676436,policy='Local checks only. No upload/push/submission in this queue.'))
    print('Prepared C042/C043 portable candidates; embedded predictor == actual C041 FP32 source',flush=True)


def check_smoke(out):
    plan=json.loads((out/'plan.json').read_text());rows=[]
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset')
    with contextlib.redirect_stdout(io.StringIO()):
        ns=build_namespace(notebook('c042'),{},STUDY/'e2e/heldout12/edge_cache')
    for stem in plan['smoke_stems']:
        new=out/'e2e/smoke';old=STUDY/'e2e/heldout12'
        with np.load(new/'edge_cache'/f'{stem}.npz') as a,np.load(old/'edge_cache'/f'{stem}.npz') as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files),('cache drift',stem)
        a,b=[load_raw_graph(ns,next((p/'predictions').rglob(stem+'.geff'))) for p in [new,old]]
        key=lambda e:(e['source_id'],e['target_id'])
        assert a[0]==b[0] and sorted(a[1],key=key)==sorted(b[1],key=key),('ILP drift',stem)
        rows.append(dict(stem=stem,exact_cache=True,exact_ilp=True))
    save_json(out/'smoke.json',dict(status='passed',rows=rows))


def write_visible(out,key):
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset')
    work=out/key;work.mkdir(exist_ok=True)
    ns=build_namespace(notebook(key),{},STUDY/'e2e/heldout12/edge_cache')
    ns.update(TEST_DIR=ROOT/'data/train',REPO_DIR=work/'writer_repo',test_stems=STEM_SETS['vis4'],
        SUBMISSION_PATH=work/'submission.csv',RUN_STATS_PATH=work/'run_stats.csv',predict_seconds=0)
    if key=='c043':assert ns['C038_MODE']=='appearance', 'production default must be appearance'
    for stem in STEM_SETS['vis4']:
        src=STUDY/'e2e/heldout12/predictions'/f'{stem}.geff'
        dest=work/'writer_repo/predictions/portable'/ns['METHOD']/'split_0'/src.name
        shutil.copytree(src,dest,dirs_exist_ok=True)
    ns['write_test_submission']('portable_visible4')
    stats=pd.read_csv(work/'run_stats.csv')
    assert len(stats)==4 and (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()
    data=pd.read_csv(work/'submission.csv')
    for stem in STEM_SETS['vis4']:
        movie=data[data.dataset==stem];n=movie[movie.row_type=='node'];e=movie[movie.row_type=='edge']
        nodes={int(r.node_id):tuple(int(getattr(r,k)) for k in ['t','z','y','x']) for r in n.itertuples()}
        edges={(int(r.source_id),int(r.target_id)) for r in e.itertuples()}
        assert (nodes,edges)==graph(STUDY/'graphs/combined_heldout12',stem,ARMS[key][1]),('writer graph drift',key,stem)
    save_json(work/'writer_parity.json',dict(status='passed',exact_final_graphs=4,repair_fallback=0,
        submission_sha256=sha(work/'submission.csv')))


def verify_local(out):
    ref=pd.read_csv(STUDY/'replay/combined_heldout12.csv');results={}
    edge_sets={}
    for key,(_,mode) in ARMS.items():
        verify(pd.read_csv(out/'replay'/f'{key}_heldout12.csv'),ref,{'as_configured':mode})
        summary=json.loads((out/key/'evaluation/summary.json').read_text())
        expected=json.loads((out/'plan.json').read_text())['local_visible4_expected']
        assert abs(summary['total_score']-expected)<1e-12,(key,summary['total_score'],expected)
        assert json.loads((out/key/'writer_parity.json').read_text())['status']=='passed'
        results[key]=dict(notebook_sha256=sha(notebook(key)),replay12='exact',visible4=summary['total_score'],
            repair_fallback=0,mode=mode,submission_sha256=sha(out/key/'submission.csv'))
        frame=pd.read_csv(out/key/'submission.csv')
        edge_sets[key]={(r.dataset,int(r.source_id),int(r.target_id)) for r in frame[frame.row_type=='edge'].itertuples()}
    save_json(out/'local_verification.json',dict(status='passed',created=stamp(),candidates=results,
        visible4_edge_symmetric_difference=len(edge_sets['c042']^edge_sets['c043']),
        limitation='Not T4 verified yet. Both visible4 scores equal; compare actual graph differences and version provenance before submissions.'))


def run(out):
    plan=json.loads((out/'plan.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift'
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset')
    q=Queue(out,4)
    try:
        q.state['plan_sha256']=sha(out/'plan.json');q.save()
        q.run('portable_inference_smoke',[sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py',
            '--repo',out/'tracking_repo','--notebook',notebook('c042'),'--stems',','.join(plan['smoke_stems']),
            '--out',out/'e2e','--label','smoke','--v1284-mode','candidate','--v1284-head',HEAD,
            '--env',f'BIOHUB_C037_CHECKPOINT={out/"dataset/transformer_mean600.pt"}',
            '--env','BIOHUB_C037_ALPHA=1.0','--env','BIOHUB_C037_CAPTURE_DIR='])
        q.run('check_smoke',[sys.executable,'-u',Path(__file__),'check_smoke','--out',out])
        stems=(STUDY/'heldout12.txt').read_text().split()
        for key in ARMS:
            q.run(key+'_replay12',replay_command(notebook(key),STUDY/'e2e/heldout12',stems,out/'as_configured.json',out/'replay'/f'{key}_heldout12.csv'))
            q.run(key+'_writer4',[sys.executable,'-u',Path(__file__),'write_visible','--out',out,'--key',key])
            q.run(key+'_official4',[sys.executable,'-u',ROOT/'src/evaluate_local.py','--csv',out/key/'submission.csv',
                '--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/key/'evaluation'])
        q.run('verify_local',[sys.executable,'-u',Path(__file__),'verify_local','--out',out])
        assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()),'input drift during validation'
        files=[p for d in ['replay','c042','c043','e2e'] for p in (out/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in files})
        q.close('complete_local_verified')
    except BaseException as exc:
        q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','check_smoke','write_visible','verify_local'])
    parser.add_argument('--out',type=Path,default=DEST);parser.add_argument('--key',choices=list(ARMS))
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    if args.command=='write_visible':write_visible(out,args.key)
    else:globals()[args.command](out)
