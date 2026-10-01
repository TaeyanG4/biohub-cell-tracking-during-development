#!/usr/bin/env python3
"""Isolate usable mechanisms from Amanatar public V6 using existing local tools.

No Kaggle operations. No new scorer. Never run concurrently with C038 follow-up.
"""
from __future__ import annotations
import argparse
import ast
import collections
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reid_probe_local import BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue
from c037_transformer_study import engine, policy, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, HEAD

DEST = ROOT / 'experiments/candidates/c039_public_v6_salvage'
SOURCE = ROOT / 'tmp/c023_output/tracking_repo'
PUBLIC = ROOT / 'state/notebook_radar/pulled/review_20260927_user_screenshot/amanatar__optimized-biohub-max-score/optimized-biohub-max-score.ipynb'
MODES = ['off', 'primary_max', 'calibrated_max']
ANCHOR = '''                    blended_det = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )'''


def fusion_patch():
    return ANCHOR + '''
                    _c039_mode = os.environ.get('BIOHUB_C039_FUSION', 'off')
                    if _c039_mode == 'calibrated_max':
                        _sec_m = secondary_det_aligned.mean()
                        _sec_s = secondary_det_aligned.std() + 1e-7
                        _pri_m = primary_det.mean()
                        _pri_s = primary_det.std() + 1e-7
                        _sec_calib = (secondary_det_aligned - _sec_m) / _sec_s * _pri_s + _pri_m
                        blended_det = torch.maximum(primary_det,
                            (1.0 - secondary_detection_weight) * primary_det
                            + secondary_detection_weight * _sec_calib)
                    elif _c039_mode == 'primary_max':
                        blended_det = torch.maximum(primary_det, blended_det)
                    elif _c039_mode != 'off':
                        raise ValueError('Unknown C039 fusion mode: ' + _c039_mode)
'''


def codes(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def save_nb(path, nb):
    for i, cell in enumerate(codes(nb)):
        compile(''.join(cell['source']), f'{path.name}:cell{i}', 'exec')
    path.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise RuntimeError('Registered C039 already exists; do not rebuild or overwrite it')
    repo = out / 'tracking_repo'
    for part in ['scripts', 'src']:
        shutil.copytree(SOURCE / part, repo / part, ignore=shutil.ignore_patterns('__pycache__'))
    script = repo / 'scripts/predict_unet_transformer.py'
    original = script.read_text(encoding='utf-8')
    assert original.count(ANCHOR) == 1
    patched = original.replace(ANCHOR, fusion_patch(), 1)
    compile(patched, str(script), 'exec'); script.write_text(patched, encoding='utf-8')
    # Portable detector notebooks use exactly the same predictor replacement.
    for mode in MODES[1:]:
        nb = json.loads(BASE.read_text(encoding='utf-8')); cc = codes(nb)
        cc[0]['source'] = (''.join(cc[0]['source']) + f"\nos.environ['BIOHUB_C039_FUSION'] = {mode!r}\n").splitlines(keepends=True)
        code = ''.join(cc[4]['source']); anchor = '_ps.write_text(_trial_source)\n'
        assert code.count(anchor) == 1
        extra = (f"\n_c039_old = {ANCHOR!r}\n_c039_new = {fusion_patch()!r}\n"
                 "_c039_code = _ps.read_text(encoding='utf-8')\nassert _c039_code.count(_c039_old) == 1\n"
                 "_c039_code = _c039_code.replace(_c039_old, _c039_new, 1)\n"
                 "compile(_c039_code, str(_ps), 'exec')\n_ps.write_text(_c039_code)\n")
        cc[4]['source'] = code.replace(anchor, anchor + extra, 1).splitlines(keepends=True)
        save_nb(out / f'{mode}.ipynb', nb)
    # Only the published division function is transplanted; all C023 settings remain.
    public_nb = json.loads(PUBLIC.read_text(encoding='utf-8')); public_code = ''.join(codes(public_nb)[5]['source'])
    node = next(n for n in ast.parse(public_code).body if isinstance(n, ast.FunctionDef) and n.name == 'add_safe_divisions_postlink')
    public_fn = ast.get_source_segment(public_code, node)
    public_fn = public_fn.replace('def add_safe_divisions_postlink(', 'def _c039_public_division(', 1)
    nb = json.loads(BASE.read_text(encoding='utf-8')); cc = codes(nb); code = ''.join(cc[5]['source'])
    addition = '''
# C039: fixed public division algorithm; repair only the two missing bindings.
SAFE_DIV_HORIZON_FRAMES = int(os.environ.get('BIOHUB_SAFE_DIV_HORIZON_FRAMES', '3'))
SAFE_DIV_HORIZON_DIVERGE_UM = float(os.environ.get('BIOHUB_SAFE_DIV_HORIZON_DIVERGE_UM', '2.8'))
if SAFE_DIV_HORIZON_FRAMES < 1 or not np.isfinite(SAFE_DIV_HORIZON_DIVERGE_UM) or SAFE_DIV_HORIZON_DIVERGE_UM < 0:
    raise ValueError('Invalid C039 public division horizon settings')
C039_DIVISION_MODE = 'off'
_c039_base_division = add_safe_divisions_postlink
''' + public_fn + '''
def add_safe_divisions_postlink(*args, **kwargs):
    if C039_DIVISION_MODE == 'off':
        return _c039_base_division(*args, **kwargs)
    if C039_DIVISION_MODE != 'public':
        raise ValueError('Invalid C039 division mode')
    return _c039_public_division(*args, **kwargs)
'''
    anchor = '\nwrite_test_submission("base")\n'; assert code.count(anchor) == 1
    cc[5]['source'] = code.replace(anchor, addition + anchor, 1).splitlines(keepends=True)
    save_nb(out / 'division_replay.ipynb', nb)
    save_json(out / 'division_variants.json', {'control': {'C039_DIVISION_MODE': 'off'}, 'public_division': {'C039_DIVISION_MODE': 'public'}})
    save_json(out / 'variants.json', {'as_configured': {}})
    for split in ['heldout12', 'confirm10']:
        (out / f'{split}.txt').write_text('\n'.join(s for sp, s in evaluation_plan() if sp == split) + '\n')
    inputs = [Path(__file__), BASE, PUBLIC, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, HEAD,
              ROOT / 'src/run_kaggle_predict_local.py', ROOT / 'src/eval_pp_variants_local.py',
              ROOT / 'src/run_last_days_local.py', ROOT / 'src/v1284_capture_local.py',
              ROOT / 'src/c037_transformer_study.py', ROOT / 'src/c037_transformer_runtime.py',
              ROOT / 'src/reid_probe_local.py', ROOT / 'src/reid_augmented_local.py']
    inputs += [p for p in repo.rglob('*.py')] + [p for p in out.glob('*.ipynb')]
    inputs += list(out.glob('*.txt')) + list(out.glob('*variants.json'))
    inputs += [OLD / f'control_fp32_{sp}.csv' for sp in ['heldout12', 'confirm10']]
    save_json(out / 'plan.json', dict(created=stamp(), arms=MODES[1:] + ['public_division'],
        source_script_version=353147685, public_score_claim='0.965+ unverified; downloaded V6 falls back on all4 visible movies',
        hashes={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        policy='C023 settings frozen; exact original/off smoke, then existing official22 replay; local FP32/math-SDPA',
        next='Run only when C038 followup_extension has stopped; no active waiting or GPU contention.'))
    print('C039 prepared; original C023 and active C038 inputs unchanged', flush=True)


def branch_check(out):
    """Execute the published failing branch on a tiny geometry fixture, CPU only."""
    out.mkdir(parents=True, exist_ok=True)
    nb = json.loads(PUBLIC.read_text(encoding='utf-8')); source = ''.join(codes(nb)[5]['source'])
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'add_safe_divisions_postlink')
    fn_source = ast.get_source_segment(source, fn)
    # Dependencies unrelated to the missing bindings are fixed; no GT or score simulation.
    ns = dict(np=np, OUTPUT_SAFE_DIVISIONS=True, SAFE_DIV_GLOBAL_FRAC_CAP=.006,
              SAFE_DIV_FRAME_FRAC_CAP=.012, SAFE_DIV_REQUIRE_MUTUAL_NN=False,
              SAFE_DIV_EXISTING_CHILD_MAX_UM=10., SAFE_DIV_MAX_UM=10.5,
              SAFE_DIV_SISTER_MAX_UM=15., SAFE_DIV_REQUIRE_DIVERGENCE=True,
              SAFE_DIV_DIVERGE_UM=1.5, DEEPCENTER_SAFE_DIV_VETO=False,
              SAFE_DIV_SISTER_SYMMETRY_TAU=.6)
    ns['edge_distance_um'] = lambda a,b: float(np.linalg.norm(np.array([a[k]-b[k] for k in ['z','y','x']])))
    exec(compile(fn_source, 'publicV6:division_branch_regression', 'exec'), ns)
    def nodes(separation=2., successor_separation=4.):
        return {0:dict(t=0,z=0.,y=0.,x=0.),
                1:dict(t=1,z=0.,y=0.,x=-separation/2), 2:dict(t=1,z=0.,y=0.,x=separation/2),
                3:dict(t=2,z=0.,y=0.,x=-successor_separation/2), 4:dict(t=2,z=0.,y=0.,x=successor_separation/2)}
    edges=[dict(source_id=a,target_id=b) for a,b in [(0,1),(1,3),(2,4)]]
    rows=[]
    for stage,expected in [('original','SAFE_DIV_HORIZON_FRAMES'),('first_binding_only','SAFE_DIV_HORIZON_DIVERGE_UM'),('both_bindings',None)]:
        if stage=='first_binding_only': ns['SAFE_DIV_HORIZON_FRAMES']=3
        if stage=='both_bindings': ns['SAFE_DIV_HORIZON_DIVERGE_UM']=2.8
        try:
            result=ns['add_safe_divisions_postlink'](nodes(),edges,collections.defaultdict(int))
        except NameError as exc:
            assert expected is not None and expected in str(exc)
            rows.append(dict(stage=stage,outcome='expected_NameError',message=str(exc)))
        else:
            assert expected is None and (0,2) in {(e['source_id'],e['target_id']) for e in result}
            rows.append(dict(stage=stage,outcome='passed',added_edges=len(result)-len(edges)))
    result=ns['add_safe_divisions_postlink'](nodes(),edges[:1],collections.defaultdict(int))
    assert len(result)==1
    rows.append(dict(stage='repaired_missing_successors',outcome='rejected_as_expected'))
    result=ns['add_safe_divisions_postlink'](nodes(4.),edges[:1],collections.defaultdict(int))
    assert len(result)==2
    rows.append(dict(stage='wide_sisters_without_successors',outcome='accepted_by_published_shortcut'))
    save_json(out/'division_branch_regression.json',dict(status='passed',rows=rows,
        note='Executable branch regression only, not biological validation or official metric. Both missing globals fixed in generated notebook; no branch bypass.'))
    print(json.dumps(rows),flush=True)


def smoke(out):
    os.environ['BIOHUB_C039_FUSION'] = 'off'
    pu = engine(out)
    code = (SOURCE / 'scripts/predict_unet_transformer.py').read_text(encoding='utf-8').replace(
        'Path("/kaggle/working")', "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    spec = importlib.util.spec_from_loader('c039_original', loader=None)
    base = importlib.util.module_from_spec(spec); base.__file__ = str(SOURCE / 'scripts/predict_unet_transformer.py')
    sys.modules[spec.name] = base; exec(compile(code, base.__file__, 'exec'), base.__dict__)
    device = torch.device('cuda'); model, w, ds = pu.load_model(PRIMARY_WEIGHTS, device)
    secondary, w2, ds2 = pu.load_model(SECONDARY_WEIGHTS, device); assert w == w2 == 2 and ds == ds2
    cfg = pu.PredictConfig(det_threshold=float(os.environ['BIOHUB_DET_THRESHOLD']))
    cfg.threshold = float(os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'])
    kw = dict(cfg=cfg, window_size=w, max_frames=4, downsample=ds, secondary_model=secondary,
        secondary_edge_weight=float(os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT']),
        secondary_detection_weight=float(os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT']),
        secondary_link_mode=os.environ['BIOHUB_SECONDARY_LINK_MODE'],
        secondary_mix_temperature=float(os.environ.get('BIOHUB_SECONDARY_MIX_TEMPERATURE', '1')),
        secondary_low_margin_max=float(os.environ.get('BIOHUB_SECONDARY_LOW_MARGIN_MAX', '.35')))
    rows = []
    with (out / 'smoke.log').open('w', encoding='utf-8') as log:
        for stem in ['44b6_12dfb391', '6bba_05db0fb1']:
            ref = None
            for mode in ['original'] + MODES:
                eng = base if mode == 'original' else pu
                os.environ['BIOHUB_C039_FUSION'] = 'off' if mode == 'original' else mode
                eng._LOWDET.clear(); eng._CACHE_EDGES.clear()
                with contextlib.redirect_stdout(log):
                    coords, edges = eng.predict_video(model, ROOT / 'data/train' / stem, device, **kw)
                coords, edges = np.asarray(coords), np.asarray(edges)
                assert np.isfinite(coords).all() and np.isfinite(edges).all()
                low = [(a.copy(), b.copy()) for a, b in eng._LOWDET]
                if mode == 'original': ref = (coords.copy(), edges.copy(), low)
                equal = np.array_equal(coords, ref[0]) and np.array_equal(edges, ref[1])
                low_equal = len(low) == len(ref[2]) and all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(low,ref[2]))
                if mode == 'off': assert equal and low_equal, 'off mode changed original inference'
                rows.append(dict(stem=stem, mode=mode, nodes=len(coords), edges=len(edges),
                                 exact_original=equal, exact_lowdet=low_equal))
    save_json(out / 'smoke.json', dict(status='passed', rows=rows, runtime=policy(), note='Four-frame execution, not an official score.'))


def analyse(out):
    baseline = pd.concat([pd.read_csv(OLD / f'control_fp32_{sp}.csv') for sp in ['heldout12', 'confirm10']])
    div = pd.concat([pd.read_csv(out / 'replay' / f'division_{sp}.csv') for sp in ['heldout12', 'confirm10']])
    a, b = div[div.config == 'control'].set_index('stem').sort_index(), baseline.set_index('stem').sort_index()
    assert list(a.index) == list(b.index) and len(a) == 22
    for k in ['nodes','edges','edge_tp','edge_fp','edge_fn','div_tp','div_fp','div_fn','adjusted_edge_jaccard']:
        assert np.allclose(a[k], b[k], rtol=0, atol=1e-10), ('division_off', k)
    with contextlib.redirect_stdout(io.StringIO()): ns, _, _ = setup_ns(CONTROL / 'control_fp32_heldout12')
    rows = []
    for arm in MODES[1:] + ['public_division']:
        frame = div[div.config == arm] if arm == 'public_division' else pd.concat([
            pd.read_csv(out / 'replay' / f'{arm}_{sp}.csv') for sp in ['heldout12','confirm10']])
        assert len(frame) == 22 and set(frame.stem) == set(baseline.stem)
        groups = {'all22': set(baseline.stem), **{g:{s for s in baseline.stem if s.startswith(g)} for g in ['44b6','6bba']},
                  **{sp:{s for split,s in evaluation_plan() if split==sp} for sp in ['heldout12','confirm10']}}
        for group, stems in groups.items():
            current, ref = frame[frame.stem.isin(stems)], baseline[baseline.stem.isin(stems)]
            cur, base = [ns['aggregate_official'](x.to_dict('records')) for x in [current,ref]]
            delta = current.set_index('stem').adjusted_edge_jaccard-ref.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(arm=arm, group=group, score=cur['proxy_score'], delta=cur['proxy_score']-base['proxy_score'],
                edge_delta=cur['adjusted_edge_jaccard']-base['adjusted_edge_jaccard'], wins=int((delta>1e-10).sum()),
                losses=int((delta < -1e-10).sum()), div_tp=cur['div_tp'], div_fp=cur['div_fp'], div_fn=cur['div_fn']))
    pd.DataFrame(rows).to_csv(out / 'official_summary.csv', index=False)
    save_json(out / 'analysis.json', dict(status='complete_review_required', rows=rows, division_off_controls=22,
        note='C023-only isolated changes. Not a reproduction or validation of the advertised0.965+.'))


def run(out):
    other = ROOT / 'experiments/candidates/c038_complementary_fusion/followup_extension/status.json'
    if json.loads(other.read_text(encoding='utf-8'))['status'] == 'running': raise RuntimeError('C038 still running; defer without waiting')
    plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / p) == v for p,v in plan['hashes'].items()), 'C039 source/input drift'
    q = Queue(out, 10)
    try:
        q.state['plan_sha256'] = sha(out / 'plan.json'); q.save()
        q.run('division_branch_regression', [sys.executable,'-u',Path(__file__),'branch_check','--out',out])
        q.run('actual_off_smoke', [sys.executable,'-u',Path(__file__),'smoke','--out',out])
        for split in ['heldout12','confirm10']:
            stems = [s for sp,s in evaluation_plan() if sp == split]
            control = CONTROL / f'control_fp32_{split}'
            q.run('division_'+split, [sys.executable,'-u',ROOT/'src/eval_pp_variants_local.py',
                '--notebook',out/'division_replay.ipynb','--variants',out/'division_variants.json',
                '--stems',','.join(stems),'--pred-root',control/'predictions','--lowdet-dir',control/'edge_cache',
                '--round-coords','--out',out/'replay'/f'division_{split}.csv'])
        for mode in MODES[1:]:
            for split in ['heldout12','confirm10']:
                label = mode+'_'+split; run_dir = out/'e2e'/label
                q.run('infer_'+label, [sys.executable,'-u',ROOT/'src/run_kaggle_predict_local.py',
                    '--repo',out/'tracking_repo','--notebook',BASE,'--stems-file',out/f'{split}.txt',
                    '--out',out/'e2e','--label',label,'--v1284-mode','candidate','--v1284-head',HEAD,
                    '--t4-fp32','--env','BIOHUB_C039_FUSION='+mode])
                q.run('replay_'+label, [sys.executable,'-u',ROOT/'src/eval_pp_variants_local.py',
                    '--notebook',out/f'{mode}.ipynb','--variants',out/'variants.json',
                    '--stems',','.join(s for sp,s in evaluation_plan() if sp==split),
                    '--pred-root',run_dir/'predictions','--lowdet-dir',run_dir/'edge_cache',
                    '--round-coords','--out',out/'replay'/f'{label}.csv'])
        q.run('analyse', [sys.executable,'-u',Path(__file__),'analyse','--out',out])
        assert all(sha(ROOT/p)==v for p,v in plan['hashes'].items()), 'C039 input changed'
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','branch_check','smoke','run','analyse'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args(); out=args.out.resolve(); out.relative_to(ROOT)
    globals()[args.command](out)
