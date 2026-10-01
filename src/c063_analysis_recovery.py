"""C063 analysis-only dtype recovery. Original source and predictions immutable."""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import c063_frozen_spatial_fit as fit
from c055_guarded_readmit import read,sha,save_json
from c063_frozen_spatial_study import DEST,hashes,verify
from run_last_days_local import Queue,stamp
ROOT=Path(__file__).resolve().parents[1]
FOLDER=DEST/'analysis_recovery'


def restored_analysis():
    """Change only original CSV shift-array dtype, retaining every gate/assert."""
    original=ast.parse(inspect.getsource(fit.analyse))
    node=copy.deepcopy(original)
    found=[]
    for current in ast.walk(node):
        if isinstance(current,ast.Assign) and len(current.targets)==1 and isinstance(current.targets[0],ast.Name) and current.targets[0].id=='shifts':
            assert ast.unparse(current.value)=="group[['shift_z_um', 'shift_y_um', 'shift_x_um']].to_numpy()"
            found.append(current.value)
    assert len(found)==1
    found[0].keywords=[ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='float32',ctx=ast.Load()))]
    ast.fix_missing_locations(node)
    reverse=copy.deepcopy(node)
    for current in ast.walk(reverse):
        if isinstance(current,ast.Assign) and len(current.targets)==1 and isinstance(current.targets[0],ast.Name) and current.targets[0].id=='shifts':
            current.value.keywords=[]
    assert ast.dump(reverse)==ast.dump(original),'Changes outside original shift dtype'
    namespace=dict(fit.__dict__)
    exec(compile(node,str(Path(__file__))+':original_analyse_fp32_restore','exec'),namespace)
    return namespace['analyse'],ast.unparse(node)


def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True)
    assert not (FOLDER/'plan.json').exists()
    original_plan=read(DEST/'study/plan.json');state=read(DEST/'study/status.json')
    assert state['status']=='failed' and state['plan_sha256']==sha(DEST/'study/plan.json')
    assert len(state['jobs'])==25 and state['jobs']['analyse']['returncode']==1
    assert sum(j['returncode']==0 for j in state['jobs'].values())==24
    verify(original_plan['hashes'])
    assert not (DEST/'component/analysis').exists()
    rows=[]
    for source in fit.EMBRYOS:
        p=DEST/'component/evaluation'/f'pairs_{source}.csv';frame=pd.read_csv(p)
        assert sha(p)==read(p.with_suffix('.json'))['prediction_sha256']
        target=frame[fit.RESIDUAL].to_numpy();shift64=frame[['shift_z_um','shift_y_um','shift_x_um']].to_numpy()
        shift32=shift64.astype(np.float32)
        err64=np.max(np.abs(frame.after_3d_um-np.linalg.norm(target-shift64,axis=1)))
        err32=np.max(np.abs(frame.after_3d_um-np.linalg.norm(target-shift32,axis=1)))
        assert err64>1e-9 and err32<1e-12
        assert np.array_equal(frame.before_3d_um[~frame.eligible],frame.after_3d_um[~frame.eligible])
        rows.append(dict(source=source,rows=len(frame),read_float64_max_error_um=float(err64),
                         restored_float32_max_error_um=float(err32),excluded_zero=True))
    function,generated=restored_analysis()
    (FOLDER/'analysis_adapter.py').write_text(generated+'\n',encoding='utf-8')
    save_json(FOLDER/'dtype_diagnosis.json',dict(status='passed',rows=rows,
        reason='Original float32 shifts serialized to decimal then CSV read defaults float64; restore original precision',
        tolerance_unchanged=1e-9,recipe_and_gates_unchanged=True,only_shift_dtype_restored=True))
    paths={ROOT/p for p in original_plan['hashes']}
    paths.update(fit.dependencies(DEST/'prefix'))
    paths.update(p for part in ['prefix','component','study'] for p in (DEST/part).rglob('*') if p.is_file())
    paths.update([Path(__file__),FOLDER/'analysis_adapter.py',FOLDER/'dtype_diagnosis.json',
        ROOT/'state/c063_review_20260929/launch_recovery.ps1',ROOT/'tools/notify_background_completion.ps1'])
    save_json(FOLDER/'plan.json',dict(created=stamp(),phase='analysis_recovery',hashes=hashes(paths),
        total_jobs=1,estimated_minutes=5,max_start_job_hours=1,no_training=True,no_inference=True,
        no_graph_or_kaggle_write=True,original_complete_jobs=24,change='CSV shifts to_numpy(dtype=np.float32) only'))
    print(json.dumps(dict(inputs=len(paths),jobs=1,estimated_minutes=5,diagnosis=rows)),flush=True)


def analyse():
    plan=read(FOLDER/'plan.json');verify(plan['hashes'])
    assert not (DEST/'component/analysis').exists()
    function,generated=restored_analysis()
    assert (FOLDER/'analysis_adapter.py').read_text(encoding='utf-8')==generated+'\n'
    function(DEST/'prefix',DEST/'component')
    verify(plan['hashes'])


def run():
    assert not (FOLDER/'status.json').exists(),'No blind resume'
    plan=read(FOLDER/'plan.json');q=Queue(FOLDER,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=1,plan_sha256=sha(FOLDER/'plan.json'));q.save()
        verify(plan['hashes'])
        q.run('analysis_dtype_recovery',[sys.executable,'-u',Path(__file__),'analyse'])
        verify(plan['hashes']);assert q.state['plan_sha256']==sha(FOLDER/'plan.json')
        paths=[p for part in ['prefix','component'] for p in (DEST/part).rglob('*') if p.is_file()]
        paths += [p for p in (FOLDER/'logs').glob('*') if p.is_file()]
        save_json(FOLDER/'output_hashes.json',hashes(paths));q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('verb',choices=['prepare','run','analyse'])
    globals()[parser.parse_args().verb]()
