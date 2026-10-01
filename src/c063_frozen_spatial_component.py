"""Finite timing then one fixed real-anchor component study; existing Queue."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import c063_frozen_spatial_extract as extract
import c063_frozen_spatial_fit as fit
from c055_guarded_readmit import read,sha,save_json
from c063_frozen_spatial_study import DEST,STEMS,hashes,verify
from run_last_days_local import Queue,stamp
ROOT=Path(__file__).resolve().parents[1]


def verify_phase(out,phase):
    folder=out/phase;plan=read(folder/'plan.json');state=read(folder/'status.json')
    assert state['status']=='complete_review_required' and state['plan_sha256']==sha(folder/'plan.json')
    assert len(state['jobs'])==plan['total_jobs'] and all(j['returncode']==0 for j in state['jobs'].values())
    verify(plan['hashes']);verify(read(folder/'output_hashes.json'))


def command(name,script,args):
    return dict(name=name,script=str(script),arguments=[str(a) for a in args])


def prepare(out,phase):
    folder=out/phase;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'plan.json').exists(),'No replacement of registered phase'
    verify_phase(out,'prefix_check')
    review=read(out/'prefix_review.json');assert review['status']=='passed'
    assert review['fit_source_sha256']==sha(Path(fit.__file__))
    paths=set(fit.dependencies(out/'prefix',partial=True))
    paths.update([Path(__file__),ROOT/'src/c063_frozen_spatial_study.py',ROOT/'src/run_last_days_local.py',
        ROOT/'src/c063_frozen_spatial_proof.py',out/'prefix_review.json',
        out/'prefix_check/plan.json',out/'prefix_check/output_hashes.json',
        ROOT/'state/c063_preflight/review_prefix_completed.py',
        ROOT/'state/c063_preflight/model/PROTOCOL.md',ROOT/'state/c063_preflight/model/FIT_PROTOCOL.md',
        ROOT/'state/c063_preflight/model/loader_controls.json',
        ROOT/'state/c063_preflight/launch_component.ps1',ROOT/'tools/notify_background_completion.ps1'])
    if phase=='timing':
        commands=[command('actual_gpu_benchmark',Path(fit.__file__),['benchmark','--capture-dir',out/'prefix',
            '--out',out/'timing_result','--source','44b6','--device','cuda'])]
        minutes=5
        timing=dict(note='Fixed30 disposable steps plus exact zero and evidence IO; conservative5min measurement allowance')
        output_dirs=['timing_result']
    else:
        verify_phase(out,'timing')
        bench=read(out/'timing_result/benchmark.json')
        assert bench['status']=='passed' and bench['actual_steps']==30 and bench['actual_samples']==960
        assert not bench['checkpoint_saved'] and bench['source_only'] and bench['zero_proof']['exact_zero_shift']
        paths.update([out/'timing/plan.json',out/'timing/output_hashes.json',out/'timing_result/benchmark.json'])
        stems=fit.expected_stems();remaining=[s for s in stems if s not in STEMS]
        assert len(remaining)==20
        paths.update(extract.dependencies(remaining,proof_dir=out/'prefix'))
        commands=[command('extract_'+s,Path(extract.__file__),['extract','--movie',s,
            '--output-dir',out/'prefix','--proof-dir',out/'prefix']) for s in remaining]
        for source in fit.EMBRYOS:
            commands.append(command('fit_'+source,Path(fit.__file__),['fit','--capture-dir',out/'prefix',
                '--out',out/'component','--source',source,'--device','cuda']))
        for source in fit.EMBRYOS:
            commands.append(command('predict_'+source,Path(fit.__file__),['predict','--capture-dir',out/'prefix',
                '--out',out/'component','--source',source,'--device','cuda']))
        commands.append(command('analyse',Path(fit.__file__),['analyse','--capture-dir',out/'prefix','--out',out/'component']))
        measured=sum(r['compute_seconds'] for r in review['rows'])
        windows=sum(len(extract.movie_plan(s)[2]['first_seen_windows']) for s in STEMS)
        remaining_windows=sum(len(extract.movie_plan(s)[2]['first_seen_windows']) for s in remaining)
        extraction=measured/windows*remaining_windows
        training=2*bench['estimated1200step_seconds']
        minutes=max(10,math.ceil((1.5*(extraction+training)+300)/60))
        timing=dict(measured_prefix_seconds=measured,measured_windows=windows,remaining_windows=remaining_windows,
            predicted_extraction_seconds=extraction,measured_gpu30_seconds=bench['seconds'],
            predicted_two_fit_seconds=training,variation_multiplier=1.5,
            hash_prediction_analysis_io_allowance_seconds=300,whole_original_pairs=16931)
        output_dirs=['prefix','component']
    save_json(folder/'plan.json',dict(created=stamp(),phase=phase,hashes=hashes(paths),commands=commands,
        total_jobs=len(commands),estimated_minutes=minutes,max_start_job_hours=3,timing=timing,
        recipe=fit.RECIPE,output_dirs=output_dirs,no_graph_edits=True,no_kaggle_writes=True,
        gpu='local4070TiSUPER',gpu_allocation='Measured local path faster than remote upload/queue for this bounded study'))
    print(json.dumps(dict(phase=phase,inputs=len(paths),jobs=len(commands),estimated_minutes=minutes,timing=timing)),flush=True)


def run(out,phase):
    folder=out/phase;assert not (folder/'status.json').exists(),'No blind duplicate/resume'
    plan=read(folder/'plan.json');q=Queue(folder,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(folder/'plan.json'));q.save()
        verify(plan['hashes'])
        for job in plan['commands']:
            q.run(job['name'],[sys.executable,'-u',job['script'],*job['arguments']])
        verify(plan['hashes']);assert q.state['plan_sha256']==sha(folder/'plan.json')
        outputs=[p for part in plan['output_dirs'] for p in (out/part).rglob('*') if p.is_file()]
        outputs += [p for p in (folder/'logs').glob('*') if p.is_file()]
        save_json(folder/'output_hashes.json',hashes(outputs));q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb',choices=['prepare','run']);parser.add_argument('--phase',choices=['timing','study'],required=True)
    parser.add_argument('--out',type=Path,default=DEST)
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    globals()[args.verb](out,args.phase)
