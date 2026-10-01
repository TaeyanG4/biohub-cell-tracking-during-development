"""Finite frozen DeepCenter axial signal diagnostic; reuse existing Queue."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import c064_deepcenter_axial_probe as probe
from c055_guarded_readmit import read, sha, save_json
from run_last_days_local import Queue, stamp

ROOT = Path(__file__).resolve().parents[1]
DEST = probe.DEST

def hashes(paths):
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(set(paths))}

def verify(table):
    for p, expected in table.items():
        assert sha(ROOT/p) == expected, p

def prepare(phase):
    folder = DEST/phase
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'plan.json').exists(), 'No replacement of registered phase'
    paths = set(probe.dependencies())
    paths.update([Path(__file__), Path(probe.__file__), ROOT/'src/run_last_days_local.py',
                  DEST/'README.md',
                  ROOT/'src/c055_guarded_readmit.py', ROOT/'state/c064_preflight/launch.ps1',
                  ROOT/'tools/notify_background_completion.ps1',
                  ROOT/'state/c064_next_review/geometry.md',
                  ROOT/'state/c064_next_review/RECOMMENDATION.md'])
    controls = read(DEST/'controls.json')
    assert controls['status'] == 'passed'
    assert controls['probe_source_sha256'] == sha(Path(probe.__file__))
    assert controls['sample_sha256'] == sha(probe.SAMPLE)
    assert controls['protocol_sha256'] == sha(probe.PROTOCOL)
    assert controls['checkpoint_sha256'] == sha(probe.CHECKPOINT)
    assert controls['recipe'] == probe.RECIPE
    paths.add(DEST/'controls.json')
    if phase == 'timing':
        commands = ['benchmark']
        minutes = 5
        timing = {'basis': 'Two exact sampled frames; conservative five minute initialization/inference/hash allowance'}
        output_paths = ['benchmark.json', 'heatmaps']
    else:
        old = read(DEST/'timing/plan.json')
        state = read(DEST/'timing/status.json')
        assert state['status'] == 'complete_review_required'
        assert state['plan_sha256'] == sha(DEST/'timing/plan.json')
        assert len(state['jobs']) == 1 and all(j['returncode'] == 0 for j in state['jobs'].values())
        verify(old['hashes'])
        previous_outputs = read(DEST/'timing/output_hashes.json')
        verify(previous_outputs)
        bench = read(DEST/'benchmark.json')
        assert bench['status'] == 'passed'
        # The probe's benchmark schema is checked before this phase is registered.
        frames = int(bench['total_frame_count'])
        measured = float(bench['seconds'])
        completed = int(bench['measured_frames'])
        assert completed == 2 and frames >= completed and measured > 0
        predicted = measured / completed * (frames - completed)
        minutes = max(5, math.ceil((predicted * 1.5 + 120) / 60))
        timing = {'measured_frames': completed, 'measured_seconds': measured,
                  'total_frame_count': frames, 'remaining_frame_count': frames-completed,
                  'predicted_remaining_seconds': predicted, 'variation_multiplier': 1.5,
                  'io_analysis_allowance_seconds': 120}
        paths.update(ROOT/p for p in previous_outputs)
        paths.update([DEST/'timing/plan.json', DEST/'timing/status.json', DEST/'timing/output_hashes.json'])
        commands = ['probe', 'analyse']
        output_paths = ['heatmaps', 'probe', 'analysis']
    save_json(folder/'plan.json', dict(created=stamp(), phase=phase, hashes=hashes(paths),
        commands=commands, total_jobs=len(commands), estimated_minutes=minutes,
        max_start_job_hours=1, timing=timing, output_paths=output_paths,
        no_training=True, no_graph_edits=True, no_kaggle_writes=True,
        interpretation='Fixed stratified signal diagnostic; no independent population or official score claim',
        gpu='local4070TiSUPER', gpu_allocation='Short local diagnostic; remote upload and queue do not shorten its critical path'))
    print(json.dumps(dict(phase=phase, inputs=len(paths), jobs=len(commands), estimated_minutes=minutes, timing=timing)), flush=True)

def run(phase):
    folder = DEST/phase
    assert not (folder/'status.json').exists(), 'No blind duplicate/resume'
    plan = read(folder/'plan.json')
    queue = Queue(folder, plan['max_start_job_hours'])
    try:
        queue.state.update(total_jobs=plan['total_jobs'], plan_sha256=sha(folder/'plan.json'))
        queue.save()
        verify(plan['hashes'])
        for verb in plan['commands']:
            queue.run(verb, [sys.executable, '-u', Path(probe.__file__), verb])
        verify(plan['hashes'])
        assert queue.state['plan_sha256'] == sha(folder/'plan.json')
        paths = []
        for name in plan['output_paths']:
            p = DEST/name
            assert p.exists(), name
            paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()])
        paths.extend(p for p in (folder/'logs').glob('*') if p.is_file())
        save_json(folder/'output_hashes.json', hashes(paths))
        queue.close('complete_review_required')
    except BaseException as exc:
        queue.close('failed', exc)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['prepare', 'run'])
    parser.add_argument('--phase', choices=['timing', 'signal'], required=True)
    args = parser.parse_args()
    globals()[args.verb](args.phase)
