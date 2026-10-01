"""C063 finite passive production-feature proof; existing Queue, no fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from c055_guarded_readmit import read, sha, save_json
from run_last_days_local import Queue, stamp
import c063_frozen_spatial_capture as capture

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/candidates/c063_frozen_spatial'
STEMS = ['44b6_12dfb391', '6bba_05db0fb1']


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify(record):
    for name, digest in record.items():
        assert sha(ROOT / name) == digest, ('Pinned input changed', name)


def prepare(out):
    folder = out / 'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'plan.json').exists(), 'No replacement of registered phase'
    proof = read(ROOT / 'state/c063_preflight/preparation_controls.json')
    assert proof['status'] == 'passed' and proof['capture_sha256'] == sha(Path(capture.__file__))
    paths = set(capture.dependencies(STEMS))
    paths.update([Path(__file__), ROOT / 'src/run_last_days_local.py', out / 'README.md',
                  ROOT / 'tools/notify_background_completion.ps1',
                  ROOT / 'state/c063_preflight/launch.ps1',
                  ROOT / 'state/c063_preflight/HANDOFF_DUPLICATE_CHECK.md',
                  ROOT / 'state/c063_preflight/geometry/CONTRACT.md',
                  ROOT / 'state/c063_preflight/preparation_controls.py',
                  ROOT / 'state/c063_preflight/preparation_controls.json',
                  ROOT / 'state/c063_preflight/root_controls/controls.json'])
    prior = ROOT / 'state/last_days_local_20260926/status.json'
    historical = read(prior)
    timed = [j for k, j in historical['jobs'].items()
             if 'control_fp32' in k and 'seconds' in j and j.get('returncode') == 0]
    # Source history supplies an allowance, not a claim of measuring the new hook.
    seconds = sum(j['seconds'] for j in timed)
    assert seconds > 0 and timed, 'Need actual historical full-pipeline timing'
    paths.add(prior)
    minutes = 15
    commands = [dict(name='controls', arguments=['controls'])]
    commands += [dict(name='capture_' + stem, arguments=['capture', '--movie', stem]) for stem in STEMS]
    save_json(folder / 'plan.json', dict(created=stamp(), phase='preflight', stems=STEMS,
        hashes=hashes(paths), commands=commands, total_jobs=len(commands),
        estimated_minutes=minutes, max_start_job_hours=2,
        timing_evidence=dict(historical_jobs=[dict(seconds=j['seconds'], command=j['command']) for j in timed],
            allowance='15min for two original/passive full movies plus cube IO and verification; new timing unmeasured'),
        no_training=True, no_graph_edits=True, no_kaggle_writes=True,
        gpu_allocation='Local4070TiSUPER critical-path proof; no remote overhead justified for two movies'))
    print(json.dumps(dict(inputs=len(paths), jobs=len(commands), estimated_minutes=minutes)), flush=True)


def run(out):
    folder = out / 'preflight'
    assert not (folder / 'status.json').exists(), 'No blind duplicate/resume'
    plan = read(folder / 'plan.json')
    q = Queue(folder, plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'], plan_sha256=sha(folder / 'plan.json'))
        q.save()
        verify(plan['hashes'])
        for job in plan['commands']:
            q.run(job['name'], [sys.executable, '-u', Path(capture.__file__), *job['arguments'],
                               '--output-dir', out / 'capture'])
        verify(plan['hashes'])
        assert q.state['plan_sha256'] == sha(folder / 'plan.json')
        outputs = [p for p in (out / 'capture').rglob('*') if p.is_file()]
        outputs += [p for p in (folder / 'logs').rglob('*') if p.is_file()]
        save_json(folder / 'output_hashes.json', hashes(outputs))
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT)
    globals()[args.verb](out)


if __name__ == '__main__':
    main()
