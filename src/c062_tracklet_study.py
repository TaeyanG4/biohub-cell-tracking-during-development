"""Finite C062 no-fit probe orchestration, reusing the established Queue."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import c062_tracklet_probe as probe
from c055_guarded_readmit import sha, save_json, read
from run_last_days_local import Queue, stamp

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/candidates/c062_tracklet_appearance'


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify(record):
    for name, expected in record.items():
        assert sha(ROOT / name) == expected, ('Pinned input changed', name)


def prepare(out, phase):
    folder = out / phase
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'plan.json').exists(), 'No replacement of registered phase'
    paths = set(probe.dependencies())
    paths.update([Path(__file__), ROOT / 'src/run_last_days_local.py', out / 'README.md',
                  ROOT / 'experiments/candidates/c058_raw_localizer/plan.json',
                  ROOT / 'state/c062_preflight/launch.ps1',
                  ROOT / 'tools/notify_background_completion.ps1',
                  ROOT / 'state/c062_preflight/HANDOFF_DUPLICATE_CHECK.md',
                  ROOT / 'state/c062_preflight/science/REVIEW.md',
                  ROOT / 'state/c062_preflight/root_controls/controls.json'])
    commands = []
    if phase == 'preflight':
        commands.append(('controls', ['controls']))
        for stem in probe.STEMS:
            commands += [('capture_' + stem, ['capture', '--stem', stem]),
                         ('benchmark_' + stem, ['benchmark', '--stem', stem])]
        # Historical same C058 passive replay measured276.42s for22 movies.
        # Two full captures plus extra topology/IO and small benchmarks only.
        prior = ROOT / 'experiments/candidates/c058_raw_localizer/status.json'
        paths.add(prior)
        measured = read(prior)['jobs']['capture_all']['seconds'] / 22 * 2
        minutes = max(5, int(np.ceil((measured * 3 + 120) / 60)))
        timing = dict(source=str(prior.relative_to(ROOT)), measured_two_movie_capture_seconds=measured,
                      capture_variation_multiplier=3, added_overhead_seconds=120,
                      minimum_allowance_minutes=5, new_encoding_not_in_this_phase=True)
    else:
        state = read(out / 'preflight/status.json')
        assert state['status'] == 'complete_review_required'
        assert state['plan_sha256'] == sha(out / 'preflight/plan.json')
        verify(read(out / 'preflight/plan.json')['hashes'])
        verify(read(out / 'preflight/output_hashes.json'))
        assert read(out / 'preflight_review.json')['status'] == 'passed'
        paths.update([out / 'preflight/plan.json', out / 'preflight/output_hashes.json', out / 'preflight_review.json'])
        paths.update(p for p in (out / 'probe').rglob('*') if p.is_file())
        seconds = 0.
        timing = []
        for stem in probe.STEMS:
            bench = read(out / 'probe' / stem / 'benchmark.json')
            assert bench['status'] == 'passed'
            seconds += bench['estimated_full_encoding_seconds']
            timing.append(bench)
            commands.append(('encode_' + stem, ['encode', '--stem', stem]))
        commands.append(('analyse', ['analyse']))
        minutes = max(5, int(np.ceil((seconds * 1.5 + 120) / 60)))
    save_json(folder / 'plan.json', dict(created=stamp(), phase=phase, policy=probe.POLICY,
        hashes=hashes(paths), commands=[dict(name=n, arguments=a) for n, a in commands],
        total_jobs=len(commands), estimated_minutes=minutes, timing_evidence=timing,
        max_start_job_hours=2, no_training=True, no_graph_edits=True, no_kaggle_writes=True))
    print(json.dumps(dict(phase=phase, inputs=len(paths), estimated_minutes=minutes, jobs=len(commands))), flush=True)


def run(out, phase):
    folder = out / phase
    assert not (folder / 'status.json').exists(), 'No blind duplicate/resume'
    plan = read(folder / 'plan.json')
    q = Queue(folder, plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'], plan_sha256=sha(folder / 'plan.json'))
        q.save()
        verify(plan['hashes'])
        for job in plan['commands']:
            q.run(job['name'], [sys.executable, '-u', Path(probe.__file__), *job['arguments'],
                               '--out', out / 'probe'])
        verify(plan['hashes'])
        assert q.state['plan_sha256'] == sha(folder / 'plan.json'), 'Phase plan changed during execution'
        outputs = [p for p in (out / 'probe').rglob('*') if p.is_file()]
        outputs += [p for p in (folder / 'logs').glob('*') if p.is_file()]
        save_json(folder / 'output_hashes.json', hashes(outputs))
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['prepare', 'run'])
    parser.add_argument('--phase', choices=['preflight', 'signal'], required=True)
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT)
    globals()[args.verb](out, args.phase)


if __name__ == '__main__':
    main()
