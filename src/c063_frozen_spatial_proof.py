"""Finite exact-primary-prefix parity proof, using the existing Queue."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import c063_frozen_spatial_extract as extract
from c055_guarded_readmit import read, sha, save_json
from c063_frozen_spatial_study import verify, hashes, DEST, STEMS
from run_last_days_local import Queue, stamp
ROOT=Path(__file__).resolve().parents[1]


def prepare(out):
    folder=out/'prefix_check';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'plan.json').exists(), 'No registered replacement'
    preflight=out/'preflight'
    assert read(preflight/'status.json')['status']=='complete_review_required'
    assert read(preflight/'status.json')['plan_sha256']==sha(preflight/'plan.json')
    verify(read(preflight/'plan.json')['hashes']);verify(read(preflight/'output_hashes.json'))
    review=read(out/'preflight_review.json');assert review['status']=='passed'
    paths=set(extract.dependencies(STEMS,reference_dir=out/'capture'))
    paths.update([Path(__file__),ROOT/'src/c063_frozen_spatial_study.py',
        ROOT/'src/run_last_days_local.py',preflight/'plan.json',preflight/'output_hashes.json',
        out/'preflight_review.json',ROOT/'state/c063_preflight/review_completed.py',
        ROOT/'state/c063_preflight/launch_prefix.ps1',ROOT/'tools/notify_background_completion.ps1',
        ROOT/'state/c063_preflight/geometry/PREFIX_README.md',
        ROOT/'state/c063_preflight/geometry/prefix_controls/controls.json'])
    commands=[dict(name='prefix_controls',arguments=['controls','--output-dir',str(out/'prefix_controls')])]
    commands += [dict(name='prove_'+stem,arguments=['prove','--movie',stem,
        '--output-dir',str(out/'prefix'),'--reference-dir',str(out/'capture')]) for stem in STEMS]
    baseline_seconds=sum(r['phase_seconds']['passive'] for r in review['rows'])
    minutes=max(5,math.ceil((baseline_seconds*1.25+120)/60))
    save_json(folder/'plan.json',dict(created=stamp(),phase='prefix_check',hashes=hashes(paths),
        commands=commands,total_jobs=len(commands),estimated_minutes=minutes,max_start_job_hours=2,
        timing=dict(actual_full_passive_seconds=baseline_seconds,multiplier=1.25,overhead_seconds=120,
            note='Conservative full-pipeline timing allowance; prefix-only time not yet measured'),
        no_training=True,no_graph_edits=True,no_kaggle_writes=True))
    print(json.dumps(dict(inputs=len(paths),jobs=len(commands),estimated_minutes=minutes)),flush=True)


def run(out):
    folder=out/'prefix_check'
    assert not (folder/'status.json').exists(),'No blind duplicate/resume'
    plan=read(folder/'plan.json');q=Queue(folder,plan['max_start_job_hours'])
    try:
        q.state.update(total_jobs=plan['total_jobs'],plan_sha256=sha(folder/'plan.json'));q.save()
        verify(plan['hashes'])
        for job in plan['commands']:
            q.run(job['name'],[sys.executable,'-u',Path(extract.__file__),*job['arguments']])
        verify(plan['hashes'])
        assert q.state['plan_sha256']==sha(folder/'plan.json')
        for stem in STEMS:
            proof=read(out/'prefix'/stem/'proof.json')
            assert proof['status']=='passed' and proof['all_requested_fields_exact']
            assert proof['entire_fp32_cubes_exact'] and proof['labels_targets_exclusions_exact']
        outputs=[p for part in ['prefix','prefix_controls'] for p in (out/part).rglob('*') if p.is_file()]
        outputs += [p for p in (folder/'logs').glob('*') if p.is_file()]
        save_json(folder/'output_hashes.json',hashes(outputs))
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed',exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb',choices=['prepare','run'])
    parser.add_argument('--out',type=Path,default=DEST)
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT)
    globals()[args.verb](out)
