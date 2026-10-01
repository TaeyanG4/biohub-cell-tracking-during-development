#!/usr/bin/env python3
"""Finite fixed-appearance extension using the existing replay, audit and Queue."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from c041_fixed_models import make_notebook
from c038_followup_local import graph, verify
from reid_probe_local import ROOT, BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue, extension_stems, replay_command

C041 = ROOT / 'experiments/candidates/c041_fixed_models'
C038 = ROOT / 'experiments/candidates/c038_complementary_fusion/followup_extension'
DEST = ROOT / 'experiments/candidates/c046_fixed_appearance_extension'


def verify_inputs(plan):
    for relative, digest in plan['hashes'].items():
        assert sha(ROOT / relative) == digest, ('input drift', relative)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'plan.json').exists(), 'Registered plan already exists'
    inputs = {Path(__file__), BASE, C041 / 'portable_appearance.py',
              C041 / 'appearance_mean_cosine.pt', out / 'README.md'}
    for folder in [C041, C038]:
        assert json.loads((folder / 'status.json').read_text())['status'] == 'complete_review_required'
        prior = json.loads((folder / 'plan.json').read_text())
        verify_inputs(prior)
        inputs.update(ROOT / p for p in prior['hashes'])
        inputs.add(folder / 'plan.json')
    pilot = evaluation_plan()
    extension, list_hashes = extension_stems({s for _, s in pilot})
    inputs.update(ROOT / p for p in list_hashes)
    assert len(pilot) == 22 and len(extension) == 75
    source = (C041 / 'portable_appearance.py').read_text(encoding='utf-8')
    checkpoint = C041 / 'appearance_mean_cosine.pt'
    variants = out / 'variants.json'
    save_json(variants, {'off': {'C038_MODE': 'off'}, 'appearance': {'C038_MODE': 'appearance'}})
    inputs.add(variants)
    jobs = []
    for split in ['heldout12', 'confirm10', 'extension75']:
        stems = extension if split == 'extension75' else [s for sp, s in pilot if sp == split]
        kind = 'extension' if split == 'extension75' else 'pilot_reproduction'
        run = CONTROL / ('control_fp32_' + split)
        inputs.add(make_notebook(out, split, checkpoint, source))
        jobs.append(dict(name=split, kind=kind, split=split, stems=stems, run=str(run)))
        inputs.add(OLD / ('control_fp32_' + split + '.csv'))
        if split != 'extension75':
            inputs.add(C041 / 'replay' / ('appearance_' + split + '.csv'))
        for stem in stems:
            assert (ROOT / 'data/train' / (stem + '.zarr')).exists()
            for folder in [ROOT / 'data/train' / (stem + '.geff'),
                           next((run / 'predictions').rglob(stem + '.geff'))]:
                inputs.update(p for p in folder.rglob('*') if p.is_file())
            inputs.update((run / 'edge_cache').glob(stem + '*'))
            if split != 'extension75':
                reference = C041 / 'graphs' / ('appearance_' + split)
                for mode in ['off', 'appearance']:
                    inputs.update(reference.glob(stem + '_' + mode + '.*'))
            else:
                reference = C038 / 'graphs' / ('extension75_' + stem.split('_')[0])
                inputs.update(reference.glob(stem + '_off.*'))
    assert all(p.is_file() for p in inputs), 'Missing pinned input'
    save_json(out / 'plan.json', dict(created=stamp(), jobs=jobs,
              hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)},
              fixed_model_sha256=sha(checkpoint), fixed_runtime_sha256=sha(C041 / 'portable_appearance.py'),
              policy='One fixed global mean-cosine appearance model on original C023. No fitting, inference, sweeps or prefix routing.',
              total_jobs=5, max_start_job_hours=6, deployment=False))
    print('Prepared5 finite jobs; pinned', len(inputs), 'inputs; original C041 model/runtime unchanged', flush=True)


def check_pilot(out):
    plan = json.loads((out / 'plan.json').read_text())
    rows = 0
    for job in [j for j in plan['jobs'] if j['kind'] == 'pilot_reproduction']:
        actual = pd.read_csv(out / 'replay' / (job['name'] + '.csv'))
        reference = pd.read_csv(C041 / 'replay' / ('appearance_' + job['name'] + '.csv'))
        verify(actual, reference, {'off': 'off', 'appearance': 'appearance'})
        for stem in job['stems']:
            for mode in ['off', 'appearance']:
                assert graph(out / 'graphs' / job['name'], stem, mode) == graph(
                    C041 / 'graphs' / ('appearance_' + job['name']), stem, mode), ('C041 graph drift', stem, mode)
                rows += 1
    assert rows == 44
    save_json(out / 'pilot_parity.json', dict(status='passed', exact_metric_rows=44, exact_full_graphs=44))


def analyse(out):
    plan = json.loads((out / 'plan.json').read_text())
    check_pilot(out)
    frames = {j['name']: pd.read_csv(out / 'replay' / (j['name'] + '.csv')) for j in plan['jobs']}
    data = pd.concat(list(frames.values()), ignore_index=True)
    baseline = pd.concat([pd.read_csv(OLD / ('control_fp32_' + sp + '.csv')) for sp in frames])
    assert data.stem.nunique() == baseline.stem.nunique() == 97
    verify(data, baseline, {'off': 'as_configured'})
    for stem in frames['extension75'].stem.unique():
        assert graph(out / 'graphs/extension75', stem, 'off') == graph(
            C038 / 'graphs' / ('extension75_' + stem.split('_')[0]), stem, 'off'), ('75 off graph drift', stem)
    with contextlib.redirect_stdout(io.StringIO()):
        ns, _, _ = setup_ns(CONTROL / 'control_fp32_heldout12')
    rows = []
    for split in ['heldout12', 'confirm10', 'pilot22', 'extension75', 'aggregate97']:
        stems = set(data.stem) if split == 'aggregate97' else (
            set(frames['heldout12'].stem) | set(frames['confirm10'].stem) if split == 'pilot22' else set(frames[split].stem))
        for group in ['all', '44b6', '6bba']:
            selected = stems if group == 'all' else {s for s in stems if s.startswith(group)}
            current = data[(data.config == 'appearance') & data.stem.isin(selected)]
            control = baseline[baseline.stem.isin(selected)]
            assert len(current) == len(control) == len(selected) > 0
            a, b = [ns['aggregate_official'](x.to_dict('records')) for x in [current, control]]
            delta = current.set_index('stem').adjusted_edge_jaccard - control.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(split=split, group=group, movies=len(selected), score=a['proxy_score'],
                control_score=b['proxy_score'], delta=a['proxy_score'] - b['proxy_score'],
                edge_delta=a['adjusted_edge_jaccard'] - b['adjusted_edge_jaccard'],
                wins=int((delta > 1e-10).sum()), losses=int((delta < -1e-10).sum()),
                div_tp=a['div_tp'], div_fp=a['div_fp'], div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out / 'official_summary.csv', index=False)
    signed = data[data.config == 'appearance'].set_index('stem').copy()
    reference = baseline.set_index('stem')
    for field in ['adjusted_edge_jaccard', 'edge_tp', 'edge_fp', 'edge_fn', 'div_tp', 'div_fp', 'div_fn']:
        signed[field + '_delta'] = signed[field] - reference[field]
    signed.to_csv(out / 'signed_movies.csv')
    import c038_followup_local as audit
    previous = audit.MODES
    try:
        audit.MODES = ['off', 'appearance']
        audit.edge_audit(out, plan)
    finally:
        audit.MODES = previous
    save_json(out / 'analysis.json', dict(status='complete_review_required', rows=rows,
        parity=dict(C041_pilot_metric_and_graph_rows=44, C023_off_movies=97, exact_extension_off_graphs=75),
        note='Review signed total/edge/subgroup effects before portable construction; no automatic score or submission gate. No claim of independent test generalization.'))


def run(out):
    plan = json.loads((out / 'plan.json').read_text())
    verify_inputs(plan)
    q = Queue(out, 6)
    try:
        q.state['plan_sha256'] = sha(out / 'plan.json')
        q.state['total_jobs'] = 5
        q.save()
        for job in plan['jobs']:
            if job['kind'] == 'extension':
                q.run('check_pilot', [sys.executable, '-u', Path(__file__), 'check_pilot', '--out', out])
            q.run(job['name'], replay_command(out / (job['name'] + '.ipynb'), Path(job['run']),
                  job['stems'], out / 'variants.json', out / 'replay' / (job['name'] + '.csv')))
        q.run('analyse', [sys.executable, '-u', Path(__file__), 'analyse', '--out', out])
        verify_inputs(plan)
        files = [p for folder in ['graphs', 'replay'] for p in (out / folder).rglob('*') if p.is_file()]
        files += [out / p for p in ['pilot_parity.json', 'official_summary.csv', 'signed_movies.csv', 'edge_audit.json', 'analysis.json']]
        save_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha(p) for p in files})
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'check_pilot', 'analyse'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args()
    folder = args.out.resolve()
    folder.relative_to(ROOT)
    globals()[args.command](folder)
