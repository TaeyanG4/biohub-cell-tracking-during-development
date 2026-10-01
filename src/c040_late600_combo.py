#!/usr/bin/env python3
"""Fixed late600+C038 orchestration, reusing Queue, graph audit and official scorer."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import pandas as pd
from c038_followup_local import (ROOT, C037, C038, BASE, CONTROL, OLD, MODES,
    Queue, sha, save_json, stamp, setup_ns, evaluation_plan, verify, edge_audit)

DEST = ROOT / 'experiments/candidates/c040_late600_combo'


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise RuntimeError('Registered plan exists; do not overwrite')
    # Retain existing scientific sources and historical hashes unchanged.
    prior = json.loads((C038 / 'followup_extension/plan.json').read_text())
    inputs = {ROOT / p for p in prior['hashes'] if p.endswith('.py') or p.endswith('.pt')}
    for p in inputs:
        assert sha(p) == prior['hashes'][str(p.relative_to(ROOT))], ('source drift', str(p))
    inputs.update([Path(__file__), BASE, C038 / 'variants.json'])
    inputs.update(OLD / f'control_fp32_{s}.csv' for s in ['heldout12', 'confirm10'])
    jobs = []
    for group in ['44b6', '6bba']:
        name = 'combo_late600_' + group
        run = C037 / 'e2e' / ('late600_' + group)
        stems = [s for _, s in evaluation_plan() if s.startswith(group)]
        jobs.append(dict(name=name, kind='combo', group=group, run=str(run), stems=stems))
        source = C038 / f'local_replay_{group}.ipynb'
        inputs.update([source, C037 / 'replay' / f'late600_{group}.csv'])
        nb = json.loads(source.read_text(encoding='utf-8'))
        code = ''.join(nb['cells'][5]['source'])
        anchor = '\nwrite_test_submission("base")\n'
        assert code.count(anchor) == 1
        addition = ('\nfrom c038_followup_local import install_graph_audit\n'
                    f'install_graph_audit(globals(), {str(out / "graphs" / name)!r})\n')
        nb['cells'][5]['source'] = code.replace(anchor, addition + anchor).splitlines(keepends=True)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{name}:cell{i}', 'exec')
        target = out / (name + '.ipynb')
        target.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        inputs.add(target)
        for stem in stems:
            graph = next((run / 'predictions').rglob(stem + '.geff'))
            inputs.update(p for p in graph.rglob('*') if p.is_file())
            caches = list((run / 'edge_cache').glob(stem + '*'))
            assert caches, ('missing edge cache', stem)
            inputs.update(caches)
    save_json(out / 'plan.json', dict(created=stamp(), jobs=jobs,
        hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)},
        selection='Retrospective late600 choice: smaller confirm10 loss than early150, before combo results.',
        policy='Fixed off/appearance/agreement; unchanged C038 knobs/models. No retraining or threshold sweep.',
        deployment=False, next='Review signed gains by split/embryo vs C023 and own off; no automatic submission.'))
    print('Prepared two replay jobs and analysis; inputs:', len(inputs), flush=True)


def analyse(out):
    plan = json.loads((out / 'plan.json').read_text())
    data = pd.concat([pd.read_csv(out / 'replay' / (j['name'] + '.csv')) for j in plan['jobs']])
    original = pd.concat([pd.read_csv(C037 / 'replay' / f'late600_{g}.csv') for g in ['44b6', '6bba']])
    assert original.config.nunique() == 1 and len(original) == 22
    verify(data, original, {'off': original.config.iloc[0]})
    baseline = pd.concat([pd.read_csv(OLD / f'control_fp32_{s}.csv') for s in ['heldout12', 'confirm10']])
    with contextlib.redirect_stdout(io.StringIO()):
        ns, _, _ = setup_ns(CONTROL / 'control_fp32_heldout12')
    rows = []
    for mode in MODES[1:]:
        for group in ['all22', 'heldout12', 'confirm10', '44b6', '6bba']:
            stems = {s for split, s in evaluation_plan()
                     if group == 'all22' or group == split or s.startswith(group)}
            current = data[(data.config == mode) & data.stem.isin(stems)]
            off = data[(data.config == 'off') & data.stem.isin(stems)]
            ref = baseline[baseline.stem.isin(stems)]
            assert len(current) == len(off) == len(ref) == len(stems)
            a, b, c = [ns['aggregate_official'](x.to_dict('records')) for x in [current, ref, off]]
            delta = current.set_index('stem').adjusted_edge_jaccard - ref.set_index('stem').adjusted_edge_jaccard
            rows.append(dict(mode=mode, group=group, movies=len(stems), score=a['proxy_score'],
                delta=a['proxy_score']-b['proxy_score'],
                edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
                delta_vs_own_off=a['proxy_score']-c['proxy_score'],
                edge_delta_vs_own_off=a['adjusted_edge_jaccard']-c['adjusted_edge_jaccard'],
                wins=int((delta>1e-10).sum()), losses=int((delta< -1e-10).sum()),
                div_tp=a['div_tp'], div_fp=a['div_fp'], div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out / 'official_summary.csv', index=False)
    edge_audit(out, plan)
    save_json(out / 'analysis.json', dict(status='complete_review_required', rows=rows,
        off_controls=22, note='Official local formula; diagnostic cross-embryo models. Not hidden LB evidence.'))


def run(out):
    for folder in [C037, C038, C038/'followup_extension', ROOT/'experiments/candidates/c039_public_v6_salvage']:
        assert json.loads((folder/'status.json').read_text())['status'] != 'running', ('queue busy', str(folder))
    plan = json.loads((out/'plan.json').read_text())
    assert all(sha(ROOT/p) == v for p,v in plan['hashes'].items()), 'input drift'
    q = Queue(out, 4)
    try:
        q.state['plan_sha256'] = sha(out/'plan.json'); q.save()
        for j in plan['jobs']:
            run_dir = Path(j['run'])
            q.run(j['name'], [sys.executable, '-u', ROOT/'src/eval_pp_variants_local.py',
                '--notebook', out/(j['name']+'.ipynb'), '--variants', C038/'variants.json',
                '--stems', ','.join(j['stems']), '--pred-root', run_dir/'predictions',
                '--lowdet-dir', run_dir/'edge_cache', '--round-coords',
                '--out', out/'replay'/(j['name']+'.csv')])
        q.run('analyse', [sys.executable, '-u', Path(__file__), 'analyse', '--out', out])
        assert all(sha(ROOT/p) == v for p,v in plan['hashes'].items()), 'input drift during replay'
        save_json(out/'artifact_hashes.json', {str(p.relative_to(out)):sha(p)
            for folder in ['graphs','replay'] for p in (out/folder).rglob('*') if p.is_file()})
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','analyse'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args(); out=args.out.resolve(); out.relative_to(ROOT)
    globals()[args.command](out)
