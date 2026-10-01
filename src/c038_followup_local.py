#!/usr/bin/env python3
"""Finite C038 extension/combination orchestration; reuse the official replay tool.

Graph capture is read-only and GT-free. No new scorer or graph-edit algorithm.
Original C037/C038 sources, notebooks, models and pilot results stay immutable.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reid_probe_local import BASE, CONTROL, OLD, sha, save_json, stamp, setup_ns
from reid_augmented_local import evaluation_plan
from run_last_days_local import Queue, extension_stems

C037 = ROOT / 'experiments/candidates/c037_transformer_finetune'
C038 = ROOT / 'experiments/candidates/c038_complementary_fusion'
DEST = C038 / 'followup_extension'
MODES = ['off', 'appearance', 'agreement']
METRICS = ['nodes', 'edges', 'edge_tp', 'edge_fp', 'edge_fn', 'div_tp', 'div_fp',
           'div_fn', 'adjusted_edge_jaccard']


def install_graph_audit(ns, folder):
    """Save returned graphs without consulting GT or changing any returned value."""
    original = ns['filter_output_graph']
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def wrapper(*args, **kwargs):
        nodes, edges, stats = original(*args, **kwargs)
        stem = kwargs['dataset']
        mode = ns['C038_MODE']
        plain = ns['nodes_by_id_to_plain'](nodes)
        ids = np.array(sorted(plain), np.int64)
        txyz = np.array([(plain[i][0], *(max(0, int(round(v))) for v in plain[i][1:]))
                         for i in ids], np.int64)
        pairs = np.array(sorted((int(e['source_id']), int(e['target_id'])) for e in edges), np.int64)
        np.savez_compressed(folder / f'{stem}_{mode}.npz', ids=ids, txyz=txyz, edges=pairs)
        save_json(folder / f'{stem}_{mode}.json', dict(stem=stem, mode=mode, nodes=len(nodes),
                  edges=len(edges), proposed=int(stats.get('c038_proposals', 0)),
                  changed=int(stats.get('c038_changed_edges', 0))))
        return nodes, edges, stats
    ns['filter_output_graph'] = wrapper


def jobs():
    pilot = evaluation_plan()
    extension, _ = extension_stems({s for _, s in pilot})
    result = []
    # Replay original pilot with passive graph capture first, requiring all66 identical rows.
    for split in ['heldout12', 'confirm10']:
        for group in ['44b6', '6bba']:
            stems = [s for sp, s in pilot if sp == split and s.startswith(group)]
            if stems:
                result.append(dict(name=f'pilot_{split}_{group}', kind='pilot', group=group,
                                   stems=stems, run=str(CONTROL / f'control_fp32_{split}')))
    # Fixed, retrospectively registered combination: best pilot edge arm, no weight sweep.
    for group in ['44b6', '6bba']:
        result.append(dict(name=f'combo_early150_{group}', kind='combo', group=group,
                           stems=[s for _, s in pilot if s.startswith(group)],
                           run=str(C037 / 'e2e' / f'early150_{group}')))
    for group in ['44b6', '6bba']:
        result.append(dict(name=f'extension75_{group}', kind='extension', group=group,
                           stems=[s for s in extension if s.startswith(group)],
                           run=str(CONTROL / 'control_fp32_extension75')))
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise RuntimeError('Existing follow-up plan; do not rebuild a registered/running batch')
    for folder in [C037, C038]:
        state = json.loads((folder / 'status.json').read_text())
        assert state['status'] == 'complete_review_required'
        for p, expected in state['source_hashes'].items():
            assert sha(ROOT / p) == expected, ('pilot source drift', p)
    plan_jobs = jobs()
    inputs = [Path(__file__), ROOT / 'src/run_last_days_local.py',
              ROOT / 'src/c038_complementary_stage.py', ROOT / 'src/reid_probe_local.py',
              ROOT / 'src/reid_augmented_local.py', ROOT / 'src/local_registration_probe.py',
              ROOT / 'src/frame_motion_audit.py', ROOT / 'src/division_crops_extract.py',
              ROOT / 'src/division_cnn_train.py', ROOT / 'src/eval_pp_variants_local.py', BASE,
              C038 / 'variants.json']
    for group in ['44b6', '6bba']:
        inputs += [C038 / f'local_replay_{group}.ipynb',
                   ROOT / f'experiments/candidates/c035_augmented_reid/models/{group}_weak_aug.pt',
                   C037 / 'replay' / f'early150_{group}.csv']
    inputs += [p for p in (C038 / 'replay').glob('*.csv') if not p.stem.endswith('_summary')]
    inputs += [OLD / f'control_fp32_{sp}.csv' for sp in ['heldout12', 'confirm10', 'extension75']]
    for job in plan_jobs:
        run = Path(job['run'])
        # Pin actual replay inputs, including all graph metadata/chunks and edge/lowdet caches.
        for stem in job['stems']:
            graph = next((run / 'predictions').rglob(stem + '.geff'))
            inputs += list(graph.rglob('*')) if graph.is_dir() else [graph]
            inputs += list((run / 'edge_cache').glob(stem + '*'))
        nb = json.loads((C038 / f'local_replay_{job["group"]}.ipynb').read_text())
        source = ''.join(nb['cells'][5]['source'])
        anchor = '\nwrite_test_submission("base")\n'
        assert source.count(anchor) == 1
        addition = ('\nfrom c038_followup_local import install_graph_audit\n'
                    f'install_graph_audit(globals(), {str(out / "graphs" / job["name"])!r})\n')
        nb['cells'][5]['source'] = source.replace(anchor, addition + anchor).splitlines(keepends=True)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{job["name"]}:cell{i}', 'exec')
        target = out / (job['name'] + '.ipynb')
        target.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        inputs.append(target)
    files = sorted(set(p.resolve() for p in inputs if p.is_file()))
    save_json(out / 'plan.json', dict(created=stamp(), jobs=plan_jobs,
              hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
              combination='early150 + each unchanged C038 mode, chosen after pilot review; no threshold sweep',
              deployment=False, note='All notebooks local-only; do not push.'))
    print('prepared', len(plan_jobs), 'jobs;', len(files), 'pinned inputs', flush=True)


def tables(out, plan, kind):
    return pd.concat([pd.read_csv(out / 'replay' / (j['name'] + '.csv'))
                      for j in plan['jobs'] if j['kind'] == kind], ignore_index=True)


def verify(frame, reference, configs):
    for mode, ref in configs.items():
        actual = frame[frame.config == mode].set_index('stem').sort_index()
        ref = reference[reference.config == ref].set_index('stem').sort_index()
        assert actual.index.is_unique and list(actual.index) == list(ref.index)
        for key in METRICS:
            assert np.allclose(actual[key], ref[key], rtol=0, atol=1e-10), ('parity', mode, key)


def check_pilot(out, plan):
    data = tables(out, plan, 'pilot')
    reference = pd.concat([pd.read_csv(p) for p in (C038 / 'replay').glob('*.csv')
                           if not p.stem.endswith('_summary')])
    verify(data, reference, {m: m for m in MODES})
    save_json(out / 'audit_parity.json', dict(status='passed', movies=22, mode_rows=66))


def graph(folder, stem, mode):
    with np.load(folder / f'{stem}_{mode}.npz') as data:
        nodes = {int(i): tuple(map(int, xyz)) for i, xyz in zip(data['ids'], data['txyz'])}
        edges = {tuple(map(int, e)) for e in data['edges']}
    return nodes, edges


def edge_audit(out, plan):
    rows = []
    for job in plan['jobs']:
        folder = out / 'graphs' / job['name']
        for stem in job['stems']:
            base_nodes, base_edges = graph(folder, stem, 'off')
            for mode in MODES[1:]:
                nodes, edges = graph(folder, stem, mode)
                assert base_nodes == nodes and len(base_edges) == len(edges)
                assert collections.Counter(a for a, _ in base_edges) == collections.Counter(a for a, _ in edges)
                removed, added = base_edges - edges, edges - base_edges
                info = json.loads((folder / f'{stem}_{mode}.json').read_text())
                assert len(removed) == len(added) == info['changed']
                row = dict(job=job['name'], kind=job['kind'], stem=stem, mode=mode,
                           proposed=info['proposed'], changed=len(added),
                           removed=sorted(removed), added=sorted(added))
                if job['kind'] == 'pilot':
                    combo_folder = out / 'graphs' / f'combo_early150_{job["group"]}'
                    cn, ce = graph(combo_folder, stem, 'off')
                    key_counts = collections.Counter(cn.values())
                    cout = collections.defaultdict(set)
                    for a, b in ce:
                        if key_counts[cn[a]] == key_counts[cn[b]] == 1:
                            cout[cn[a]].add(cn[b])
                    base_counts = collections.Counter(base_nodes.values())
                    old = {a: b for a, b in removed}
                    overlap = collections.Counter()
                    for a, b in added:
                        if any(base_counts[base_nodes[i]] != 1 for i in [a, b, old[a]]):
                            overlap['ambiguous_coordinate'] += 1
                        elif base_nodes[a] not in cout:
                            overlap['unmatched_coordinate'] += 1
                        elif cout[base_nodes[a]] == {base_nodes[b]}:
                            overlap['already_in_c037'] += 1
                        elif cout[base_nodes[a]] == {base_nodes[old[a]]}:
                            overlap['c037_retains_original'] += 1
                        else:
                            overlap['different_c037_edge'] += 1
                    row['c037_coordinate_overlap'] = dict(overlap)
                    if stem == '44b6_12dfb391':
                        row['old_diagnostic_source_25937'] = dict(
                            before=[b for a, b in base_edges if a == 25937],
                            after=[b for a, b in edges if a == 25937])
                rows.append(row)
    save_json(out / 'edge_audit.json', dict(rows=rows,
              note='GT-free saved-graph changes; coordinate overlap is descriptive, not biological correctness. Unannotated links remain unknown. Official score is computed only by the existing harness.'))


def analyse(out):
    plan = json.loads((out / 'plan.json').read_text())
    check_pilot(out, plan)
    pilot, combo, extension = [tables(out, plan, k) for k in ['pilot', 'combo', 'extension']]
    old = pd.concat([pd.read_csv(OLD / f'control_fp32_{s}.csv') for s in ['heldout12', 'confirm10', 'extension75']])
    verify(extension, old[old.stem.isin(extension.stem)], {'off': 'as_configured'})
    early = pd.concat([pd.read_csv(C037 / 'replay' / f'early150_{g}.csv') for g in ['44b6', '6bba']])
    # C037 variants use the exact single registered label, whatever its spelling.
    assert early.config.nunique() == 1
    verify(combo, early, {'off': early.config.iloc[0]})
    with contextlib.redirect_stdout(io.StringIO()):
        ns, _, _ = setup_ns(CONTROL / 'control_fp32_heldout12')
    rows = []
    for kind, data in [('extension75', extension), ('aggregate97', pd.concat([pilot, extension])), ('combo22', combo)]:
        for mode in MODES[1:]:
            for group in ['all', '44b6', '6bba'] + (['heldout12', 'confirm10'] if kind == 'combo22' else []):
                stems = set(data.stem)
                if group in ['44b6', '6bba']:
                    stems = {s for s in stems if s.startswith(group)}
                elif group in ['heldout12', 'confirm10']:
                    stems &= {s for sp, s in evaluation_plan() if sp == group}
                current = data[(data.config == mode) & data.stem.isin(stems)]
                ref = old[old.stem.isin(stems)]
                a, b = [ns['aggregate_official'](d.to_dict('records')) for d in [current, ref]]
                off = ns['aggregate_official'](data[(data.config == 'off') & data.stem.isin(stems)].to_dict('records'))
                delta = current.set_index('stem').adjusted_edge_jaccard - ref.set_index('stem').adjusted_edge_jaccard
                rows.append(dict(kind=kind, mode=mode, group=group, movies=len(stems), score=a['proxy_score'],
                    delta=a['proxy_score']-b['proxy_score'], edge_delta=a['adjusted_edge_jaccard']-b['adjusted_edge_jaccard'],
                    delta_vs_own_off=a['proxy_score']-off['proxy_score'], wins=int((delta>1e-10).sum()),
                    losses=int((delta < -1e-10).sum()), div_tp=a['div_tp'], div_fp=a['div_fp'], div_fn=a['div_fn']))
    pd.DataFrame(rows).to_csv(out / 'official_summary.csv', index=False)
    edge_audit(out, plan)
    save_json(out / 'analysis.json', dict(status='complete_review_required', rows=rows,
              parity=dict(original_pilot_rows=66, extension_off_movies=75, c037_off_movies=22),
              instruction='Review signed gains and harms. No automatic submission or hidden embryo router. A fixed deployment model requires fresh actual replay.'))


def run(out):
    plan = json.loads((out / 'plan.json').read_text())
    for folder in [C037, C038]:
        assert json.loads((folder / 'status.json').read_text())['status'] != 'running'
    assert all(sha(ROOT / p) == v for p, v in plan['hashes'].items()), 'follow-up input drift'
    q = Queue(out, 12)
    try:
        q.state['plan_sha256'] = sha(out / 'plan.json'); q.save()
        pilot_checked = False
        for job in plan['jobs']:
            if job['kind'] != 'pilot' and not pilot_checked:
                check_pilot(out, plan); pilot_checked = True
            run_dir = Path(job['run'])
            q.run(job['name'], [sys.executable, '-u', ROOT / 'src/eval_pp_variants_local.py',
                '--notebook', out / (job['name'] + '.ipynb'), '--variants', C038 / 'variants.json',
                '--stems', ','.join(job['stems']), '--pred-root', run_dir / 'predictions',
                '--lowdet-dir', run_dir / 'edge_cache', '--round-coords',
                '--out', out / 'replay' / (job['name'] + '.csv')])
        q.run('analyse', [sys.executable, '-u', Path(__file__), 'analyse', '--out', out])
        assert all(sha(ROOT / p) == v for p, v in plan['hashes'].items()), 'input drift during follow-up'
        save_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha(p)
                  for d in ['graphs', 'replay'] for p in (out / d).rglob('*') if p.is_file()})
        q.close('complete_review_required')
    except BaseException as exc:
        q.close('failed', exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'analyse'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args(); out = args.out.resolve(); out.relative_to(ROOT)
    globals()[args.command](out)


if __name__ == '__main__':
    main()
