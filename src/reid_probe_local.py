#!/usr/bin/env python3
"""C034 appearance diagnostic on the real C023 motion-relink candidate sets.

Reuses public arnav170 descriptors and the existing official replay namespace.
Records candidates without changing costs, assignments, nodes or final outputs.
Only GT-matched negative targets are used for fitting; unmatched targets remain
unknown. Cross-embryo models compare geometry with geometry + appearance.
This diagnostic is not a new submission scorer or a production linker.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from eval_pp_variants_local import build_namespace, code_cells, install_frame_caches, load_raw_graph

BASE = ROOT / 'experiments/candidates/c023_x138_head_stabilize_restore/biohub-c023-x138-head-stabilize-restore.ipynb'
PUBLIC = ROOT / 'state/notebook_radar/pulled/arnav170__biohub-reid3s/biohub-reid3s.ipynb'
CONTROL = ROOT / 'experiments/candidates/c032_temporal_context/e2e'
OLD = ROOT / 'state/last_days_local_20260926/replay'
DEST = ROOT / 'experiments/candidates/c034_appearance_reid'
GEOMETRY = ['raw_um', 'motion_um', 'transformer_prob', 'c023_cost', 'cost_rank', 'cost_gap', 'n_gated']
APPEARANCE_INDEX = [0, 1, 2, 5, 6, 7, 11, 12]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def public_namespace(ns):
    """Extract only descriptor definitions/constants, never notebook execution."""
    wanted = {'_reid_cubes', 'reid_descriptors_for_frame', 'reid_standardise', 'reid_pair_features'}
    pieces = []
    for source in code_cells(PUBLIC):
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                pieces.append(ast.get_source_segment(source, node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                segment = ast.get_source_segment(source, node)
                if segment.startswith(('REID_RZ,', 'REID_BLOCKS ', 'REID_DIM ', 'REID_DESC:',
                                       'REID_SCALER:', 'REID_PAIR_NAMES ', '_reid_')):
                    pieces.append(segment)
    source = '\n\n'.join(pieces)
    env = dict(np=np, cKDTree=cKDTree, VOXEL_SCALE_UM=ns['VOXEL_SCALE_UM'],
               read_test_frame=ns['read_test_frame'], deepcenter_score_point=ns['deepcenter_score_point'])
    exec(compile(source, str(PUBLIC) + ':descriptor_subset', 'exec'), env)
    assert wanted.issubset(env) and env['REID_DIM'] == 28
    return env, source


class Recorder:
    def __init__(self, ns, gt_nodes, gt_edges):
        self.ns, self.gt_nodes = ns, gt_nodes
        self.gt_out = {}
        self.gt_in = {}
        for a, b in gt_edges:
            self.gt_out.setdefault(a, []).append(b)
            self.gt_in.setdefault(b, []).append(a)
        self.groups = {}
        self.nodes = {}
        self.p2g, self.g2p = {}, {}
        self.sources = set()

    def begin(self, nodes):
        # C023 can replay relinking internally; retain only the latest invocation.
        self.groups = {}
        self.sources = set()
        self.nodes = copy.deepcopy(nodes)  # original image coordinates, before stabilization
        plain = self.ns['nodes_by_id_to_plain'](nodes)
        plain = {i: (t, *(max(0, int(round(v))) for v in zyx)) for i, (t, *zyx) in plain.items()}
        self.p2g, self.g2p = self.ns['match_nodes_bipartite'](plain, self.gt_nodes, max_dist=7.)
        for pid, gid in self.p2g.items():
            kids = self.gt_out.get(gid, [])
            if len(kids) != 1 or kids[0] not in self.g2p:
                continue
            child = kids[0]
            # Exclude the divider, its children and immediately post-division links.
            parents = self.gt_in.get(gid, [])
            if any(len(self.gt_out.get(g, [])) > 1 for g in parents):
                continue
            if len(self.gt_out.get(child, [])) > 1:
                continue
            if int(self.gt_nodes[child][0]) == int(self.gt_nodes[gid][0]) + 1:
                self.sources.add(pid)

    def capture(self, phase, source_ids, target_ids, raw, motion, prob, cost, rows, cols):
        if phase == 'seed':
            return
        assigned = {int(a): int(b) for a, b in zip(rows, cols) if np.isfinite(raw[a, b])}
        targets = np.asarray(target_ids, dtype=np.int64)
        for i, sid in enumerate(source_ids):
            if sid not in self.sources:
                continue
            jj = np.flatnonzero(np.isfinite(raw[i]))
            if not len(jj):
                self.groups.pop(sid, None)
                continue
            values = cost[i, jj]
            order = np.argsort(values, kind='stable')
            ranks = np.empty(len(jj)); ranks[order] = np.arange(len(jj))
            features = np.column_stack([raw[i, jj], motion[i, jj], prob[i, jj], values,
                                        ranks, values - values.min(), np.full(len(jj), len(jj))])
            chosen = target_ids[assigned[i]] if i in assigned else -1
            # Later refinement rounds and relaxed passes overwrite earlier records.
            self.groups[sid] = (targets[jj].copy(), features.astype(np.float32), int(chosen), phase)


def install_recorder(ns, recorder):
    source = next(ast.get_source_segment(code, node)
                  for code in code_cells(BASE) for node in ast.parse(code).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'motion_relink_edges'
                  and 'def assign_pass' in ast.get_source_segment(code, node))
    patches = [
        ('        flow_exclude_um: float = 0.0,\n',
         '        flow_exclude_um: float = 0.0,\n        probe_phase: str = "",\n'),
        ('        row_ind, col_ind = linear_sum_assignment(cost)\n',
         '        row_ind, col_ind = linear_sum_assignment(cost)\n'
         '        _REID_PROBE_CAPTURE(probe_phase, source_ids, target_ids, raw_dist, motion_dist, prob_matrix, cost, row_ind, col_ind)\n'),
        ('seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)',
         'seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow, probe_phase="seed")'),
        ('matches = assign_pass(pass_sources, pass_targets, gate_um, flow, flow_exclude_um)',
         'matches = assign_pass(pass_sources, pass_targets, gate_um, flow, flow_exclude_um, probe_phase=pass_name)'),
    ]
    for old, new in patches:
        if source.count(old) != 1:
            raise ValueError('relink capture anchor mismatch: ' + old)
        source = source.replace(old, new, 1)
    original_outer = ns['motion_relink_edges']
    ns['_REID_PROBE_CAPTURE'] = recorder.capture
    exec(compile(source, 'C023:read_only_reid_capture', 'exec'), ns)
    ns['_unstabilized_motion_relink_edges'] = ns['motion_relink_edges']

    def outer(nodes, stats, learned_edge_probs=None):
        recorder.begin(nodes)
        return original_outer(nodes, stats, learned_edge_probs)

    ns['motion_relink_edges'] = outer


def setup_ns(run):
    ns = build_namespace(BASE, {}, run / 'edge_cache')
    ns['TEST_DIR'] = ROOT / 'data/train'
    frames, heatmaps = install_frame_caches(ns)
    return ns, frames, heatmaps


def score(ns, nodes, edges, gt_nodes, gt_edges, gt_path):
    plain = ns['nodes_by_id_to_plain'](nodes)
    plain = {i: (t, *(max(0, int(round(v))) for v in zyx)) for i, (t, *zyx) in plain.items()}
    return ns['score_sample'](plain, [(e['source_id'], e['target_id']) for e in edges],
                             gt_nodes, gt_edges, ns['read_estimated_true_node_count'](gt_path))


def extract_movie(stem, split, out, smoke=False):
    run = CONTROL / ('control_fp32_' + split)
    with (out / 'logs' / (stem + ('.smoke.log' if smoke else '.log'))).open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        ns, frames, heatmaps = setup_ns(run)
        path = next((run / 'predictions').rglob(stem + '.geff'))
        gt_path = ROOT / 'data/train' / (stem + '.geff')
        nodes, edges = load_raw_graph(ns, path)
        gt_nodes, gt_edges = ns['graph_to_plain'](ns['graph_from_geff'](gt_path))
        if smoke:
            nodes = {i: n for i, n in nodes.items() if 25 <= int(n['t']) <= 28}
            edges = [e for e in edges if e['source_id'] in nodes and e['target_id'] in nodes]
            baseline = ns['filter_output_graph'](copy.deepcopy(nodes), copy.deepcopy(edges), dataset=stem,
                                                 deepcenter_bundle=ns['DEEPCENTER_VETO_DETECTOR'])
        recorder = Recorder(ns, gt_nodes, gt_edges)
        install_recorder(ns, recorder)
        final_nodes, final_edges, stats = ns['filter_output_graph'](copy.deepcopy(nodes), copy.deepcopy(edges),
                                    dataset=stem, deepcenter_bundle=ns['DEEPCENTER_VETO_DETECTOR'])
        if smoke:
            assert baseline[0] == final_nodes and baseline[1] == final_edges and baseline[2] == stats, 'capture changes C023 output'
        else:
            metrics = score(ns, final_nodes, final_edges, gt_nodes, gt_edges, gt_path)
            ref = pd.read_csv(OLD / f'control_fp32_{split}.csv').set_index('stem').loc[stem]
            for key in ('edge_tp', 'edge_fp', 'edge_fn', 'div_tp', 'div_fp', 'div_fn', 'adjusted_edge_jaccard'):
                assert abs(float(metrics[key]) - float(ref[key])) < 1e-10, (stem, key, metrics[key], ref[key])
            assert len(final_nodes) == ref['nodes'] and len(final_edges) == ref['edges']
        public, public_source = public_namespace(ns)
        # Only appearance and size blocks are analysed; context/DeepCenter excluded.
        # Compute all frame nodes so the reused descriptor's density has its normal semantics.
        by_t = {}
        for i, n in recorder.nodes.items():
            by_t.setdefault(int(n['t']), []).append((i, float(n['z']), float(n['y']), float(n['x'])))
        needed_times = {int(recorder.nodes[s]['t']) + dt for s in recorder.groups for dt in (0, 1)}
        for t in sorted(needed_times):
            frame_nodes = sorted(by_t.get(t, []))
            if len(frame_nodes) < 2:
                raise ValueError('public descriptor requires at least two nodes per frame')
            public['reid_descriptors_for_frame'](stem, t, frame_nodes, frames, None, heatmaps)
        desc = public['REID_DESC'].get(stem, {})
        source_ids, target_ids, xs, ds, dt, labels, selected, phases = [], [], [], [], [], [], [], []
        eligible = 0
        for sid, (tids, geom, chosen, phase) in sorted(recorder.groups.items()):
            if len(tids) < 2 or sid not in desc:
                continue
            gid = recorder.p2g[sid]
            truth = recorder.g2p[recorder.gt_out[gid][0]]
            if truth not in tids:  # record reachable annotated successor comparisons only
                continue
            eligible += 1
            for j, tid in enumerate(tids):
                tid = int(tid)
                if tid not in desc:
                    raise ValueError('descriptor missing for candidate')
                label = 1 if tid == truth else (0 if tid in recorder.p2g else -1)
                source_ids.append(sid); target_ids.append(tid); xs.append(geom[j])
                ds.append(desc[sid]); dt.append(desc[tid]); labels.append(label)
                selected.append(int(tid == chosen)); phases.append(phase)
        if not xs:
            raise ValueError('no ambiguous GT-reachable groups')
        arrays = dict(source=np.array(source_ids, np.int64), target=np.array(target_ids, np.int64),
                      geometry=np.array(xs, np.float32), ds=np.array(ds, np.float32), dt=np.array(dt, np.float32),
                      label=np.array(labels, np.int8), selected=np.array(selected, np.int8), phase=np.array(phases),
                      stem=np.array(stem), split=np.array(split))
        assert np.isfinite(arrays['geometry']).all() and np.isfinite(arrays['ds']).all() and np.isfinite(arrays['dt']).all()
        destination = out / ('smoke' if smoke else 'pairs') / (stem + '.npz')
        destination.parent.mkdir(exist_ok=True)
        np.savez_compressed(destination, **arrays)
        info = dict(stem=stem, groups=eligible, pairs=len(labels), positive=labels.count(1),
                    known_negative=labels.count(0), unknown=labels.count(-1),
                    capture_matches_baseline=True, public_subset_sha256=hashlib.sha256(public_source.encode()).hexdigest())
        save_json(destination.with_suffix('.json'), info)
        frames.clear(); heatmaps.clear()
    return info


def analyse(out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    data = []
    for file in sorted((out / 'pairs').glob('*.npz')):
        with np.load(file) as z:
            a = {k: z[k].copy() for k in ['geometry', 'ds', 'dt', 'label', 'selected', 'source']}
            a['movie'] = np.repeat(str(z['stem']), len(a['label']))
        data.append(a)
    all_data = {k: np.concatenate([a[k] for a in data]) for k in data[0]}
    group = np.array([m.split('_')[0] for m in all_data['movie']])
    # Load the same public pair-feature implementation; no image/GT evaluation needed.
    public, _ = public_namespace(dict(VOXEL_SCALE_UM=(1.4, .406, .406), read_test_frame=None, deepcenter_score_point=None))
    result = dict(kind='cross_embryo_diagnostic_not_official_score', folds=[], geometry_features=GEOMETRY,
                  appearance_features=[public['REID_PAIR_NAMES'][i] for i in APPEARANCE_INDEX],
                  unknown_training_labels_excluded=True, model_recipe=dict(max_iter=150, max_leaf_nodes=7,
                      learning_rate=.05, min_samples_leaf=30, l2_regularization=10., early_stopping=False, random_state=3401))
    for test_group in ['44b6', '6bba']:
        train = (group != test_group) & (all_data['label'] >= 0)
        test = group == test_group
        known_test = test & (all_data['label'] >= 0)
        if min(np.sum(all_data['label'][train] == v) for v in [0, 1]) < 20:
            result['folds'].append(dict(test_embryo=test_group, status='insufficient_known_training_pairs'))
            continue
        train_desc = np.concatenate([all_data['ds'][train], all_data['dt'][train]])
        mu, sd = train_desc.mean(0), train_desc.std(0)
        public['REID_SCALER'] = dict(mu=mu, sd=np.maximum(sd, 1e-3))
        g = all_data['geometry']
        f = public['reid_pair_features'](all_data['ds'], all_data['dt'], g[:, 0], g[:, 1], g[:, 2], g[:, 4], g[:, 5], g[:, 6])
        features = {'geometry': g, 'geometry_appearance': np.concatenate([g, f[:, APPEARANCE_INDEX]], axis=1)}
        fold = dict(test_embryo=test_group, train_movies=sorted(set(all_data['movie'][train])),
                    test_movies=sorted(set(all_data['movie'][test])), known_train_pairs=int(train.sum()),
                    known_test_pairs=int(known_test.sum()), models={})
        probabilities = {}
        for name, x in features.items():
            model = HistGradientBoostingClassifier(**result['model_recipe'])
            model.fit(x[train], all_data['label'][train])
            prob = model.predict_proba(x[test])[:, 1]
            probabilities[name] = prob
            test_labels = all_data['label'][test]
            valid = test_labels >= 0
            fold['models'][name] = dict(auc=float(roc_auc_score(test_labels[valid], prob[valid])),
                                  ap=float(average_precision_score(test_labels[valid], prob[valid])))
        tested = {k: v[test] for k, v in all_data.items()}
        grouped = {}
        for i, (movie, sid) in enumerate(zip(tested['movie'], tested['source'])):
            grouped.setdefault((str(movie), int(sid)), []).append(i)
        movie_counts = {}
        for (movie, sid), indices in grouped.items():
            ids = np.array(indices)
            m = movie_counts.setdefault(movie, dict(groups=0, c023_selected_correct=0, geometry_correct=0,
                  geometry_appearance_correct=0, appearance_fixes_geometry=0, appearance_harms_geometry=0,
                  fixed_margin_changes=0, fixed_margin_fixes_c023=0, fixed_margin_harms_c023=0))
            m['groups'] += 1
            old_correct = bool(np.any((tested['selected'][ids] == 1) & (tested['label'][ids] == 1)))
            m['c023_selected_correct'] += int(old_correct)
            correct = {}
            for name, prob in probabilities.items():
                best = ids[int(np.argmax(prob[ids]))]
                correct[name] = bool(tested['label'][best] == 1)
                m[name + '_correct'] += int(correct[name])
            m['appearance_fixes_geometry'] += int(correct['geometry_appearance'] and not correct['geometry'])
            m['appearance_harms_geometry'] += int(correct['geometry'] and not correct['geometry_appearance'])
            prob = probabilities['geometry_appearance']; order = ids[np.argsort(prob[ids], kind='stable')]
            best = order[-1]
            # Fixed diagnostic selection, not tuned and not yet a degree-valid assignment.
            if len(order) > 1 and prob[best] >= .8 and prob[best] - prob[order[-2]] >= .2 and tested['selected'][best] == 0:
                m['fixed_margin_changes'] += 1
                m['fixed_margin_fixes_c023'] += int(correct['geometry_appearance'] and not old_correct)
                m['fixed_margin_harms_c023'] += int(old_correct and not correct['geometry_appearance'])
        totals = {k: sum(v[k] for v in movie_counts.values()) for k in next(iter(movie_counts.values()))}
        fold.update(status='complete', totals=totals, movies=movie_counts)
        fold['appearance_net_over_geometry'] = totals['appearance_fixes_geometry'] - totals['appearance_harms_geometry']
        fold['fixed_margin_net_over_c023'] = totals['fixed_margin_fixes_c023'] - totals['fixed_margin_harms_c023']
        result['folds'].append(fold)
        print('FOLD',test_group, json.dumps(totals), flush=True)
    # Compute allocation only. Annotation agreement is not official graph score.
    result['advance_to_replay_review'] = all(f.get('status') == 'complete' and
        f['appearance_net_over_geometry'] >= 5 and f['fixed_margin_net_over_c023'] >= 5 for f in result['folds'])
    result['limitations'] = ['Public detector already trained on both embryos.',
        'Unknown pairs excluded from fitting; ranking includes them, assessed only for reachable annotated successors.',
        'Capture-time node matching can differ from final matching after later stages.',
        'Per-source ranking ignores assignment competition and divisions; not an official-score improvement.',
        'Public DeepCenter descriptor disabled; context/centre features excluded to isolate image appearance.']
    save_json(out / 'analysis.json', result)
    lines = ['# C034 cross-embryo appearance diagnostic', '',
             'Existing C023 candidates, unchanged original costs/assignments; no new official-score claim.', '',
             '| Test embryo | Groups | Appearance net vs geometry | Fixed-margin net vs C023 |',
             '|---|---:|---:|---:|']
    for f in result['folds']:
        if f.get('status') == 'complete':
            lines.append(f"| {f['test_embryo']} | {f['totals']['groups']} | {f['appearance_net_over_geometry']:+d} | {f['fixed_margin_net_over_c023']:+d} |")
        else:
            lines.append(f"| {f['test_embryo']} | insufficient known labels | — | — |")
    lines += ['', 'Advance to official-replay implementation review: '+str(result['advance_to_replay_review']), '',
              'This gate requires both cross-embryo directions to show >=5 net gains in each comparison; it is a compute rule, not proof.', '',
              *['- ' + x for x in result['limitations']]]
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['smoke', 'run', 'analyse'])
    ap.add_argument('--out', type=Path, default=DEST)
    ap.add_argument('--max-hours', type=float, default=6)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for sub in ['logs', 'pairs', 'smoke']:
        (out / sub).mkdir(exist_ok=True)
    if args.command == 'analyse':
        with threadpool_limits(limits=4): analyse(out)
        return
    lock = out / 'probe.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    state_path = out / ('smoke_status.json' if args.command == 'smoke' else 'status.json')
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(jobs={})
    tracked = [Path(__file__), BASE, PUBLIC, ROOT / 'src/eval_pp_variants_local.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    started = time.time()
    try:
        if state.get('source_hashes') and state['source_hashes'] != hashes:
            raise ValueError('source changed; review before reusing completed extraction')
        state.update(status='running', pid=os.getpid(), started=stamp(), source_hashes=hashes)
        state.pop('error', None); save_json(state_path, state)
        public, public_source = public_namespace(dict(VOXEL_SCALE_UM=(1.4, .406, .406), read_test_frame=None, deepcenter_score_point=None))
        (out / 'public_descriptor_subset.py').write_text(public_source+'\n', encoding='utf-8')
        save_json(out / 'public_provenance.json', dict(source=str(PUBLIC), sha256=sha(PUBLIC),
            url='https://www.kaggle.com/code/arnav170/biohub-reid3s', author='arnav170',
            use='unmodified descriptor functions/constants for local diagnostic; no public model or notebook training loop executed'))
        plan = [('heldout12', s) for s in (ROOT/'experiments/candidates/c012_v1284_head/heldout_stems.txt').read_text().split()]
        plan += [('confirm10', s) for s in (ROOT/'experiments/candidates/c012_v1284_head/confirm_stems.txt').read_text().split()]
        if args.command == 'smoke':
            plan = [('heldout12', s) for s in ['44b6_12dfb391', '6bba_05db0fb1']]
        else:
            smoke = json.loads((out / 'smoke_status.json').read_text(encoding='utf-8'))
            assert smoke['status'] == 'complete' and smoke['source_hashes'] == hashes, 'real smoke required on exact sources'
        for split, stem in plan:
            if state['jobs'].get(stem, {}).get('status') == 'complete':
                continue
            if time.time() - started > args.max_hours * 3600:
                raise TimeoutError('finite extraction time budget reached')
            state['current'] = stem; save_json(state_path, state)
            print(stamp(), 'extract', split, stem, flush=True)
            t0 = time.time()
            with threadpool_limits(limits=4): info = extract_movie(stem, split, out, args.command == 'smoke')
            state['jobs'][stem] = dict(status='complete', seconds=time.time()-t0, **info)
            save_json(state_path, state); print(json.dumps(state['jobs'][stem]), flush=True)
        if args.command == 'run':
            state['current'] = 'cross_embryo_analysis'; save_json(state_path, state)
            with threadpool_limits(limits=4): result = analyse(out)
            state['advance_to_replay_review'] = result['advance_to_replay_review']
        state.update(status='complete', current=None, ended=stamp()); save_json(state_path, state)
    except Exception as exc:
        state.update(status='failed', error=str(exc), ended=stamp()); save_json(state_path, state)
        raise
    finally:
        os.close(fd); lock.unlink()


if __name__ == '__main__':
    main()
