"""C033 adapter for the public fixed structured-trajectory assignment.

Shared by the existing replay harness and the candidate notebook verbatim.
No GT, fitting, threshold sweep, node addition or coordinate modification.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import numpy as np


def install_structured_trajectory(ns, public_source, model_config):
    public = {'__name__': '__public_structured_trajectory__'}
    exec(compile(public_source, 'public_structured_trajectory.py', 'exec'), public)
    if tuple(model_config['features']) != tuple(public['FEATURES']):
        raise ValueError('structured model feature order mismatch')
    weights = np.asarray(model_config['weights'], dtype=np.float64)
    ns['STRUCTURED_TRAJECTORY_MODE'] = os.environ.get('BIOHUB_STRUCTURED_TRAJECTORY_MODE', 'off')
    inner = ns['filter_output_graph']

    def wrapper(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
        mode = str(ns['STRUCTURED_TRAJECTORY_MODE'])
        if mode == 'off':
            return inner(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
        if mode not in ('raw', 'stabilized'):
            raise ValueError('structured mode must be off/raw/stabilized')
        initial = {'nodes': copy.deepcopy(nodes_by_id), 'edges': copy.deepcopy(raw_edges)}
        nodes, edges, stats = inner(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=deepcenter_bundle)
        if not edges:
            return nodes, edges, stats
        if not dataset or not os.environ.get('BIOHUB_CACHE_DIR'):
            raise ValueError('structured stage requires this run\'s edge_cache')
        with np.load(Path(os.environ['BIOHUB_CACHE_DIR']) / (dataset + '.npz')) as dump:
            coords = dump['coords']
            admitted = dump['admitted'].copy()
            ids = np.asarray(sorted(initial['nodes']), dtype=np.int64)
            expected = np.array([[initial['nodes'][int(i)][k] for k in ('t', 'z', 'y', 'x')] for i in ids])
            if ids.min() < 0 or ids.max() >= len(coords) or not np.allclose(coords[ids], expected, atol=1e-4, rtol=0):
                raise ValueError('ILP/cache node identities or coordinates differ')
        final = {'nodes': copy.deepcopy(nodes), 'edges': copy.deepcopy(edges)}
        if mode == 'stabilized':
            scale = public['SCALE']
            displacements = {}
            for e in initial['edges']:
                a, b = initial['nodes'].get(int(e['source_id'])), initial['nodes'].get(int(e['target_id']))
                if a is None or b is None or int(b['t']) != int(a['t']) + 1 or float(e.get('edge_prob') or 0) < .5:
                    continue
                delta = np.array([b[k] - a[k] for k in ('z', 'y', 'x')]) * scale
                displacements.setdefault(int(a['t']), []).append(delta)
            shifts = {t: np.median(v, axis=0) for t, v in displacements.items() if len(v) >= 8}
            cumulative, acc = {}, np.zeros(3)
            for t in sorted({int(n['t']) for n in initial['nodes'].values()} | {int(n['t']) for n in nodes.values()}):
                cumulative[t] = acc.copy()
                if t in shifts:
                    acc += shifts[t]
            for graph in (initial, final):
                for node in graph['nodes'].values():
                    offset = cumulative[int(node['t'])] / scale
                    for k, delta in zip(('z', 'y', 'x'), offset):
                        node[k] = float(node[k]) - float(delta)
        groups = public['candidates'](initial, final)
        if not len(groups['parents']):
            return nodes, edges, stats
        matrix = public['features'](initial, final, groups, admitted)
        result, audit = public['apply'](final, groups, matrix, weights)
        probabilities = {(int(a), int(b)): float(p) for a, b, p, _ in admitted}
        for old, new in zip(edges, result['edges']):
            if int(old['source_id']) != int(new['source_id']):
                a, b = int(new['source_id']), int(new['target_id'])
                new['edge_prob'] = probabilities.get((a, b))
                new['distance_um'] = ns['edge_distance_um'](nodes[a], nodes[b])
                new['structured_reassigned'] = 1
        stats['structured_changed_edges'] = audit['changed_edges']
        stats['structured_protected_nodes'] = audit['protected_nodes']
        print(f"  [{dataset}] structured {mode}: changed={audit['changed_edges']}, protected={audit['protected_nodes']}", flush=True)
        # Always emit original coordinates, even when costs used shifted ones.
        return nodes, result['edges'], stats

    ns['filter_output_graph'] = wrapper
