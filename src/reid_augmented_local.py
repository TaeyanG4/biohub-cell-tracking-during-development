#!/usr/bin/env python3
"""C035: GT-supervised appearance learning, then a read-only C023 diagnostic.

Reuses C034's recorder and the official replay namespace, C031's crop utility
and small CNN backbone. This program never modifies a tracking graph or submits.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from division_cnn_train import DivisionCNN
from division_crops_extract import CROP, crop_at
from reid_probe_local import (BASE, CONTROL, OLD, Recorder, install_recorder,
                              load_raw_graph, save_json, score, setup_ns, sha, stamp)

DEST = ROOT / 'experiments/candidates/c035_augmented_reid'
C034 = ROOT / 'experiments/candidates/c034_appearance_reid'
SCALE = np.array([1.625, .40625, .40625])
RECIPE = dict(seed=3501, movies_per_embryo=12, min_eligible_links=120,
              links_per_movie=192, negative_separation_um=[6., 40.],
              crop=list(CROP), width=8, embedding=32, steps=1200, batch=48,
              lr=.0003, weight_decay=.0001, margin=.15, temperature=.1,
              arms=['basic', 'weak_aug'], cosine_gain=.10, cosine_margin=.05,
              max_cost_increase=1., min_net=5, min_positive_movies=2)


def stem_list(path):
    return [s for s in re.split(r'[,\s]+', path.read_text(encoding='utf-8-sig').strip()) if s]


def evaluation_plan():
    folder = ROOT / 'experiments/candidates/c012_v1284_head'
    return [(split, s) for split, f in [('heldout12', 'heldout'), ('confirm10', 'confirm')]
            for s in stem_list(folder / (f + '_stems.txt'))]


def data_meta(stem):
    folder = ROOT / 'data/train' / (stem + '.zarr')
    return {str(p.relative_to(ROOT)): sha(p) for p in [folder/'zarr.json', folder/'0/zarr.json']}


def gt_triplets(ns, stem):
    """Unique consecutive links and nearest distinct annotated child-frame nucleus.

    No unannotated detection is used as a negative. Reject divisions and their
    two-step lineage neighbourhood, duplicate/near-coincident GT centres, and
    cross-time/gap links. Negative identity is defined at the same child frame.
    """
    nodes, edges = ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(stem+'.geff')))
    children, parents, frames = (collections.defaultdict(list) for _ in range(3))
    for a, b in edges:
        children[a].append(b); parents[b].append(a)
    for i, n in nodes.items():
        frames[int(n[0])].append(i)
    unsafe = {a for a, b in children.items() if len(b) > 1}
    frontier = set(unsafe)
    for _ in range(2):
        frontier = {v for i in frontier for v in children[i] + parents[i]} - unsafe
        unsafe.update(frontier)
    triples, distances = [], []
    for a in sorted(nodes):
        if a in unsafe or len(children[a]) != 1:
            continue
        b = children[a][0]
        if b in unsafe or len(parents[b]) != 1 or int(nodes[b][0]) != int(nodes[a][0]) + 1:
            continue
        candidates = sorted(i for i in frames[int(nodes[b][0])] if i != b and i not in unsafe)
        if not candidates:
            continue
        d = np.linalg.norm((np.array([nodes[i][1:] for i in candidates]) - np.array(nodes[b][1:])) * SCALE, axis=1)
        valid = np.flatnonzero((d >= RECIPE['negative_separation_um'][0]) & (d <= RECIPE['negative_separation_um'][1]))
        # Near-coincident labels can denote ambiguous GT instances; skip source.
        if np.any(d < 2.) or not len(valid):
            continue
        j = int(valid[np.argmin(d[valid])])
        triples.append((a, b, candidates[j])); distances.append(float(d[j]))
    payload = repr((sorted(nodes.items()), sorted(edges))).encode()
    return nodes, np.array(triples, np.int64).reshape(-1, 3), np.array(distances), hashlib.sha256(payload).hexdigest()


def make_plan(out, tick):
    path = out / 'plan.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['recipe'] == RECIPE
        return result
    excluded = {s for _, s in evaluation_plan()}
    for p in sorted((ROOT/'experiments/candidates/c016_division_scorer').glob('stems_b0*.txt')):
        excluded.update(stem_list(p))
    assert len(excluded) == 97
    with contextlib.redirect_stdout(io.StringIO()):
        ns, _, _ = setup_ns(CONTROL/'control_fp32_heldout12')
    inventory = []
    for p in sorted((ROOT/'data/train').glob('*.geff')):
        if p.stem in excluded:
            continue
        tick('inventory:' + p.stem)
        nodes, triples, distances, digest = gt_triplets(ns, p.stem)
        inventory.append(dict(stem=p.stem, nodes=len(nodes), eligible_links=len(triples),
                              negative_distance_median=float(np.median(distances)) if len(distances) else None,
                              graph_sha256=digest, image_metadata=data_meta(p.stem)))
    selected = []
    for group in ['44b6', '6bba']:
        pool = [r for r in inventory if r['stem'].startswith(group) and r['eligible_links'] >= RECIPE['min_eligible_links']]
        pool.sort(key=lambda r: hashlib.sha256(('C035:' + r['stem']).encode()).hexdigest())
        if len(pool) < RECIPE['movies_per_embryo']:
            raise ValueError('insufficient separate training movies: ' + group)
        selected += pool[:RECIPE['movies_per_embryo']]
    result = dict(created=stamp(), recipe=RECIPE, excluded_official97=sorted(excluded),
                  selected=selected, inventory=inventory, selection='GT eligibility, then fixed SHA256 order; no score selection')
    save_json(path, result)
    return result


def patches(ns, frames, stem, ids, nodes):
    """Reuse C031's exact crop/normalization, decoding each needed frame once."""
    group = zarr.open_group(str(ROOT/'data/train'/(stem+'.zarr')), mode='r')
    quantiles = group.attrs['image_statistics']['quantiles']
    lo, hi = float(quantiles['0.001']), float(quantiles['0.999'])
    assert hi > lo
    out = np.empty((len(ids),) + CROP, np.float16)
    tzyx = np.array([nodes[int(i)] for i in ids], np.float32)
    for t in sorted(set(int(v) for v in tzyx[:, 0])):
        frame = ns['read_test_frame'](stem, t, frames)
        for row in np.flatnonzero(tzyx[:, 0].astype(int) == t):
            z, y, x = (int(round(float(v))) for v in tzyx[row, 1:])
            # A one-frame array makes all three utility channels identical.
            crop = crop_at(frame[None], 0, z, y, x)[1]
            out[row] = np.clip((crop-lo)/(hi-lo+1e-6), 0., 3.).astype(np.float16)
        frames.clear()
    assert np.isfinite(out).all()
    return out, tzyx


def extract_training(out, item, limited=False):
    stem = item['stem']
    with contextlib.redirect_stdout(io.StringIO()):
        ns, frames, heatmaps = setup_ns(CONTROL/'control_fp32_heldout12')
    nodes, triples, distances, digest = gt_triplets(ns, stem)
    assert digest == item['graph_sha256'] and data_meta(stem) == item['image_metadata']
    seed = RECIPE['seed'] + int(hashlib.sha256(stem.encode()).hexdigest()[:7], 16)
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(triples), min(len(triples), 2 if limited else RECIPE['links_per_movie']), replace=False)
    triples, distances = triples[keep], distances[keep]
    ids = np.unique(triples)
    crops, coords = patches(ns, frames, stem, ids, nodes)
    result = dict(crops=crops, triplets=np.searchsorted(ids, triples), ids=ids,
                  tzyx=coords, negative_distance_um=distances.astype(np.float32))
    if not limited:
        np.savez_compressed(out/'train'/f'{stem}.npz', **result)
        save_json(out/'train'/f'{stem}.json', dict(stem=stem, graph_sha256=digest,
            image_metadata=item['image_metadata'], patches=len(ids), triplets=len(triples),
            negative_distance_median=float(np.median(distances)), negatives_within_12um=int(np.sum(distances <= 12.))))
    frames.clear(); heatmaps.clear()
    return result


def extract_evaluation(out, split, stem):
    """Replay unchanged C023, check C034 pair identity, then export original crops."""
    with (out/'logs'/f'{stem}.replay.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        run = CONTROL / ('control_fp32_' + split)
        ns, frames, heatmaps = setup_ns(run)
        gt_path = ROOT/'data/train'/(stem+'.geff')
        nodes, edges = load_raw_graph(ns, next((run/'predictions').rglob(stem+'.geff')))
        gt_nodes, gt_edges = ns['graph_to_plain'](ns['graph_from_geff'](gt_path))
        rec = Recorder(ns, gt_nodes, gt_edges)
        install_recorder(ns, rec)
        final_nodes, final_edges, stats = ns['filter_output_graph'](nodes, edges, dataset=stem,
            deepcenter_bundle=ns['DEEPCENTER_VETO_DETECTOR'])
        metrics = score(ns, final_nodes, final_edges, gt_nodes, gt_edges, gt_path)
        ref = pd.read_csv(OLD/f'control_fp32_{split}.csv').set_index('stem').loc[stem]
        for k in ['edge_tp', 'edge_fp', 'edge_fn', 'div_tp', 'div_fp', 'div_fn', 'adjusted_edge_jaccard']:
            assert abs(float(metrics[k])-float(ref[k])) < 1e-10, (stem, k, metrics[k], ref[k])
        assert len(final_nodes) == ref['nodes'] and len(final_edges) == ref['edges']
        with np.load(C034/'pairs'/f'{stem}.npz') as old:
            data = {k: old[k].copy() for k in ['source', 'target', 'geometry', 'label', 'selected', 'phase']}
        expected = {sid for sid, (tids, _, _, _) in rec.groups.items() if len(tids) >= 2
                    and rec.g2p[rec.gt_out[rec.p2g[sid]][0]] in tids}
        assert expected == set(data['source']), 'C034 source group drift'
        for sid in np.unique(data['source']):
            jj = np.flatnonzero(data['source'] == sid)
            tids, geom, chosen, phase = rec.groups[int(sid)]
            truth = rec.g2p[rec.gt_out[rec.p2g[int(sid)]][0]]
            labels = np.array([1 if int(t) == truth else 0 if int(t) in rec.p2g else -1 for t in tids])
            assert np.array_equal(tids, data['target'][jj])
            assert np.array_equal(geom, data['geometry'][jj]), 'C034 geometry drift'
            assert np.array_equal(data['selected'][jj], tids == chosen)
            assert np.array_equal(data['label'][jj], labels)
            assert np.all(data['phase'][jj] == phase)
        ids = np.unique(np.r_[data['source'], data['target']])
        original = {i: (int(n['t']), float(n['z']), float(n['y']), float(n['x'])) for i, n in rec.nodes.items()}
        crops, coords = patches(ns, frames, stem, ids, original)
        data.update(crops=crops, ids=ids, tzyx=coords,
                    source_ix=np.searchsorted(ids, data['source']), target_ix=np.searchsorted(ids, data['target']))
        np.savez_compressed(out/'eval'/f'{stem}.npz', **data)
        info = dict(stem=stem, split=split, patches=len(ids), pairs=len(data['source']),
                    c034_pair_sha256=sha(C034/'pairs'/f'{stem}.npz'), c023_control_verified=True,
                    exact_c034_pairs=True, image_metadata=data_meta(stem), original_coordinates=True)
        save_json(out/'eval'/f'{stem}.json', info)
        frames.clear(); heatmaps.clear()
        return info


class AppearanceEncoder(DivisionCNN):
    def __init__(self):
        super().__init__(in_ch=1, width=RECIPE['width'])
        self.head = nn.Linear(RECIPE['width'] * 4, RECIPE['embedding'])

    def forward(self, x):
        return F.normalize(super().forward(x), dim=1, eps=1e-6)


def augmentation(x, arm):
    # One shared XY orientation for the whole triplet batch. Never mix z with XY.
    if torch.rand((), device=x.device) < .5:
        x = x.flip(-1)
    if torch.rand((), device=x.device) < .5:
        x = x.flip(-2)
    if torch.rand((), device=x.device) < .5:
        x = x.transpose(-1, -2)
    if arm == 'basic':
        return x.contiguous()
    n = len(x)
    theta = torch.eye(3, 4, device=x.device)[None].repeat(n, 1, 1)
    # Independent localisation uncertainty: +/-1.5 xy voxels, +/-0.3 z voxel.
    theta[:, :, 3] = (torch.rand(n, 3, device=x.device)*2-1) * x.new_tensor([3/32, 3/32, .6/8])
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    shape = (n, 1, 1, 1, 1)
    gain = .85 + .30*torch.rand(shape, device=x.device)
    background = .025*(2*torch.rand(shape, device=x.device)-1)
    noise = .015*torch.rand(shape, device=x.device)*torch.randn_like(x)
    smooth = F.avg_pool3d(F.pad(x, (1, 1, 1, 1, 0, 0), mode='replicate'), (1, 3, 3), stride=1)
    blend = .2*torch.rand(shape, device=x.device)
    return ((x*(1-blend)+smooth*blend)*gain+background+noise).clamp(0., 3.)


def objective(model, batch):
    a, b, c = model(batch).chunk(3)
    positive, negative = (a*b).sum(1), (a*c).sum(1)
    loss = F.softplus((negative-positive+RECIPE['margin']) / RECIPE['temperature']).mean()
    return loss, (positive > negative).float().mean()


def runtime_policy():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA GPU required for the bounded C035 study')
    return dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), dtype='float32',
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, amp=False,
                note='Local Ada training/diagnostic, not a T4 execution verification')


def smoke(out, plan):
    torch.manual_seed(RECIPE['seed'])
    checks = []
    for group in ['44b6', '6bba']:
        item = next(i for i in plan['selected'] if i['stem'].startswith(group))
        data = extract_training(out, item, limited=True)
        assert len(data['triplets']) == 2
        batch = torch.from_numpy(data['crops'][data['triplets'].T.reshape(-1)]).to('cuda', dtype=torch.float32)[:, None]
        assert batch.std() > 0 and batch.shape[2:] == CROP
        model = AppearanceEncoder().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'])
        start = model.head.weight.detach().clone()
        for arm in RECIPE['arms']:
            loss, acc = objective(model, augmentation(batch.clone(), arm))
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        assert not torch.equal(start, model.head.weight)
        model.eval()
        with torch.no_grad():
            embeddings = model(batch)
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(len(batch), device='cuda'), atol=1e-5)
        clone = AppearanceEncoder().cuda().eval(); clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            assert torch.equal(embeddings, clone(batch))
        checks.append(dict(embryo=group, stem=item['stem'], real_patches=len(data['crops']),
                           loss=float(loss), gradients_finite=True, weight_update=True, load_equal=True))
    return dict(status='complete', checks=checks, runtime=runtime_policy())


def train(out, plan, group, arm, tick):
    xs, triples, movie_ranges, offset = [], [], [], 0
    for item in plan['selected']:
        if not item['stem'].startswith(group):
            continue
        with np.load(out/'train'/(item['stem']+'.npz')) as d:
            x, t = d['crops'].copy(), d['triplets'].copy() + offset
        xs.append(x); triples.append(t); movie_ranges.append(t); offset += len(x)
    x = np.concatenate(xs)
    torch.manual_seed(RECIPE['seed'])
    rng = np.random.default_rng(RECIPE['seed'])
    model = AppearanceEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, RECIPE['steps'])
    log = []
    started = time.time()
    for step in range(RECIPE['steps']):
        if step % 100 == 0:
            tick(f'train:{group}:{arm}:{step}/{RECIPE["steps"]}')
        # Equal movie probability; sampling order is the same in both arms.
        movies = rng.integers(len(movie_ranges), size=RECIPE['batch'])
        tri = np.stack([movie_ranges[m][rng.integers(len(movie_ranges[m]))] for m in movies])
        batch = torch.from_numpy(x[tri.T.reshape(-1)]).to('cuda', dtype=torch.float32)[:, None]
        batch = augmentation(batch, arm)
        loss, accuracy = objective(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss: {group}/{arm}/{step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step(); schedule.step()
        if step % 100 == 0 or step == RECIPE['steps']-1:
            row = dict(step=step+1, loss=float(loss), training_triplet_accuracy=float(accuracy), seconds=time.time()-started)
            log.append(row)
            save_json(out/'models'/f'{group}_{arm}.training.json', log)
            print(stamp(), group, arm, json.dumps(row), flush=True)
    destination = out/'models'/f'{group}_{arm}.pt'
    torch.save(dict(state_dict=model.cpu().state_dict(), recipe=RECIPE,
                    train_embryo=group, arm=arm,
                    train_movies=[i['stem'] for i in plan['selected'] if i['stem'].startswith(group)]), destination)
    return dict(model_sha256=sha(destination), patches=len(x), triplets=sum(map(len, triples)),
                steps=RECIPE['steps'], seconds=time.time()-started, final_training=log[-1])


@torch.no_grad()
def embeddings(model, x):
    result = []
    for start in range(0, len(x), 128):
        batch = torch.from_numpy(x[start:start+128]).to('cuda', dtype=torch.float32)[:, None]
        result.append(model(batch).cpu().numpy())
    return np.concatenate(result)


def evaluate_embeddings(out, tick):
    models = {}
    for group in ['44b6', '6bba']:
        for arm in RECIPE['arms']:
            checkpoint = torch.load(out/'models'/f'{group}_{arm}.pt', map_location='cpu', weights_only=True)
            assert checkpoint['recipe'] == RECIPE and checkpoint['train_embryo'] == group
            model = AppearanceEncoder().cuda().eval()
            model.load_state_dict(checkpoint['state_dict'])
            models[group, arm] = model
    for _, stem in evaluation_plan():
        tick('diagnostic:' + stem)
        with np.load(out/'eval'/f'{stem}.npz') as d:
            data = {k: d[k].copy() for k in ['source', 'target', 'geometry', 'selected', 'label', 'source_ix', 'target_ix']}
            x = d['crops'].copy()
        train_group = '6bba' if stem.startswith('44b6') else '44b6'
        for arm in RECIPE['arms']:
            embed = embeddings(models[train_group, arm], x)
            data[arm] = (embed[data['source_ix']]*embed[data['target_ix']]).sum(1)
        # Cheap untrained control. No shift search, no fitted weights.
        pixels = torch.from_numpy(x.astype(np.float32))[:, None]
        pixels = F.avg_pool3d(pixels, (1, 2, 2)).flatten(1)
        pixels = F.normalize(pixels-pixels.mean(1, keepdim=True), dim=1).numpy()
        data['patch_ncc'] = (pixels[data['source_ix']]*pixels[data['target_ix']]).sum(1)
        np.savez_compressed(out/'scores'/f'{stem}.npz', **data)


def analyse(out):
    rows = []
    for _, stem in evaluation_plan():
        with np.load(out/'scores'/f'{stem}.npz') as z:
            data = {k: z[k].copy() for k in z.files}
        for arm in RECIPE['arms'] + ['patch_ncc']:
            row = dict(stem=stem, embryo=stem[:4], arm=arm, groups=0, c023_correct=0,
                       cost_correct=0, appearance_correct=0, rank_fixes_cost=0, rank_harms_cost=0,
                       changes=0, fixes_c023=0, harms_c023=0, unknown_choices=0)
            for sid in np.unique(data['source']):
                ids = np.flatnonzero(data['source'] == sid)
                labels, sim, geo = data['label'][ids], data[arm][ids], data['geometry'][ids]
                assert np.sum(labels == 1) == 1 and len(ids) >= 2
                row['groups'] += 1
                old = np.flatnonzero(data['selected'][ids])
                assert len(old) <= 1
                old_correct = bool(len(old) and labels[old[0]] == 1)
                cost_correct = bool(labels[np.argmin(geo[:, 3])] == 1)
                order = np.argsort(-sim, kind='stable'); top, second = order[:2]
                correct = bool(labels[top] == 1)
                row['c023_correct'] += int(old_correct); row['cost_correct'] += int(cost_correct)
                row['appearance_correct'] += int(correct)
                row['rank_fixes_cost'] += int(correct and not cost_correct)
                row['rank_harms_cost'] += int(cost_correct and not correct)
                change = (len(old) == 1 and top != old[0] and
                          sim[top]-sim[old[0]] >= RECIPE['cosine_gain'] and
                          sim[top]-sim[second] >= RECIPE['cosine_margin'] and
                          geo[top, 3] <= geo[old[0], 3] + RECIPE['max_cost_increase'])
                if change:
                    row['changes'] += 1
                    row['fixes_c023'] += int(correct and not old_correct)
                    row['harms_c023'] += int(old_correct and not correct)
                    row['unknown_choices'] += int(labels[top] < 0)
            row['net_rank'] = row['rank_fixes_cost'] - row['rank_harms_cost']
            row['net_conservative'] = row['fixes_c023'] - row['harms_c023']
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out/'movie_diagnostics.csv', index=False)
    summary = []
    for (arm, embryo), group in table.groupby(['arm', 'embryo']):
        values = {k: int(group[k].sum()) for k in table.columns if k not in ['stem', 'embryo', 'arm']}
        positive = int((group['net_conservative'] > 0).sum())
        gate = (values['net_rank'] >= RECIPE['min_net'] and values['net_conservative'] >= RECIPE['min_net']
                and positive >= RECIPE['min_positive_movies'])
        summary.append(dict(arm=arm, test_embryo=embryo, **values, positive_net_movies=positive,
                            negative_net_movies=int((group['net_conservative'] < 0).sum()), compute_gate=gate))
    advancing = [arm for arm in RECIPE['arms'] if all(r['compute_gate'] for r in summary if r['arm'] == arm)]
    result = dict(kind='conditional_candidate_ranking_not_official_graph_score', completed=stamp(),
                  recipe=RECIPE, folds=summary, advance_to_replay_review=advancing,
                  limitations=[
                      'All public detectors already saw both embryos; this is not independent whole-pipeline validation.',
                      'Training excludes all official97 movie IDs but only two embryos exist.',
                      'Distinct GT cells give real negatives, often easier and farther apart than actual relink candidates.',
                      'Unknown C034 targets were never used as negative training labels. Ranking success means retrieval of the matched GT child, not biological truth for every unmatched detection.',
                      'Only sources with an annotated reachable successor are evaluated. Missing detections and assignment conflicts are outside this diagnostic.',
                      'Movie effects are reported; pair counts are not independent samples. No threshold/checkpoint search on the target embryo.',
                      'Ranking and conservative proposals do not execute a valid global graph update. Official replay and actual T4 checks remain mandatory.',
                      'Patch NCC is a fixed, unaligned diagnostic control, not a tested registration/linking algorithm.'])
    save_json(out/'analysis.json', result)
    lines = ['# C035 GT-supervised appearance diagnostic', '', 'No modified-graph official score or submission is produced by this study.', '',
             '| Arm | Test embryo | Rank net vs cost | Conservative fixes / harms | Net | Positive movies | Compute gate |',
             '|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        lines.append(f"| {r['arm']} | {r['test_embryo']} | {r['net_rank']} | {r['fixes_c023']} / {r['harms_c023']} | {r['net_conservative']} | {r['positive_net_movies']} | {r['compute_gate']} |")
    lines += ['', 'Arms warranting graph-integration review: ' + (', '.join(advancing) or 'none'), '',
              *['- ' + s for s in result['limitations']]]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['run', 'analyse'])
    ap.add_argument('--out', type=Path, default=DEST)
    ap.add_argument('--max-hours', type=float, default=6.)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ['train', 'eval', 'models', 'scores', 'logs']:
        (out/name).mkdir(exist_ok=True)
    if args.command == 'analyse':
        analyse(out)
        return
    lock = out/'study.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    status = out/'status.json'
    state = json.loads(status.read_text(encoding='utf-8')) if status.exists() else dict(jobs={})
    sources = [Path(__file__), BASE, ROOT/'src/reid_probe_local.py', ROOT/'src/division_cnn_train.py',
               ROOT/'src/division_crops_extract.py', ROOT/'src/eval_pp_variants_local.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    start = time.time()
    def tick(current):
        if time.time()-start > args.max_hours*3600:
            raise TimeoutError('finite C035 execution budget exhausted; review before restart')
        state.update(current=current, updated=stamp())
        save_json(status, state)
    def job(name, fn):
        entry = state['jobs'].get(name, {})
        if entry.get('status') == 'complete':
            # Verify every cached artifact before resuming a completed phase.
            assert all((ROOT/p).is_file() and sha(ROOT/p) == h for p, h in entry['artifacts'].items()), name
            return
        tick(name)
        t0 = time.time(); info, artifacts = fn()
        state['jobs'][name] = dict(status='complete', seconds=time.time()-t0, info=info,
            artifacts={str(p.relative_to(ROOT)): sha(p) for p in artifacts})
        tick(name + ':complete')
        print(stamp(), name, 'complete', flush=True)
    try:
        if state.get('source_hashes') and state['source_hashes'] != hashes:
            raise ValueError('source drift: do not reuse stale C035 artifacts; review provenance first')
        state.update(status='running', pid=os.getpid(), started=stamp(), source_hashes=hashes, runtime=runtime_policy())
        state.pop('error', None); save_json(status, state)
        with threadpool_limits(limits=4):
            tick('training_inventory')
            plan = make_plan(out, tick)
            if state.get('plan_sha256'):
                assert state['plan_sha256'] == sha(out/'plan.json')
            state['plan_sha256'] = sha(out/'plan.json')
            def do_smoke():
                result = smoke(out, plan); save_json(out/'smoke.json', result)
                return result, [out/'smoke.json']
            job('real_crop_gradient_smoke', do_smoke)
            for item in plan['selected']:
                def extract(item=item):
                    data = extract_training(out, item)
                    return dict(patches=len(data['crops']), triplets=len(data['triplets'])), [out/'train'/(item['stem']+'.npz'), out/'train'/(item['stem']+'.json')]
                job('train_crops:' + item['stem'], extract)
            for split, stem in evaluation_plan():
                job('eval_crops:' + stem, lambda split=split, stem=stem: (extract_evaluation(out, split, stem), [out/'eval'/f'{stem}.npz', out/'eval'/f'{stem}.json']))
            for group in ['44b6', '6bba']:
                for arm in RECIPE['arms']:
                    job(f'model:{group}:{arm}', lambda group=group, arm=arm: (train(out, plan, group, arm, tick), [out/'models'/f'{group}_{arm}.pt', out/'models'/f'{group}_{arm}.training.json']))
            job('cross_embryo_scores', lambda: (evaluate_embeddings(out, tick), [out/'scores'/f'{s}.npz' for _, s in evaluation_plan()]))
            tick('analysis')
            result = analyse(out)
            assert {str(p.relative_to(ROOT)): sha(p) for p in sources} == hashes, 'source changed during execution'
            state.update(status='complete', current=None, ended=stamp(), advance_to_replay_review=result['advance_to_replay_review'])
            save_json(status, state)
    except Exception as exc:
        state.update(status='failed', error=str(exc), ended=stamp(), traceback=traceback.format_exc())
        save_json(status, state)
        raise
    finally:
        os.close(fd); lock.unlink()


if __name__ == '__main__':
    main()
