"""Fixed expanded-data whole-embryo transfer, using unchanged C023 execution."""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timedelta
import json
import math
import os
import sys
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import numpy as np
import pandas as pd
import torch
import c052_division_transformer as pilot
import c058_localizer_baseline as official
import evaluate_local
from c047_hard_example_study import read, verify_hashes
from reid_probe_local import ROOT, BASE, sha, save_json, stamp
from run_last_days_local import Queue, replay_command

CANDIDATE = ROOT/'experiments/candidates/c066_expanded_ordinary'
FIT = CANDIDATE/'fit'
OUT = CANDIDATE/'transfer97'
HISTORY = ROOT/'state/validation_audit_20260929'
PREFLIGHT = ROOT/'state/c066_transfer_preflight'


def membership():
    data = pd.read_csv(HISTORY/'actual_per_movie.csv')
    control = data[data.candidate == 'C023'].copy()
    assert len(control) == 97 and control.stem.is_unique
    expected = {s for j in read(ROOT/'experiments/candidates/c065_pooled_transformer/deploy/plan.json')['jobs'] for s in j['stems']}
    assert set(control.stem) == expected
    return control


def inference(job):
    command = pilot.old.inference(FIT, job['name'], OUT/(job['name']+'.txt'), [
        'BIOHUB_C037_CHECKPOINT='+job['checkpoint'], 'BIOHUB_C037_ALPHA=1.0',
        'BIOHUB_C037_CAPTURE_DIR=', 'BIOHUB_CACHE_EDGE_THRESHOLD=0.02'])
    command[command.index('--out')+1] = OUT/'e2e'
    return command


def runtime_proof(job):
    records = []
    for line in (OUT/'e2e'/job['name']/'predict.log').read_text(encoding='utf-8').splitlines():
        if 'C037_PRIMARY ' not in line:
            continue
        value = ast.literal_eval(line.split('C037_PRIMARY ', 1)[1].strip())
        assert value['train_embryo'] == job['train_embryo'] != job['test_embryo']
        assert value['step'] == 600 and value['alpha'] == 1.
        assert Path(value['path']).resolve() == Path(job['checkpoint']).resolve()
        records.append(value)
    assert records, job['name']
    return dict(job=job['name'], parsed_records=records, checkpoint_sha256=sha(Path(job['checkpoint'])))


def score_graph(path, stem):
    with np.load(path) as graph:
        assert np.issubdtype(graph['txyz'].dtype, np.integer)
        return official.official_score(graph['ids'], graph['txyz'], graph['edges'], ROOT/'data/train'/(stem+'.geff'))


def summaries(data):
    history = pd.read_csv(HISTORY/'actual_per_movie.csv')
    summarize = evaluate_local._load_official()[-1]
    groups = [('all97', data), ('all22', data[data.split != 'extension75'])]
    groups += [(s, data[data.split == s]) for s in ['heldout12', 'confirm10', 'extension75']]
    groups += [(e, data[data.embryo == e]) for e in ['44b6', '6bba']]
    rows = []
    for group, part in groups:
        current = summarize(part.to_dict('records'))
        row = dict(group=group, movies=len(part), score=current['score'],
                   adjusted_edge=current['adj_edge_jaccard'], division_jaccard=current['division_jaccard'])
        for key in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']:
            row[key] = int(part[key].sum())
        for name in ['C023', 'C052']:
            baseline = history[(history.candidate == name) & history.stem.isin(part.stem)]
            assert len(baseline) == len(part) and set(baseline.stem) == set(part.stem)
            previous = summarize(baseline.to_dict('records'))
            row['delta_vs_'+name] = current['score']-previous['score']
            row['edge_delta_vs_'+name] = current['adj_edge_jaccard']-previous['adj_edge_jaccard']
            deltas = part.set_index('stem').adj_edge_jaccard-baseline.set_index('stem').adj_edge_jaccard
            row['edge_wins_vs_'+name] = int((deltas > 1e-12).sum())
            row['edge_losses_vs_'+name] = int((deltas < -1e-12).sum())
            for key in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']:
                row[key+'_delta_vs_'+name] = int(part[key].sum()-baseline[key].sum())
        rows.append(row)
    return rows


def prepare():
    assert not (OUT/'plan.json').exists()
    assert read(FIT/'status.json')['status'] == 'complete_review_required'
    assert read(ROOT/'state/c066_fit_review.json')['status'] == 'passed'
    assert read(PREFLIGHT/'verification.json')['status'] == 'passed'
    verify_hashes(read(PREFLIGHT/'input_hashes.json'))
    verify_hashes(read(PREFLIGHT/'output_hashes.json'))
    inputs = {Path(__file__), OUT/'launch.ps1', CANDIDATE/'README.md', FIT/'plan.json', FIT/'output_hashes.json',
              ROOT/'state/c066_fit_review.json', PREFLIGHT/'verification.json', PREFLIGHT/'input_hashes.json', PREFLIGHT/'output_hashes.json',
              HISTORY/'actual_per_movie.csv', HISTORY/'independent_verification.json',
              ROOT/'experiments/candidates/c065_pooled_transformer/deploy/plan.json'}
    # Inherit the proven C023 source/model/97-input bindings, not its predictions.
    hashes = dict(read(ROOT/'experiments/candidates/c065_pooled_transformer/deploy/plan.json')['hashes'])
    hashes.update(read(FIT/'plan.json')['hashes'])
    hashes.update(read(FIT/'output_hashes.json'))
    control = membership(); jobs = []
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT/'variants.json', {'as_configured': {}})
    for group in ['44b6', '6bba']:
        other = '6bba' if group == '44b6' else '44b6'
        model = FIT/'models'/f'{other}_step600.pt'
        checkpoint = torch.load(model, map_location='cpu', weights_only=True)
        stems = sorted(control[control.embryo == group].stem)
        assert checkpoint['train_embryo'] == other and checkpoint['step'] == 600
        assert all(s.startswith(other) for s in checkpoint['train_movies'])
        assert not set(checkpoint['train_movies']) & set(stems)
        name = 'opposite_'+group
        (OUT/(name+'.txt')).write_text('\n'.join(stems)+'\n', encoding='utf-8')
        pilot.replay(OUT, name, stems, OUT/'e2e'/name)
        jobs.append(dict(name=name, test_embryo=group, train_embryo=other, stems=stems,
                         checkpoint=str(model), checkpoint_sha256=sha(model)))
    for folder in [FIT/'tracking_repo', evaluate_local.VENDOR_SRC]:
        inputs.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['c058_localizer_baseline.py', 'evaluate_local.py', 'run_kaggle_predict_local.py',
                 'v1284_capture_local.py', 'eval_pp_variants_local.py', 'c066_expanded_fit.py']:
        inputs.add(ROOT/'src'/name)
    inputs.update(p for p in OUT.iterdir() if p.is_file())
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
    verify_hashes(hashes)
    save_json(OUT/'plan.json', dict(created=stamp(), hashes=hashes, jobs=jobs, total_jobs=5,
        estimated_minutes=130, max_start_job_hours=3, membership=control[['stem','split','embryo']].to_dict('records'),
        metric_commit=evaluate_local.METRIC_COMMIT, actual_organizer_metric=True,
        estimate_basis='Prior fixed97 inference about102min, replay about20min plus hash/actual-score checks',
        whole_embryo_component_transfer=True, public_frozen_models_previously_exposed=True,
        no_deployment_prefix_router=True, no_kaggle_writes=True))
    print(json.dumps(dict(prepared=True, inputs=len(hashes), movies=len(control), jobs=5)), flush=True)


def analyse():
    plan = read(OUT/'plan.json'); by_stem = {r['stem']:r for r in plan['membership']}
    rows = []; proofs = []; controls = []
    for job in plan['jobs']:
        assert sha(Path(job['checkpoint'])) == job['checkpoint_sha256']
        proofs.append(runtime_proof(job))
        replay = pd.read_csv(OUT/'replay'/(job['name']+'.csv'))
        assert len(replay) == len(job['stems']) and set(replay.stem) == set(job['stems'])
        for stem in job['stems']:
            with np.load(OUT/'e2e'/job['name']/'edge_cache'/(stem+'.npz')) as new, np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as old:
                for key in ['coords', 'low_coords', 'low_score']:
                    assert np.array_equal(new[key], old[key], equal_nan=True), (stem, key)
            graph = OUT/'graphs'/job['name']/(stem+'_final.npz')
            row = score_graph(graph, stem)
            rows.append(dict(candidate='C066_opposite', **by_stem[stem], **row))
            controls.append(dict(stem=stem, frozen_detector_exact=True, graph_sha256=sha(graph)))
    data = pd.DataFrame(rows)
    assert len(data) == 97 and data.stem.is_unique and set(data.stem) == set(by_stem)
    data.to_csv(OUT/'official_per_movie97.csv', index=False)
    result = summaries(data); pd.DataFrame(result).to_csv(OUT/'official_summary.csv', index=False)
    save_json(OUT/'analysis.json', dict(status='complete_review_required', rows=result,
        runtime_proofs=proofs, exact_frozen_detector=controls, metric_commit=evaluate_local.METRIC_COMMIT,
        actual_organizer_metric=True, limitation='Two embryos with correlated crops; frozen public detectors exposed. Opposite-embryo additional Transformer fit only.'))
    print(json.dumps(result), flush=True)


def run():
    assert not (OUT/'status.json').exists()
    plan = read(OUT/'plan.json')
    deadline = datetime.fromisoformat('2026-09-30T00:00:00+09:00')
    assert datetime.now().astimezone()+timedelta(minutes=130+145+60+20) < deadline, 'Full remaining path no longer fits midnight'
    os.environ['PYTHONUTF8'] = '1'
    queue = Queue(OUT, 3)
    try:
        queue.state.update(total_jobs=5, plan_sha256=sha(OUT/'plan.json')); queue.save()
        verify_hashes(plan['hashes'])
        for job in plan['jobs']:
            name = job['name']
            queue.run('infer_'+name, inference(job))
            runtime_proof(job)
            queue.run('replay_'+name, replay_command(OUT/(name+'.ipynb'), OUT/'e2e'/name, job['stems'], OUT/'variants.json', OUT/'replay'/(name+'.csv')))
        queue.run('actual97_analysis', [sys.executable, '-X', 'utf8', '-u', Path(__file__), 'analyse'])
        verify_hashes(plan['hashes'])
        files = [p for folder in ['e2e', 'graphs', 'replay', 'logs'] for p in (OUT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        files += [OUT/'analysis.json', OUT/'official_per_movie97.csv', OUT/'official_summary.csv']
        save_json(OUT/'output_hashes.json', {str(p.relative_to(ROOT)):sha(p) for p in files})
        queue.close('complete_review_required')
    except BaseException as exc:
        queue.close('failed', exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare','run','analyse'])
    globals()[parser.parse_args().command]()
