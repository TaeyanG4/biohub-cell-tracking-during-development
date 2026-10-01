#!/usr/bin/env python3
"""Portable C046 packaging and four existing replay/writer/evaluator jobs."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
from build_fixed_model_candidates import asset_source, numerical_policy, DATASET
from c046_fixed_appearance_extension import ROOT, BASE, CONTROL, C041, DEST as STUDY, sha, save_json, stamp
from c038_followup_local import verify, graph
from eval_pp_variants_local import build_namespace, STEM_SETS
from run_last_days_local import Queue, replay_command

DEST = ROOT / 'state/c046_portable'
CANDIDATE = ROOT / 'experiments/candidates/c046_fixed_appearance'
NOTEBOOK = CANDIDATE / 'biohub-c046-fixed-appearance.ipynb'
ASSETS = ROOT / 'state/c042_c043_portable/dataset'
RUN = CONTROL / 'control_fp32_heldout12'
BEGIN = '# BEGIN C046 NUMERICAL POLICY\n'
END = '# END C046 NUMERICAL POLICY\n'


def prepare(out):
    assert not (out / 'plan.json').exists(), 'Existing immutable verification plan'
    out.mkdir(parents=True, exist_ok=True)
    assert json.loads((STUDY / 'status.json').read_text())['status'] == 'complete_review_required'
    assert json.loads((STUDY / 'decision.json').read_text())['status'] == 'local_review_complete_portable_candidate'
    prior = json.loads((STUDY / 'plan.json').read_text())['hashes']
    artifacts = json.loads((STUDY / 'artifact_hashes.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in prior.items())
    assert all(sha(STUDY / p) == h for p, h in artifacts.items())
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    assert not NOTEBOOK.exists(), 'Candidate already exists'
    asset = ASSETS / 'appearance_mean_cosine.pt'
    assert sha(asset) == sha(C041 / asset.name)
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    cells = nb['cells']
    cells[0]['source'] = ''.join(cells[0]['source']).replace("'''Biohub C023:", "'''Biohub C046:", 1).splitlines(keepends=True)
    cells[2]['source'] = (''.join(cells[2]['source']) + asset_source({asset.name: sha(asset)})).splitlines(keepends=True)
    block = BEGIN + '_c046_source = _ps.read_text()\n'
    block += "assert _c046_source.count('import torch\\n') == 1\n"
    block += "_c046_source = _c046_source.replace('import torch\\n', " + repr(numerical_policy()) + ", 1)\n"
    block += "compile(_c046_source, str(_ps), 'exec')\n_ps.write_text(_c046_source)\n" + END
    code = ''.join(cells[4]['source']); anchor = '_ps.write_text(_trial_source)\n'
    assert code.count(anchor) == 1
    cells[4]['source'] = code.replace(anchor, anchor + '\n' + block, 1).splitlines(keepends=True)
    appearance = (C041 / 'portable_appearance.py').read_text(encoding='utf-8')
    payload = f'\nexec(compile({appearance!r}, "c041_portable_appearance", "exec"), globals())\n'
    payload += "install_complementary(globals(), str(_fixed_asset('appearance_mean_cosine.pt')))\nC038_MODE = 'appearance'\n"
    code = ''.join(cells[5]['source']); anchor = '\nwrite_test_submission("base")\n'
    assert code.count(anchor) == 1
    cells[5]['source'] = code.replace(anchor, payload + anchor, 1).splitlines(keepends=True)
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']); compile(source, f'C046:cell{i}', 'exec')
        assert 'H:/' not in source and 'H:\\' not in source and 'from c038_followup_local' not in source
        assert 'BIOHUB_C037_CHECKPOINT' not in source
        cell['outputs'] = []; cell['execution_count'] = None
    nb['metadata']['title'] = NOTEBOOK.stem
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')
    meta = json.loads((BASE.parent / 'kernel-metadata.json').read_text())
    meta.update(id='taeyangg4/' + NOTEBOOK.stem, title=NOTEBOOK.stem, code_file=NOTEBOOK.name)
    meta['dataset_sources'] = sorted(set(meta['dataset_sources'] + [DATASET]))
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
    save_json(CANDIDATE / 'kernel-metadata.json', meta)
    # Execute only the embedded numeric patch on the actual unchanged C023 predictor.
    ps = out / 'predict_unet_transformer.py'
    original = ROOT / 'tmp/c023_output/tracking_repo/scripts/predict_unet_transformer.py'
    ps.write_text(original.read_text(encoding='utf-8'), encoding='utf-8')
    embedded = ''.join(cells[4]['source']).split(BEGIN, 1)[1].split(END, 1)[0]
    exec(compile(embedded, 'C046_embedded_numeric_patch', 'exec'), dict(_ps=ps))
    normalized = ps.read_text(encoding='utf-8')
    for quoted in ["Path('/kaggle/working')", 'Path("/kaggle/working")']:
        normalized = normalized.replace(quoted, "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    reference = RUN / '_work/tracking_repo/scripts/predict_unet_transformer.py'
    assert normalized == reference.read_text(encoding='utf-8'), 'Original C023 actual predictor mismatch'
    save_json(out / 'source_parity.json', dict(status='passed', predictor_matches_actual_C023=True,
              original_transformer_unchanged=True, portable_appearance_sha256=sha(C041 / 'portable_appearance.py'),
              appearance_model_sha256=sha(asset)))
    save_json(out / 'as_configured.json', {'as_configured': {}})
    table = pd.read_csv(STUDY / 'replay/heldout12_summary.csv')
    expected = float(table[(table.config == 'appearance') & (table.group == 'vis4')].iloc[0].proxy_score)
    inputs = {ROOT / p for p in prior} | {STUDY / p for p in artifacts}
    inputs.update([Path(__file__), ROOT / 'src/build_fixed_model_candidates.py', NOTEBOOK,
                   CANDIDATE / 'kernel-metadata.json', out / 'README.md', out / 'as_configured.json',
                   out / 'source_parity.json', ps, reference, original, asset,
                   STUDY / 'decision.json', STUDY / 'FINAL_REVIEW.md'])
    assert all(p.is_file() for p in inputs)
    save_json(out / 'plan.json', dict(created=stamp(), hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)},
              notebook=str(NOTEBOOK.relative_to(ROOT)), ref=meta['id'], expected_visible4=expected,
              total_jobs=4, policy='Original C023 inference, unchanged fixed appearance; actual12 replay/4 writer/evaluator. No Kaggle writes.'))
    print('C046 portable prepared; expected visible4', expected, '; pinned', len(inputs), flush=True)


def write_visible(out):
    os.environ['BIOHUB_FIXED_ASSET_DIR'] = str(ASSETS)
    work = out / 'writer'; work.mkdir(exist_ok=True)
    ns = build_namespace(NOTEBOOK, {}, RUN / 'edge_cache')
    assert ns['C038_MODE'] == 'appearance'
    ns.update(TEST_DIR=ROOT / 'data/train', REPO_DIR=work / 'writer_repo', test_stems=STEM_SETS['vis4'],
              SUBMISSION_PATH=work / 'submission.csv', RUN_STATS_PATH=work / 'run_stats.csv', predict_seconds=0)
    for stem in STEM_SETS['vis4']:
        source = next((RUN / 'predictions').rglob(stem + '.geff'))
        shutil.copytree(source, work / 'writer_repo/predictions/portable' / ns['METHOD'] / 'split_0' / source.name, dirs_exist_ok=True)
    ns['write_test_submission']('portable_visible4')
    stats = pd.read_csv(work / 'run_stats.csv')
    assert len(stats) == 4 and (stats.repair_fallback == 0).all() and (stats.deadline_degraded == 0).all()
    frame = pd.read_csv(work / 'submission.csv')
    for stem in STEM_SETS['vis4']:
        movie = frame[frame.dataset == stem]
        nodes = {int(r.node_id): tuple(int(getattr(r, k)) for k in ['t', 'z', 'y', 'x']) for r in movie[movie.row_type == 'node'].itertuples()}
        edges = {(int(r.source_id), int(r.target_id)) for r in movie[movie.row_type == 'edge'].itertuples()}
        assert (nodes, edges) == graph(STUDY / 'graphs/heldout12', stem, 'appearance'), ('Writer drift', stem)
    save_json(work / 'writer_parity.json', dict(status='passed', exact_graphs=4, repair_fallback=0,
              deadline_degraded=0, submission_sha256=sha(work / 'submission.csv')))


def verify_local(out):
    verify(pd.read_csv(out / 'replay/heldout12.csv'), pd.read_csv(STUDY / 'replay/heldout12.csv'), {'as_configured': 'appearance'})
    plan = json.loads((out / 'plan.json').read_text())
    result = json.loads((out / 'writer/evaluation/summary.json').read_text())
    assert abs(result['total_score'] - plan['expected_visible4']) < 1e-12
    assert json.loads((out / 'writer/writer_parity.json').read_text())['status'] == 'passed'
    save_json(out / 'local_verification.json', dict(status='passed', created=stamp(),
        notebook_sha256=sha(NOTEBOOK), metadata_sha256=sha(CANDIDATE / 'kernel-metadata.json'),
        visible4=result['total_score'], replay12='exact', writer4='exact', repair_fallback=0,
        deadline_degraded=0, submission_sha256=sha(out / 'writer/submission.csv'),
        next='Actual T4 source/model/original-Transformer/fallback/version parity before submission; no gain guarantee.'))


def run(out):
    plan = json.loads((out / 'plan.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in plan['hashes'].items()), 'Input drift'
    os.environ['BIOHUB_FIXED_ASSET_DIR'] = str(ASSETS)
    q = Queue(out, 3)
    try:
        q.state.update(plan_sha256=sha(out / 'plan.json'), total_jobs=4); q.save()
        study_plan = json.loads((STUDY / 'plan.json').read_text())
        stems = next(j['stems'] for j in study_plan['jobs'] if j['name'] == 'heldout12')
        q.run('replay12', replay_command(NOTEBOOK, RUN, stems, out / 'as_configured.json', out / 'replay/heldout12.csv'))
        q.run('writer4', [sys.executable, '-u', Path(__file__), 'write_visible', '--out', out])
        q.run('official4', [sys.executable, '-u', ROOT / 'src/evaluate_local.py', '--csv', out / 'writer/submission.csv',
              '--gt-dir', ROOT / 'data/visible_gt/train', '--out-dir', out / 'writer/evaluation'])
        q.run('verify_local', [sys.executable, '-u', Path(__file__), 'verify_local', '--out', out])
        assert all(sha(ROOT / p) == h for p, h in plan['hashes'].items()), 'Input drift'
        files = [p for folder in ['replay', 'writer'] for p in (out / folder).rglob('*') if p.is_file()]
        files.append(out / 'local_verification.json')
        save_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha(p) for p in files})
        q.close('complete_local_verified')
    except BaseException as exc:
        q.close('failed', exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'write_visible', 'verify_local'])
    parser.add_argument('--out', type=Path, default=DEST)
    args = parser.parse_args(); folder = args.out.resolve(); folder.relative_to(ROOT)
    globals()[args.command](folder)
