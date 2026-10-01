#!/usr/bin/env python3
"""Finite, resumable LOCAL C032/C033 experiment queue. No Kaggle operations.

Orchestrates existing inference and official replay tools on one GPU serially.
Runs matched FP32 controls, three temporal modes, and two fixed structured modes
on heldout-12 and confirm-10. Promising arms advance to the remaining 75 movies.
Writes status.json, RESULTS.md and per-job logs; the agent need not wait/poll.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from build_temporal_context_candidate import BASE, DEST as TC, SOURCE, build_repo, build_notebook
from build_structured_candidate import ASSETS, DEST as ST, build as build_structured

TEMPORAL = ('mean_det', 'future_det', 'mean_det_head')
HEAD = ROOT/'artifacts/anvithpothula_v1284_head_s075/v1284_head.pt'
SPLITS = ROOT/'experiments/candidates/c012_v1284_head'
REPLAY = ROOT/'src/eval_pp_variants_local.py'
INFERENCE = ROOT/'src/run_kaggle_predict_local.py'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_stems(path):
    # C012 uses one stem per line; C016's b00/b01/b02 files use commas.
    stems = [s for s in re.split(r'[,\s]+', Path(path).read_text(encoding='utf-8-sig').strip()) if s]
    if not stems or len(stems) != len(set(stems)):
        raise ValueError(f'empty or duplicate stem list: {path}')
    if any(not re.fullmatch(r'(44b6|6bba)_[0-9a-f]{8}', s) for s in stems):
        raise ValueError(f'invalid movie stem in {path}')
    return stems


def extension_stems(pilot_stems):
    files = [ROOT/f'experiments/candidates/c016_division_scorer/stems_b0{i}.txt' for i in range(3)]
    more = [stem for path in files for stem in read_stems(path)]
    if len(more) != 75 or len(set(more)) != 75:
        raise ValueError('expected 75 distinct extension movies')
    if set(more).intersection(pilot_stems):
        raise ValueError('extension movies overlap pilot movies')
    missing = [s for s in more if not (ROOT/f'data/train/{s}.geff').exists()]
    if missing:
        raise ValueError(f'missing extension GT: {missing}')
    return more, {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def summary_for(path, config='as_configured', group='all'):
    summary = path.with_name(path.stem + '_summary.csv')
    return next(r for r in read_rows(summary) if r['config'] == config and r['group'] == group)


class Queue:
    def __init__(self, folder, max_hours):
        self.folder = folder.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.logs = self.folder/'logs'; self.logs.mkdir(exist_ok=True)
        self.path = self.folder/'status.json'
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(jobs={}, created=stamp())
        self.state.update(status='running', pid=os.getpid(), started=stamp(), current=None)
        self.state.pop('error', None)
        self.state.pop('ended', None)
        self.end = time.time() + 3600 * max_hours
        self.lock = self.folder/'queue.lock'
        # Only this queue owns this file; no external processes are interrupted.
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, str(os.getpid()).encode('ascii'))
        self.save()

    def save(self):
        self.state['updated'] = stamp()
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        temporary.replace(self.path)

    def run(self, name, args, required=True):
        command = [str(x) for x in args]
        old = self.state['jobs'].get(name)
        if old and old.get('returncode') == 0 and old.get('command') == command:
            return True
        if time.time() >= self.end:
            raise TimeoutError('Queue time budget reached before starting ' + name)
        record = dict(command=command, started=stamp(), status='running', log=str(self.logs/(name+'.log')))
        self.state['jobs'][name] = record; self.state['current'] = name; self.save()
        print(stamp(), name, flush=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        begin = time.time()
        with Path(record['log']).open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(command) + '\n'); stream.flush()
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                code = process.wait(timeout=8*3600)
            except subprocess.TimeoutExpired:
                # Kill only the subprocess tree owned by this timed-out job.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
                else:
                    process.kill()
                process.wait()
                code = 124
        record.update(returncode=code, ended=stamp(), seconds=time.time()-begin,
                      status='complete' if code == 0 else 'failed')
        self.save()
        if code and required:
            raise RuntimeError(f'{name} failed ({code}); see {record["log"]}')
        return code == 0

    def close(self, status, error=None):
        self.state.update(status=status, current=None, ended=stamp())
        if error:
            self.state['error'] = str(error)
        self.save()
        os.close(self.fd)
        self.lock.unlink()


def inference_command(repo, notebook, stems_file, label):
    return [sys.executable, '-u', INFERENCE, '--repo', repo, '--notebook', notebook,
            '--stems-file', stems_file, '--out', TC/'e2e', '--label', label,
            '--v1284-mode', 'candidate', '--v1284-head', HEAD, '--t4-fp32']


def replay_command(notebook, run, stems, variants, out, structured=False):
    cmd = [sys.executable, '-u', REPLAY, '--notebook', notebook, '--variants', variants,
           '--stems', ','.join(stems), '--pred-root', run/'predictions',
           '--lowdet-dir', run/'edge_cache', '--round-coords', '--out', out]
    if structured:
        cmd += ['--structured-trajectory', ASSETS]
    return cmd


def validate_structured_equivalence(harness_csv, notebook_csv):
    # Reuse original metric outputs; never substitute a different scorer.
    ref = {r['stem']: r for r in read_rows(harness_csv) if r['config'] == 'stabilized'}
    for r in read_rows(notebook_csv):
        for key in ('edge_tp', 'edge_fp', 'edge_fn', 'div_tp', 'div_fp', 'div_fn', 'nodes', 'edges'):
            if int(r[key]) != int(ref[r['stem']][key]):
                raise ValueError(f'C033 notebook/harness mismatch: {r["stem"]}/{key}')
        if abs(float(r['adjusted_edge_jaccard']) - float(ref[r['stem']]['adjusted_edge_jaccard'])) > 1e-12:
            raise ValueError('C033 notebook/harness score mismatch')


def write_results(folder, results, note=''):
    lines = ['# Local C032/C033 results', '',
             'Status is in status.json. These are train-embryo diagnostics, not hidden-test gains or submission approval.', '',
             '| Set | Arm | Total | Delta | Adj. edge delta | Division TP/FP/FN |',
             '|---|---|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['set']} | {r['arm']} | {r['score']:.6f} | {r['delta']:+.6f} | {r['edge_delta']:+.6f} | {r['divisions']} |")
    lines += ['', note, '', 'No upload, Kaggle push, submission, or final-pick change is performed by this queue.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'state/last_days_local_20260926')
    ap.add_argument('--max-hours', type=float, default=36)
    ap.add_argument('--extra-extension-arm', action='append', choices=TEMPORAL, default=[],
                    help='reviewed extra compute allocation; requires positive total/edge on both pilot sets')
    args = ap.parse_args()
    queue = Queue(args.out, args.max_hours)
    results = []
    write_results(queue.folder, results, 'Preparing executable controls and fixed experiment arms.')
    try:
        build_repo(SOURCE, TC/'tracking_repo')
        notebooks = {mode: build_notebook(mode, TC/mode) for mode in TEMPORAL}
        st_nb = build_structured('stabilized')
        build_structured('raw')
        tracked = ['src/build_temporal_context_candidate.py', 'src/run_kaggle_predict_local.py',
                   'src/eval_pp_variants_local.py', 'src/structured_trajectory_stage.py',
                   'src/build_structured_candidate.py', 'src/run_last_days_local.py',
                   'src/verify_temporal_context_local.py', str(BASE.relative_to(ROOT)),
                   str((SOURCE/'scripts/predict_unet_transformer.py').relative_to(ROOT)), str(HEAD.relative_to(ROOT))]
        hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in tracked}
        if queue.state.get('code_hashes') and queue.state['code_hashes'] != hashes:
            raise ValueError('Code changed since this queue started; use a new --out folder instead of reusing results')
        queue.state['code_hashes'] = hashes; queue.save()
        # Short real inference, not an AST-only guard. Completed when launched again
        # so equivalence always belongs to this exact built source.
        queue.run('temporal_execution_smoke', [sys.executable, '-u', ROOT/'src/verify_temporal_context_local.py'])
        smoke = json.loads((TC/'smoke.json').read_text(encoding='utf-8'))
        if smoke['status'] != 'passed':
            raise ValueError('temporal smoke failed')
        variants = queue.folder/'as_configured.json'
        variants.write_text(json.dumps({'as_configured': {}}), encoding='utf-8')
        st_variants = queue.folder/'structured_variants.json'
        st_variants.write_text(json.dumps({m: {'STRUCTURED_TRAJECTORY_MODE': m} for m in ('off','raw','stabilized')}), encoding='utf-8')
        report_dir = queue.folder/'replay'; report_dir.mkdir(exist_ok=True)
        # Discover integration failures immediately, before the long inference
        # queue. This uses existing C023 caches only as a stage smoke test.
        old_run = ROOT/'experiments/candidates/c012_v1284_head/e2e/head_x138_s075'
        smoke_stems = ['44b6_12dfb391', '6bba_05db0fb1']
        st_smoke = report_dir/'structured_existing_cache_smoke.csv'
        structured_ok = queue.run('structured_existing_cache_smoke',
            replay_command(BASE, old_run, smoke_stems, st_variants, st_smoke, True), required=False)
        if structured_ok:
            st_nb_smoke = report_dir/'structured_existing_notebook_smoke.csv'
            structured_ok = queue.run('structured_existing_notebook_smoke',
                replay_command(st_nb, old_run, smoke_stems, variants, st_nb_smoke), required=False)
            if structured_ok:
                validate_structured_equivalence(st_smoke, st_nb_smoke)
        queue.state['structured_execution_smoke_ok'] = structured_ok; queue.save()
        stems_by_set = {}
        run_by_set = {}
        arm_scores = {}
        for split, file in [('heldout12', SPLITS/'heldout_stems.txt'), ('confirm10', SPLITS/'confirm_stems.txt')]:
            stems = read_stems(file)
            stems_by_set[split] = stems
            control_label = 'control_fp32_' + split
            control_run = TC/'e2e'/control_label
            run_by_set[split] = control_run
            queue.run('infer_'+control_label, inference_command(SOURCE, BASE, file, control_label))
            if split == 'heldout12':
                queue.run('compare_control_with_existing_T4', [sys.executable, ROOT/'src/compare_kaggle_local_outputs.py',
                          '--kaggle', ROOT/'tmp/c023_output/edge_cache', '--local', control_run/'edge_cache'])
            control_csv = report_dir/(control_label+'.csv')
            queue.run('replay_'+control_label, replay_command(BASE, control_run, stems, variants, control_csv))
            baseline = summary_for(control_csv)
            arm_scores[(split, 'control')] = baseline
            results.append(dict(set=split, arm='control', score=float(baseline['proxy_score']), delta=0., edge_delta=0.,
                                divisions='/'.join(baseline[k] for k in ('div_tp','div_fp','div_fn'))))
            for mode in TEMPORAL:
                label = mode + '_' + split
                if not queue.run('infer_'+label, inference_command(TC/'tracking_repo', notebooks[mode], file, label), required=False):
                    continue
                csv_path = report_dir/(label+'.csv')
                if not queue.run('replay_'+label, replay_command(notebooks[mode], TC/'e2e'/label, stems, variants, csv_path), required=False):
                    continue
                s = summary_for(csv_path); arm_scores[(split, mode)] = s
                results.append(dict(set=split, arm=mode, score=float(s['proxy_score']),
                                    delta=float(s['proxy_score'])-float(baseline['proxy_score']),
                                    edge_delta=float(s['adjusted_edge_jaccard'])-float(baseline['adjusted_edge_jaccard']),
                                    divisions='/'.join(s[k] for k in ('div_tp','div_fp','div_fn'))))
                write_results(queue.folder, results, 'Running fixed heldout/confirmation arms; no settings are tuned.')
            st_csv = report_dir/('structured_'+split+'.csv')
            if structured_ok:
                structured_ok = queue.run('structured_'+split, replay_command(BASE, control_run, stems, st_variants, st_csv, True), required=False)
            if structured_ok:
                for mode in ('raw','stabilized'):
                    s = summary_for(st_csv, mode); arm_scores[(split, 'structured_'+mode)] = s
                    results.append(dict(set=split, arm='structured_'+mode, score=float(s['proxy_score']),
                                        delta=float(s['proxy_score'])-float(baseline['proxy_score']),
                                        edge_delta=float(s['adjusted_edge_jaccard'])-float(baseline['adjusted_edge_jaccard']),
                                        divisions='/'.join(s[k] for k in ('div_tp','div_fp','div_fn'))))
                off = summary_for(st_csv, 'off')
                if abs(float(off['proxy_score']) - float(baseline['proxy_score'])) > 1e-12:
                    raise ValueError('C033 off path does not reproduce control')
                if split == 'heldout12':
                    nb_csv = report_dir/'structured_notebook_smoke.csv'
                    structured_ok = queue.run('structured_notebook_smoke', replay_command(st_nb, control_run, stems[:2], variants, nb_csv), required=False)
                    if structured_ok:
                        validate_structured_equivalence(st_csv, nb_csv)
            write_results(queue.folder, results, 'Heldout and confirmation use exactly matched head/precision controls.')
        # Screening rule decides compute allocation only, never submission.
        advancing = []
        for arm in (*TEMPORAL, 'structured_raw', 'structured_stabilized'):
            if not all((s,arm) in arm_scores for s in ('heldout12','confirm10')):
                continue
            delta = lambda split,key: float(arm_scores[(split,arm)][key])-float(arm_scores[(split,'control')][key])
            if (delta('heldout12','proxy_score') >= .001 and delta('confirm10','proxy_score') > 0
                    and delta('heldout12','adjusted_edge_jaccard') > 0 and delta('confirm10','adjusted_edge_jaccard') > 0):
                advancing.append(arm)
        queue.state['automatic_advancing_to_75'] = list(advancing)
        for arm in args.extra_extension_arm:
            if not all((split, arm) in arm_scores and all(
                    float(arm_scores[(split, arm)][key]) > float(arm_scores[(split, 'control')][key])
                    for key in ('proxy_score', 'adjusted_edge_jaccard'))
                    for split in ('heldout12', 'confirm10')):
                raise ValueError(f'extra arm {arm} requires positive total and adjusted edge in both pilot sets')
            if arm not in advancing:
                advancing.append(arm)
        queue.state['reviewed_extra_extension_arms'] = args.extra_extension_arm
        queue.state['advancing_to_75'] = advancing; queue.save()
        if advancing:
            more, input_hashes = extension_stems({s for stems in stems_by_set.values() for s in stems})
            if queue.state.get('extension_input_hashes') and queue.state['extension_input_hashes'] != input_hashes:
                raise ValueError('extension stem files changed since the previous run')
            queue.state['extension_input_hashes'] = input_hashes
            queue.state['extension_stems'] = more; queue.save()
            write_results(queue.folder, results, 'Extending to 75 disjoint movies: '+', '.join(advancing)+
                          '. Automatic gate and reviewed extra arms are recorded separately in status.json.')
            extension_file = queue.folder/'extension75.txt'; extension_file.write_text('\n'.join(more)+'\n', encoding='utf-8')
            label = 'control_fp32_extension75'; control_run = TC/'e2e'/label
            queue.run('infer_'+label, inference_command(SOURCE, BASE, extension_file, label))
            control_csv = report_dir/(label+'.csv')
            queue.run('replay_'+label, replay_command(BASE, control_run, more, variants, control_csv))
            baseline = summary_for(control_csv)
            results.append(dict(set='extension75', arm='control', score=float(baseline['proxy_score']),delta=0.,edge_delta=0.,
                                divisions='/'.join(baseline[k] for k in ('div_tp','div_fp','div_fn'))))
            for arm in advancing:
                if arm in TEMPORAL:
                    label = arm+'_extension75'; path = report_dir/(label+'.csv')
                    if not queue.run('infer_'+label, inference_command(TC/'tracking_repo', notebooks[arm], extension_file, label), required=False):
                        continue
                    if not queue.run('replay_'+label, replay_command(notebooks[arm], TC/'e2e'/label, more, variants, path), required=False):
                        continue
                    s = summary_for(path)
                else:
                    path = report_dir/'structured_extension75.csv'
                    if not queue.run('structured_extension75', replay_command(BASE, control_run, more, st_variants, path, True), required=False):
                        continue
                    s = summary_for(path, arm.removeprefix('structured_'))
                results.append(dict(set='extension75',arm=arm,score=float(s['proxy_score']),
                                    delta=float(s['proxy_score'])-float(baseline['proxy_score']),
                                    edge_delta=float(s['adjusted_edge_jaccard'])-float(baseline['adjusted_edge_jaccard']),
                                    divisions='/'.join(s[k] for k in ('div_tp','div_fp','div_fn'))))
                write_results(queue.folder, results, 'Extension screening complete for this arm; review subgroup CSVs before any decision.')
        write_results(queue.folder, results, 'Queue finished. Extended arms: '+(', '.join(advancing) or 'none; fixed pilot directions did not pass the compute-allocation rule.')+
                      ' A new candidate still needs actual T4 visible-4 verification by the user before submission; no hidden-score gain is asserted.')
        queue.state['results'] = results
        failed = any(job.get('returncode', 0) != 0 for job in queue.state['jobs'].values())
        queue.close('complete_with_failures' if failed else 'complete')
    except Exception as exc:
        write_results(queue.folder, results, 'Queue stopped: '+str(exc)+'. See status.json and the current job log.')
        queue.close('failed', exc)
        raise


if __name__ == '__main__':
    main()
