"""Two frozen C052 members as separate uniform production hypotheses, no refit."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import types
import numpy as np
import pandas as pd
import torch
import c053_fixed_division_transformer as original
from c047_hard_example_study import read, verify_hashes
from reid_probe_local import ROOT, sha, save_json, stamp

CONFIG = {'c067':'44b6', 'c068':'6bba'}
HISTORY = ROOT/'state/validation_audit_20260929'

def root(key): return ROOT/'experiments/candidates'/f'{key}_single_source_{CONFIG[key]}'

def install(key, out):
    group = CONFIG[key]
    source = original.STUDY/'models'/f'{group}_step600.pt'
    meta = read(source.with_suffix('.json'))
    assert sha(source) == meta['model_sha256']
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    assert checkpoint['train_embryo'] == group and checkpoint['step'] == 600
    assert all(s.startswith(group) for s in checkpoint['train_movies'])
    assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
    shutil.copy2(source, out/'transformer_mean600.pt')
    save_json(out/'mean_provenance.json', dict(source='Byte-identical C052 single source model; asset filename is compatibility only',
        source_checkpoint=str(source.relative_to(ROOT)), model_sha256=sha(source), train_embryo=group,
        step=600, parameter_averaging=False, no_new_training=True, no_prefix_router=True,
        interpretation='One fixed checkpoint for EVERY movie. Opposite embryo is component transfer; own embryo is fit-domain.'))

def module_for(key):
    source = Path(original.__file__).read_text(encoding='utf-8')
    source = source.replace('C053', key.upper()).replace('c053', key)
    start = source.index('def average_model(out):'); end = source.index('\n\ndef prepare(out):', start)
    source = source[:start]+'def average_model(out):\n    return _install(out)\n'+source[end:]
    assert source.count("'fixed_both_no_routing' in log") == 1
    source = source.replace("'fixed_both_no_routing' in log", repr(CONFIG[key])+" in log")
    module = types.ModuleType(key+'_single_source_portable')
    module.__file__ = str(Path(__file__).resolve())
    module.__dict__['_install'] = lambda out:install(key, out)
    exec(compile(source, str(Path(__file__))+'::'+key, 'exec'), module.__dict__)
    module.DEST = root(key)
    module.SLUG = f'biohub-{key}-fixed-single-{CONFIG[key]}'
    return module, source

def audit_adapter(stems, expected_hash):
    return '''
# Research-only input adapter. Inference/postprocessing cells are unchanged.
# This train-only notebook must never be submitted to the competition.
_research_manifest_path = _fixed_asset('transformer_mean600.pt').parent/'research_inputs.json'
assert hashlib.sha256(_research_manifest_path.read_bytes()).hexdigest() == '''+repr(expected_hash)+'''
_research_manifest = json.loads(_research_manifest_path.read_text())
_research_stems = '''+repr(stems)+'''
assert _research_manifest['stems'] == _research_stems
_research_train = COMP_DIR/'train'
for _name, _want in _research_manifest['hashes'].items():
    _file = _research_train/_name
    _digest = hashlib.sha256()
    with _file.open('rb') as _stream:
        for _block in iter(lambda: _stream.read(8<<20), b''):
            _digest.update(_block)
    assert _digest.hexdigest() == _want, _name
TEST_DIR = WORKING_DIR/'research_inputs'
TEST_DIR.mkdir(exist_ok=False)
for _stem in _research_stems:
    (TEST_DIR/(_stem+'.zarr')).symlink_to(_research_train/(_stem+'.zarr'), target_is_directory=True)
(WORKING_DIR/'research_input_receipt.json').write_text(json.dumps(dict(
    status='passed', files=len(_research_manifest['hashes']), stems=_research_stems,
    manifest_sha256='''+repr(expected_hash)+''', no_gt_used_at_runtime=True), indent=2))
print('RESEARCH_INPUTS_VERIFIED', len(_research_manifest['hashes']), 'files', len(_research_stems), 'movies', flush=True)
'''

def archive_cell():
    return '''
import hashlib, json, zipfile
from pathlib import Path
import pandas as pd
_research_work = Path('/kaggle/working')
_research_stats = pd.read_csv(RUN_STATS_PATH)
assert len(_research_stats) == len(_research_stems)
assert set(_research_stats['dataset']) == set(_research_stems)
assert (_research_stats['repair_fallback'] == 0).all()
assert (_research_stats['deadline_degraded'] == 0).all()
_research_files = [Path(SUBMISSION_PATH), Path(RUN_STATS_PATH), _research_work/'research_input_receipt.json']
for _folder in [_research_work/'edge_cache', REPO_DIR/'predictions', REPO_DIR/'scripts']:
    _research_files += [p for p in _folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
_research_files += list(_research_work.glob('retention_guard_*.jsonl'))
_research_hashes = {}
for _file in sorted(set(_research_files)):
    _digest = hashlib.sha256()
    with _file.open('rb') as _stream:
        for _block in iter(lambda:_stream.read(8<<20),b''): _digest.update(_block)
    _research_hashes[str(_file.relative_to(_research_work))] = _digest.hexdigest()
(_research_work/'research_output_hashes.json').write_text(json.dumps(_research_hashes,indent=2))
with zipfile.ZipFile(_research_work/'research_artifacts.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as _zip:
    for _name in _research_hashes: _zip.write(_research_work/_name,_name)
    _zip.write(_research_work/'research_output_hashes.json','research_output_hashes.json')
print('RESEARCH_ARCHIVE_COMPLETE', len(_research_hashes), 'files', flush=True)
'''

def prepare(key):
    out = root(key); group = CONFIG[key]
    assert not (out/'plan.json').exists(), 'Registered candidate is immutable'
    module, derived = module_for(key)
    out.mkdir(parents=True, exist_ok=True)
    (out/'derived_portable.py').write_text(derived, encoding='utf-8')
    module.prepare(out)
    historical = pd.read_csv(HISTORY/'actual_per_movie.csv')
    control = historical[historical.candidate=='C023']
    assert len(control)==97 and control.stem.is_unique
    new_stems = sorted(control[control.embryo==group].stem)
    reused_stems = sorted(control[control.embryo!=group].stem)
    smoke = sorted(s for s in module.STEM_SETS['vis4'] if not s.startswith(group))
    assert len(smoke)==2 and set(smoke)<=set(reused_stems)
    remote_stems = new_stems+smoke
    # Reuse already pinned original raw data hashes; no arbitrary new train sample selection.
    plan = read(out/'plan.json'); inputs={}; inherited=plan['hashes']
    for stem in remote_stems:
        folder=ROOT/'data/train'/(stem+'.zarr')
        for path in folder.rglob('*'):
            if not path.is_file():continue
            key_path=str(path.relative_to(ROOT)); digest=sha(path)
            assert inherited.get(key_path)==digest, ('unbound original data',key_path)
            inputs[str(path.relative_to(ROOT/'data/train')).replace('\\','/')]=digest
    manifest=out/'dataset/research_inputs.json'
    save_json(manifest,dict(stems=remote_stems,hashes=inputs,source_group=group,role='validation inputs only; no GT labels'))
    clean=read(module.nbpath(out)); audit=copy.deepcopy(clean)
    adapter=audit_adapter(remote_stems,sha(manifest))
    # hashlib is imported in the preceding asset block under a private alias.
    adapter='\nimport hashlib\n'+adapter
    audit['cells'][2]['source']=(''.join(audit['cells'][2]['source'])+adapter).splitlines(keepends=True)
    audit['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=archive_cell().splitlines(keepends=True)))
    audit_slug=f'biohub-{key}-single-{group}-train-audit'
    audit_folder=out/'remote_audit';audit_folder.mkdir()
    audit['metadata']['title']=audit_slug
    (audit_folder/(audit_slug+'.ipynb')).write_text(json.dumps(audit,indent=1)+'\n',encoding='utf-8')
    meta=read(out/'kernel-metadata.json');meta.update(id='taeyangg4/'+audit_slug,title=audit_slug,code_file=audit_slug+'.ipynb')
    save_json(audit_folder/'kernel-metadata.json',meta)
    for i,cell in enumerate(audit['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'{key}:auditcell{i}','exec')
    assert all(audit['cells'][i]['source']==clean['cells'][i]['source'] for i in range(len(clean['cells'])) if i!=2)
    assert ''.join(audit['cells'][2]['source'])==''.join(clean['cells'][2]['source'])+adapter
    assert sha(out/'dataset/transformer_mean600.pt')==read(out/'mean_provenance.json')['model_sha256']
    proof=dict(status='passed',candidate=key,source_group=group,model_sha256=plan['model_sha256'],
        clean_notebook_sha256=sha(module.nbpath(out)),research_notebook_sha256=sha(audit_folder/meta['code_file']),
        identical_inference_cell=True,identical_postprocess_writer_cell=True,input_adapter_only=True,
        research_only_never_submit=True,new_movies=len(new_stems),reused_movies=len(reused_stems),two_exact_smoke_movies=smoke)
    save_json(out/'prepare_proof.json',proof)
    (out/'README.md').write_text(f'''# {key.upper()} single fixed C052 source model

Use the unchanged {group} step600 checkpoint for every movie, without averaging,
retraining, thresholds or per-movie routing. Both C052 models transferred better
than their mean on the opposite domain; the user's latest instruction favors
using up to five worthwhile distinct exploratory submissions by midnight.

{len(reused_stems)} opposite-domain movies already have exact-model C052 results.
Run {len(new_stems)} missing source-domain movies plus two opposite-domain
reproduction controls privately on T4. The train audit differs from production
only in selected raw-input directory and appended receipts/archive. Never
submit the research notebook. Require all97 actual organizer evaluation,
source/inputs/cache/ILP/finalgraph parity, local portable12 and originalwriter4,
then clean production notebook exact-version T4 verification before submission.

Own-domain movies are fit-domain. Opposite domain supplies component transfer;
public frozen detection models were exposed to both embryos. No claim of new
independent validation. Both domains and graph tradeoffs must be reported.
Preserve C023/C0240.954 anchors; no old-candidate resubmission/score polling.
''',encoding='utf-8')
    paths={Path(__file__),out/'README.md',out/'prepare_proof.json',manifest,HISTORY/'actual_per_movie.csv',HISTORY/'independent_verification.json'}
    paths.update(p for p in audit_folder.iterdir() if p.is_file())
    plan['hashes'].update({str(path.relative_to(ROOT)):sha(path) for path in paths})
    plan.update(candidate=key,source_group=group,new_stems=new_stems,reused_stems=reused_stems,
        remote_stems=remote_stems,smoke_stems=smoke,research_ref=meta['id'],research_only_never_submit=True,
        parameter_averaging=False,total_jobs=None,execution='remote train audit then local merged97 and portable review; original run() is not used',
        policy='One immutable C052 member for every movie. Prior opposite-domain outputs reused only after actual source/cache/ILP/finalgraph parity.',
        estimated_remote_minutes=90 if key=='c067' else 180,
        estimate_basis='T4 visible4 about12-15min incl setup; twoGPU shard runtime for29/72movies plus source/input hashing and queue margin',
        completion_target='2026-09-30T00:00:00+09:00')
    save_json(out/'plan.json',plan)
    print(json.dumps(dict(prepared=True,**proof,inputs=len(plan['hashes']))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare']);parser.add_argument('--candidate',choices=list(CONFIG),required=True)
    args=parser.parse_args();prepare(args.candidate)
