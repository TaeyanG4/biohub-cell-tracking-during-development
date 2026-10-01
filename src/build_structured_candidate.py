#!/usr/bin/env python3
"""C033: extract already-pulled public structured code and embed it after C023 PP.
Never downloads, trains, uploads, pushes or submits. Original attribution saved.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/candidates/c023_x138_head_stabilize_restore/biohub-c023-x138-head-stabilize-restore.ipynb'
PUBLIC = ROOT/'state/notebook_radar/pulled/indarkarhana__biohub-structured-trajectory-candidate/biohub-structured-trajectory-candidate.ipynb'
ASSETS = ROOT/'artifacts/public_structured_trajectory'
DEST = ROOT/'experiments/candidates/c033_structured_trajectory'


def extract_assets():
    nb = json.loads(PUBLIC.read_text(encoding='utf-8'))
    overlays = None
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            for node in ast.walk(ast.parse(''.join(c['source']))):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_overlays' for t in node.targets):
                    overlays = ast.literal_eval(node.value)
    if overlays is None:
        raise ValueError('public source overlays not found')
    ASSETS.mkdir(parents=True, exist_ok=True)
    names = ('structured-trajectory.py', 'structured-trajectory-model.json', 'NOTICE.txt')
    for name in names:
        (ASSETS/name).write_text(overlays[name], encoding='utf-8')
    provenance = dict(source='https://www.kaggle.com/code/indarkarhana/biohub-structured-trajectory-candidate',
                      source_notebook_sha256=hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),
                      files={name: hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() for name in names},
                      note='Original embedded code/weights/NOTICE; no local fitting; NOTICE identifies Apache-2.0.')
    (ASSETS/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return overlays['structured-trajectory.py'], json.loads(overlays['structured-trajectory-model.json'])


def build(mode):
    source, model = extract_assets()
    stage = (ROOT/'src/structured_trajectory_stage.py').read_text(encoding='utf-8').replace('from __future__ import annotations\n', '')
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    codes = [c for c in nb['cells'] if c['cell_type'] == 'code']
    cell0 = ''.join(codes[0]['source']).replace("'''Biohub C023:", "'''Biohub C033:", 1)
    cell0 += f"\nos.environ['BIOHUB_STRUCTURED_TRAJECTORY_MODE'] = {mode!r}\n"
    codes[0]['source'] = cell0.splitlines(keepends=True)
    cell5 = ''.join(codes[5]['source'])
    anchor = '\nwrite_test_submission("base")\n'
    assert cell5.count(anchor) == 1
    payload = '\n# Public attribution: indarkarhana/biohub-structured-trajectory-candidate, Apache-2.0.\n'
    payload += '# Original NOTICE follows:\n' + ''.join('# ' + l + '\n' for l in (ASSETS/'NOTICE.txt').read_text(encoding='utf-8').splitlines())
    payload += stage + '\n'
    payload += 'install_structured_trajectory(globals(), ' + repr(source) + ', ' + repr(model) + ')\n'
    codes[5]['source'] = cell5.replace(anchor, payload + anchor, 1).splitlines(keepends=True)
    for i,c in enumerate(codes):
        compile(''.join(c['source']), f'cell{i}', 'exec')
    slug = 'biohub-c033-structured-' + mode
    nb.setdefault('metadata', {})['title'] = slug
    meta = json.loads((BASE.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='taeyangg4/'+slug, title=slug, code_file=slug+'.ipynb')
    folder = DEST/mode; folder.mkdir(parents=True, exist_ok=True)
    path = folder/meta['code_file']
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(path)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('off', 'raw', 'stabilized'), default='stabilized')
    build(ap.parse_args().mode)
