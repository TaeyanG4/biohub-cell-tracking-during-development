#!/usr/bin/env python3
"""C032: reuse the next window's detection map, with bounded lookahead storage.

No new inference/scoring implementation: extracts C023's exact encode block and
delays its existing detection/association loop by one window. Each window is
encoded once. Modes: off (original code), past_control (lookahead, unchanged
predictions), mean_det, future_det, mean_det_head (head features also averaged).
Creates local repos and self-contained candidate notebooks; never uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/candidates/c023_x138_head_stabilize_restore/biohub-c023-x138-head-stabilize-restore.ipynb"
SOURCE = ROOT / "tmp/c023_output/tracking_repo"
DEST = ROOT / "experiments/candidates/c032_temporal_context"
MODES = ("off", "past_control", "mean_det", "future_det", "mean_det_head")


def make_patches(source: str) -> list[tuple[str, str, str]]:
    start = source.index("        frame_indices = list(range(ws, ws + W))\n")
    stop = source.index("        # --- Detect cells in each frame", start)
    encode = source[start:stop]
    assert encode.count("        del imgs\n") == 1
    assert encode.count('if int(frame_indices[f]) not in seen_frames:') == 1
    helper_body = textwrap.dedent(encode).replace(
        'if int(frame_indices[f]) not in seen_frames:',
        'if int(frame_indices[f]) not in _tc_logged_frames:')
    helper = """
    # C032 uses the same encodes, in the same order, but delays node creation.
    _tc_mode = os.environ.get('BIOHUB_TEMPORAL_CONTEXT', 'off')
    if _tc_mode not in ('off', 'past_control', 'mean_det', 'future_det', 'mean_det_head'):
        raise ValueError('invalid BIOHUB_TEMPORAL_CONTEXT: ' + _tc_mode)
    if _tc_mode != 'off' and W != 2:
        raise ValueError('C032 requires window_size=2')
    _tc_logged_frames = set()

    def _tc_encode(ws):
""" + textwrap.indent(helper_body, "        ") + """
        _tc_logged_frames.update(frame_indices)
        return unet_out, det_logits, secondary_unet_out

    def _tc_packets():
        # Only the current and following packets persist between iterations.
        current = _tc_encode(window_starts[0])
        for index, ws in enumerate(window_starts):
            following = _tc_encode(window_starts[index + 1]) if index + 1 < len(window_starts) else None
            features, maps, secondary_features = current
            maps = list(maps)
            head_feature = None
            if following is not None and _tc_mode != 'past_control':
                if window_starts[index + 1] != ws + 1:
                    raise RuntimeError('C032 expected consecutive windows')
                maps[1] = following[1][0] if _tc_mode == 'future_det' else 0.5 * (maps[1] + following[1][0])
                if _tc_mode == 'mean_det_head':
                    head_feature = 0.5 * (features[:, 1] + following[0][:, 0])
            yield features, maps, secondary_features, head_feature
            current = following

    _tc_iterator = _tc_packets() if _tc_mode != 'off' else None
    print('TEMPORAL_CONTEXT', _tc_mode, 'windows=', len(window_starts), flush=True)
"""
    anchor = "    for ws in tqdm(\n        window_starts,\n"
    replacement = (
        "        _tc_head_feature = None\n"
        "        if _tc_iterator is not None:\n"
        "            frame_indices = list(range(ws, ws + W))\n"
        "            unet_out, det_logits, secondary_unet_out, _tc_head_feature = next(_tc_iterator)\n"
        "        else:\n" + textwrap.indent(encode, "    ")
    )
    head_old = "                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n"
    head_new = (
        "                _head_feature = _tc_head_feature if f_idx == 1 and _tc_head_feature is not None else unet_out[:, f_idx]\n"
        "                arr = _v1284_refine(ds_path, t, arr, _head_feature)\n"
        "                del _head_feature\n"
    )
    return [("lookahead helper", anchor, helper + "\n" + anchor),
            ("window dispatch", encode, replacement), ("head context", head_old, head_new)]


def patch_text(source: str) -> str:
    if "def _tc_packets():" in source:
        raise ValueError("source already has temporal-context patch")
    patches = make_patches(source)
    # Apply the encode replacement first: otherwise its text also occurs inside
    # the new helper. Every anchor is checked against the original source.
    for label, old, _ in patches:
        if source.count(old) != 1:
            raise ValueError(f"{label}: expected one anchor, found {source.count(old)}")
    for label, old, new in (patches[1], patches[2], patches[0]):
        source = source.replace(old, new, 1)
    compile(source, "temporal_context_predictor", "exec")
    return source


def build_repo(source: Path, dest: Path) -> Path:
    source, dest = source.resolve(), dest.resolve()
    dest.relative_to(ROOT)
    if dest == source or source.is_relative_to(dest):
        raise ValueError("source and output repo must be separate")
    for part in ("scripts", "src"):
        shutil.copytree(source / part, dest / part, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    script = dest / "scripts/predict_unet_transformer.py"
    script.write_text(patch_text((source / "scripts/predict_unet_transformer.py").read_text(encoding="utf-8")), encoding="utf-8")
    return dest


def build_notebook(mode: str, dest: Path, source: Path = SOURCE) -> Path:
    if mode not in MODES:
        raise ValueError(mode)
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    codes = [c for c in nb['cells'] if c['cell_type'] == 'code']
    code0 = ''.join(codes[0]['source']).replace("'''Biohub C023:", "'''Biohub C032:", 1)
    code0 += f"\nos.environ['BIOHUB_TEMPORAL_CONTEXT'] = {mode!r}\n"
    codes[0]['source'] = code0.splitlines(keepends=True)
    original = (source / 'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    patches = make_patches(original)
    ordered = [patches[i] for i in (1, 2, 0)]
    extra = "\n# C032: same checked replacements as build_temporal_context_candidate.py\n"
    extra += "_tc_source = _ps.read_text()\n_tc_patches = " + repr(ordered) + "\n"
    extra += "for _label, _old, _new in _tc_patches:\n    if _tc_source.count(_old) != 1:\n        raise RuntimeError(('C032 anchor mismatch', _label))\n    _tc_source = _tc_source.replace(_old, _new, 1)\n"
    extra += "compile(_tc_source, str(_ps), 'exec')\n_ps.write_text(_tc_source)\n"
    extra += f"os.environ['BIOHUB_TEMPORAL_CONTEXT'] = {mode!r}\n"
    code4 = ''.join(codes[4]['source'])
    anchor = '_ps.write_text(_trial_source)\n'
    assert code4.count(anchor) == 1
    codes[4]['source'] = code4.replace(anchor, anchor + extra, 1).splitlines(keepends=True)
    for i, c in enumerate(codes):
        compile(''.join(c['source']), f'cell{i}', 'exec')
    # Independently execute the embedded replacements, without notebook I/O.
    embedded = original
    for _, old, new in ordered:
        assert embedded.count(old) == 1
        embedded = embedded.replace(old, new, 1)
    assert embedded == patch_text(original)
    slug = 'biohub-c032-temporal-' + mode.replace('_', '-')
    nb.setdefault('metadata', {})['title'] = slug
    meta = json.loads((BASE.parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='taeyangg4/' + slug, title=slug, code_file=slug + '.ipynb')
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / meta['code_file']
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (dest / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=MODES, default='mean_det')
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=DEST)
    args = ap.parse_args()
    repo = build_repo(args.source, args.out / 'tracking_repo')
    nb = build_notebook(args.mode, args.out / args.mode, args.source)
    print(f'repo={repo}\nnotebook={nb}\nsha256={hashlib.sha256(nb.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
