#!/usr/bin/env python3
"""Execute C032's original/off/lookahead controls and all modes on real short movies.

Uses the existing predictor directly, including dual-seed D4 and the x138 head.
This is an execution/equivalence smoke check, not a new evaluation harness.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=ROOT/'experiments/candidates/c032_temporal_context/tracking_repo')
    ap.add_argument('--out', type=Path, default=ROOT/'experiments/candidates/c032_temporal_context/smoke.json')
    args = ap.parse_args()
    from build_temporal_context_candidate import BASE, SOURCE
    from v1284_capture_local import notebook_env, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS
    import numpy as np
    import torch
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(notebook_env(BASE))
    os.environ.update(V1284_MODE='candidate', V1284_HEAD=str(ROOT/'artifacts/anvithpothula_v1284_head_s075/v1284_head.pt'),
                      BIOHUB_CACHE_DIR='', BIOHUB_LOCAL_WORKING=str(args.out.parent))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    sys.path[:0] = [str(args.repo/'scripts'), str(args.repo/'src')]
    import predict_unet_transformer as pu
    # Only log paths differ from the original source; scientific code is intact.
    original = (SOURCE/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    original = original.replace('Path("/kaggle/working")', "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    spec = importlib.util.spec_from_loader('original_c023_smoke', loader=None)
    base = importlib.util.module_from_spec(spec)
    base.__file__ = str(SOURCE/'scripts/predict_unet_transformer.py')
    sys.modules[spec.name] = base
    exec(compile(original, base.__file__, 'exec'), base.__dict__)
    # Localize the patched source's diagnostic output paths without rebuilding.
    patched = (args.repo/'scripts/predict_unet_transformer.py').read_text(encoding='utf-8')
    patched = patched.replace('Path("/kaggle/working")', "Path(os.environ['BIOHUB_LOCAL_WORKING'])")
    exec(compile(patched, pu.__file__, 'exec'), pu.__dict__)
    device = torch.device('cuda')
    model, window, downsample = pu.load_model(PRIMARY_WEIGHTS, device)
    secondary, w2, d2 = pu.load_model(SECONDARY_WEIGHTS, device)
    assert window == w2 == 2 and downsample == d2
    pu.UNetNodeTransformer._index_features = pu._v1284_index
    cfg = pu.PredictConfig(det_threshold=float(os.environ['BIOHUB_DET_THRESHOLD']))
    cfg.threshold = float(os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'])
    kwargs = dict(cfg=cfg, window_size=window, max_frames=4, downsample=downsample,
                  secondary_model=secondary, secondary_edge_weight=float(os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT']),
                  secondary_detection_weight=float(os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT']),
                  secondary_link_mode=os.environ['BIOHUB_SECONDARY_LINK_MODE'],
                  secondary_mix_temperature=float(os.environ.get('BIOHUB_SECONDARY_MIX_TEMPERATURE', '1')),
                  secondary_low_margin_max=float(os.environ.get('BIOHUB_SECONDARY_LOW_MARGIN_MAX', '.35')))
    rows = []
    with args.out.with_suffix('.log').open('w', encoding='utf-8') as log:
        for stem in ('44b6_12dfb391', '6bba_05db0fb1'):
            ref = None
            for mode in ('original', 'off', 'past_control', 'mean_det', 'future_det', 'mean_det_head'):
                engine = base if mode == 'original' else pu
                os.environ['BIOHUB_TEMPORAL_CONTEXT'] = mode if mode != 'original' else 'off'
                engine._LOWDET.clear(); engine._CACHE_EDGES.clear()
                with contextlib.redirect_stdout(log):
                    coords, edges = engine.predict_video(model, ROOT/'data/train'/stem, device, **kwargs)
                coords = np.asarray(coords); edges = np.asarray(edges)
                assert np.isfinite(coords).all() and np.isfinite(edges).all()
                assert set(coords[:, 0].astype(int)) == {0, 1, 2, 3}
                assert all(coords[int(t), 0] == coords[int(s), 0] + 1 for s,t,_,_ in edges)
                if mode == 'original':
                    ref = (coords.copy(), edges.copy(), [(a.copy(), b.copy()) for a,b in engine._LOWDET])
                equal = (coords.shape == ref[0].shape and edges.shape == ref[1].shape
                         and np.array_equal(coords, ref[0]) and np.array_equal(edges, ref[1]))
                low_equal = len(engine._LOWDET) == len(ref[2]) and all(
                    np.array_equal(a, c) and np.array_equal(b, d)
                    for (a,b),(c,d) in zip(engine._LOWDET, ref[2]))
                if mode in ('off', 'past_control'):
                    assert equal and low_equal, f'{stem}/{mode} differs from original C023'
                rows.append(dict(stem=stem, mode=mode, nodes=len(coords), edges=len(edges),
                                 exactly_equals_original=equal, lowdet_exactly_equals_original=low_equal))
                print(rows[-1], flush=True)
    args.out.write_text(json.dumps(dict(status='passed', rows=rows, fp32=True, math_sdpa=True,
                                        note='Four-frame real inference only; no ILP or official score.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
