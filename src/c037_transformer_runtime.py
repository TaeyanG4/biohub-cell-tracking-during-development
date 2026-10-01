"""C037 inference hooks: record primary transformer inputs or load fixed weights.

This same file is copied into the existing predictor's scripts directory.
Empty checkpoint/capture settings are exact no-ops. No labels enter inference.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
import torch


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def apply_primary(model):
    path = os.environ.get('BIOHUB_C037_CHECKPOINT', '').strip()
    alpha = float(os.environ.get('BIOHUB_C037_ALPHA', '1'))
    if not path or alpha == 0:
        return
    if not 0 < alpha <= 1:
        raise ValueError('C037 alpha must be in (0,1]')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    current = model.transformer.state_dict()
    if state_digest(current) != ckpt['base_transformer_sha256']:
        raise ValueError('C037 checkpoint does not match primary base transformer')
    assert set(current) == set(ckpt['state_dict'])
    mixed = {k: (1-alpha)*v + alpha*ckpt['state_dict'][k].to(device=v.device, dtype=v.dtype)
             for k, v in current.items()}
    model.transformer.load_state_dict(mixed, strict=True)
    model.eval()
    print('C037_PRIMARY', dict(path=path, alpha=alpha, train_embryo=ckpt['train_embryo'], step=ckpt['step']), flush=True)


def capture(ds_path, t_src, t_tgt, fsrc, ftgt, csrc, ctgt, psrc, ptgt, msrc, mtgt, logits):
    folder = os.environ.get('BIOHUB_C037_CAPTURE_DIR', '').strip()
    if not folder:
        return
    stem = Path(ds_path).name.removesuffix('.zarr')
    dest = Path(folder)/stem; dest.mkdir(parents=True, exist_ok=True)
    path = dest/f'{int(t_src):03d}_{int(t_tgt):03d}.npz'
    if path.exists():
        raise FileExistsError('C037 refuses to overwrite captured inputs: ' + str(path))
    assert bool(msrc.all()) and bool(mtgt.all()) and len(logits) == 1
    a = torch.cat([fsrc, psrc], dim=-1)[0].detach().cpu().numpy()
    b = torch.cat([ftgt, ptgt], dim=-1)[0].detach().cpu().numpy()
    rows = np.unique(np.linspace(0, len(a)-1, min(32, len(a))).astype(int))
    cols = np.unique(np.linspace(0, len(b)-1, min(32, len(b))).astype(int))
    probe = logits[0][torch.as_tensor(rows, device=logits.device)[:, None],
                      torch.as_tensor(cols, device=logits.device)[None]].detach().cpu().numpy()
    np.savez_compressed(path, feat_src=a, feat_tgt=b,
                        coords_src=csrc[0].detach().cpu().numpy(), coords_tgt=ctgt[0].detach().cpu().numpy(),
                        t_src=np.array(t_src), t_tgt=np.array(t_tgt),
                        probe_rows=rows, probe_cols=cols, probe_logits=probe)
