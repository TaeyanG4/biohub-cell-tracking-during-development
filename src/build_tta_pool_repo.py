#!/usr/bin/env python3
"""Build a local tracking_repo with probability-level edge TTA pooling (idea 9 of reports/untried_ideas_20260924.md).

x138 averages the UNet feature maps of 8 D4 views (identity, flip-x, flip-y, flip-xy, rot90, rot270, transpose,
anti-transpose) and predicts the edge logits once. This patch keeps the 8 per-view feature maps (67 MB each at
64x64x64), predicts the forward edge logits per view and pools the per-target parent distributions:

  BIOHUB_EDGE_TTA_POOL   off (default, byte-identical behaviour) | js_log_pool (yusuketogashi "Biohub 154":
                         per-target view weights 1/(1 + JS/median JS) to the view consensus, log opinion pool)
                         | mean_log_pool | mean_prob
  BIOHUB_EDGE_TTA_POOL_VIEWS         8 (all D4) | 4 (identity, flip-x, flip-y, transpose)
  BIOHUB_EDGE_TTA_POOL_INCLUDE_MEAN  1 adds x138's averaged-feature prediction as one more pool member
  BIOHUB_EDGE_TTA_POOL_REF           mean (default: rescale pooled logits to the averaged-feature logits' centre/scale,
                                     which x138's thresholds were tuned on) | identity (yusuketogashi's choice)

The pooled logits replace x138's forward logits; the reverse pass, the secondary-seed fusion, the candidate
threshold and the ILP are untouched. Usage:

    python src/build_tta_pool_repo.py [--source tmp/c022_output/tracking_repo] [--dest tmp/c024_tta_pool/tracking_repo]
    python src/run_kaggle_predict_local.py --repo tmp/c024_tta_pool/tracking_repo ... --env BIOHUB_EDGE_TTA_POOL=js_log_pool
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

POOL_BLOCK = """
            if _POOL_MODE != 'off':
                if _pool_feats is None or len(_pool_feats) != 8:
                    raise RuntimeError('EDGE_TTA_POOL: expected 8 view feature maps, got %s'
                                       % (None if _pool_feats is None else len(_pool_feats)))
                _sel = list(range(8)) if _POOL_VIEWS >= 8 else [0, 1, 2, 6]
                _view_logits = []
                for _vi in _sel:
                    _vf = _pool_feats[_vi]
                    _fs = model._index_features(_vf[:, f_idx], p_coords_src, p_mask_src)
                    _ft = model._index_features(_vf[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                    _view_logits.append(model.predict_edges(
                        _fs, _ft, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                        p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
                    ).float())
                if _POOL_INCLUDE_MEAN:
                    _view_logits.append(edge_logits_pair.float())
                _stack = torch.stack(_view_logits, dim=0)  # (V, 1, n_src, n_tgt)
                _prob_stack = torch.softmax(_stack, dim=2).clamp_min(1e-8)  # parent distribution per target, per view
                if _POOL_MODE == 'js_log_pool':
                    _cons = _prob_stack.mean(dim=0, keepdim=True).clamp_min(1e-8)
                    _mix = 0.5 * (_prob_stack + _cons)
                    _js = 0.5 * ((_prob_stack * (torch.log(_prob_stack) - torch.log(_mix))).sum(dim=2)
                                 + (_cons * (torch.log(_cons) - torch.log(_mix))).sum(dim=2))  # (V, 1, n_tgt)
                    _js_scale = torch.median(_js, dim=0).values.clamp_min(1e-6)
                    _w = 1.0 / (1.0 + _js / _js_scale.unsqueeze(0))
                    _w = _w / _w.sum(dim=0, keepdim=True).clamp_min(1e-8)
                    _pooled_log = (_w.unsqueeze(2) * torch.log(_prob_stack)).sum(dim=0)
                elif _POOL_MODE == 'mean_log_pool':
                    _pooled_log = torch.log(_prob_stack).mean(dim=0)
                else:
                    _pooled_log = torch.log(_prob_stack.mean(dim=0).clamp_min(1e-8))
                _pooled_logits = torch.log(torch.softmax(_pooled_log, dim=1).clamp_min(1e-8))
                _ref = edge_logits_pair.float() if _POOL_REF == 'mean' else _view_logits[0]
                _ref_center = _ref.mean(dim=1, keepdim=True)
                _ref_scale = _ref.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                _p_center = _pooled_logits.mean(dim=1, keepdim=True)
                _p_scale = _pooled_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                _ratio = (_ref_scale / _p_scale).clamp(0.5, 2.0)
                _new_logits = ((_pooled_logits - _p_center) * _ratio + _ref_center).to(edge_logits_pair.dtype)
                _POOL_STATS['pairs'] += 1
                _POOL_STATS['delta'] += float((_new_logits - edge_logits_pair).abs().mean())
                edge_logits_pair = _new_logits
                del _view_logits, _stack, _prob_stack, _pooled_log, _pooled_logits, _new_logits
"""

FORWARD_ANCHOR = (
    "            edge_logits_pair = model.predict_edges(\n"
    "                unet_feat_src, unet_feat_tgt,\n"
    "                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n"
    "                p_pos_src, p_pos_tgt,\n"
    "                p_mask_src, p_mask_tgt,\n"
    "            )  # (1, n_src, n_tgt)\n"
    "\n"
    "            _bidirectional_weight = float(\n"
)

PATCHES = [
    ("module settings",
     "_LOWDET: list = []\n",
     "_LOWDET: list = []\n"
     "_POOL_MODE = os.environ.get('BIOHUB_EDGE_TTA_POOL', 'off')\n"
     "_POOL_VIEWS = int(os.environ.get('BIOHUB_EDGE_TTA_POOL_VIEWS', '8'))\n"
     "_POOL_INCLUDE_MEAN = os.environ.get('BIOHUB_EDGE_TTA_POOL_INCLUDE_MEAN', '0') != '0'\n"
     "_POOL_REF = os.environ.get('BIOHUB_EDGE_TTA_POOL_REF', 'mean')\n"
     "_POOL_STATS = {'pairs': 0, 'delta': 0.0}\n"
     "if _POOL_MODE not in ('off', 'js_log_pool', 'mean_log_pool', 'mean_prob'):\n"
     "    raise RuntimeError('BIOHUB_EDGE_TTA_POOL must be off | js_log_pool | mean_log_pool | mean_prob')\n"),
    ("det tta start",
     "        if cfg.det_tta:\n"
     "            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'\n"
     "            _unet_acc = unet_out.clone() if _edge_tta else None\n"
     "            _nv = 1\n",
     "        _pool_feats = [unet_out.clone()] if _POOL_MODE != 'off' else None\n"
     "        if _POOL_MODE != 'off' and not cfg.det_tta:\n"
     "            raise RuntimeError('BIOHUB_EDGE_TTA_POOL needs det_tta')\n"
     "        if cfg.det_tta:\n"
     "            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'\n"
     "            _unet_acc = unet_out.clone() if _edge_tta else None\n"
     "            _nv = 1\n"),
    ("flip views",
     "                if _edge_tta:\n                    _unet_acc = _unet_acc + _u_flip.flip(dims)\n",
     "                if _edge_tta:\n                    _unet_acc = _unet_acc + _u_flip.flip(dims)\n"
     "                if _pool_feats is not None:\n                    _pool_feats.append(_u_flip.flip(dims).contiguous())\n"),
    ("rot views",
     "                if _edge_tta:\n                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims=(-2, -1))\n",
     "                if _edge_tta:\n                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims=(-2, -1))\n"
     "                if _pool_feats is not None:\n                    _pool_feats.append(torch.rot90(_u_rot, -_k, dims=(-2, -1)).contiguous())\n"),
    ("transpose view",
     "            if _edge_tta:\n                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)\n",
     "            if _edge_tta:\n                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)\n"
     "            if _pool_feats is not None:\n                _pool_feats.append(_u_t.transpose(-1, -2).contiguous())\n"),
    ("anti-transpose view",
     "            if _edge_tta:\n                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))\n",
     "            if _edge_tta:\n                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))\n"
     "            if _pool_feats is not None:\n"
     "                _pool_feats.append(torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1)).contiguous())\n"),
    ("forward pooling",
     FORWARD_ANCHOR,
     FORWARD_ANCHOR.replace("\n\n            _bidirectional_weight = float(\n", POOL_BLOCK + "\n            _bidirectional_weight = float(\n")),
    ("window cleanup",
     "        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n",
     "        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n"
     "        if _pool_feats is not None:\n            del _pool_feats\n"),
    ("stats print",
     "    return coords, all_edges\n",
     "    if _POOL_MODE != 'off':\n"
     "        print('EDGE_TTA_POOL', _POOL_MODE, 'views=', _POOL_VIEWS, 'include_mean=', int(_POOL_INCLUDE_MEAN), 'ref=', _POOL_REF,\n"
     "              'pairs=', _POOL_STATS['pairs'],\n"
     "              'mean_abs_logit_delta=', round(_POOL_STATS['delta'] / max(_POOL_STATS['pairs'], 1), 6), flush=True)\n"
     "        _POOL_STATS['pairs'] = 0\n"
     "        _POOL_STATS['delta'] = 0.0\n"
     "    return coords, all_edges\n"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "tmp" / "c022_output" / "tracking_repo")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "tmp" / "c024_tta_pool" / "tracking_repo")
    args = ap.parse_args()
    if args.dest.exists():
        shutil.rmtree(args.dest)
    for part in ("scripts", "src"):
        shutil.copytree(args.source / part, args.dest / part, ignore=shutil.ignore_patterns("__pycache__"))
    script = args.dest / "scripts" / "predict_unet_transformer.py"
    text = script.read_text(encoding="utf-8")
    for label, old, new in PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f"{label}: expected one anchor, found {text.count(old)}")
        text = text.replace(old, new, 1)
    script.write_text(text, encoding="utf-8")
    compile(text, str(script), "exec")
    print(f"wrote {script} ({len(PATCHES)} patches)")


if __name__ == "__main__":
    main()
