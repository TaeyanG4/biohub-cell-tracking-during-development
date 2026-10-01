"""C057 conditional parent CE on known target links, with identity ambiguity masked.

Unmatched cells are never declared false detections. For a target with an
annotated unique parent and an existing matched positive detection, only
unmatched alternative sources outside that parent's fixed 7 um identity gate
join the conditional parent competition. Near-parent unmatched alternatives
stay masked. Original targets, known competitors, weights and sampling remain.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IDENTITY_RADIUS_UM = 7.0
SCALE_UM = np.array([1.625, 0.40625, 0.40625], np.float64)


def build_labels(packet, original_lab, gt_nodes, gt_edges, ns):
    """Reproduce C037 matching/labels exactly, then append conditional masks.

    ``ns`` is the existing official replay namespace. No alternate matcher or
    scorer is implemented here. Return (numpy label dictionary, audit metadata).
    ``allowed_sources`` is target-major [n_targets, n_sources].
    """
    lab = {k: np.array(v, copy=True) for k, v in original_lab.items()}
    source = np.asarray(packet['coords_src'])
    target = np.asarray(packet['coords_tgt'])
    ta, tb = int(packet['t_src']), int(packet['t_tgt'])
    assert tb == ta + 1, 'C057 expects original forward consecutive packets'
    plain = {i: (ta, *(max(0, int(round(float(v)))) for v in xyz))
             for i, xyz in enumerate(source)}
    plain.update({len(source)+i: (tb, *(max(0, int(round(float(v)))) for v in xyz))
                  for i, xyz in enumerate(target)})
    mapping, _ = ns['match_nodes_bipartite'](plain, gt_nodes, max_dist=IDENTITY_RADIUS_UM)
    known = np.array(sorted(i for i in mapping if i < len(source)), np.int64)
    parent_to_local = {mapping[int(i)]: j for j, i in enumerate(known)}
    parents, children = {}, {}
    for a, b in gt_edges:
        parents.setdefault(b, []).append(a)
        children.setdefault(a, []).append(b)
    targets, labels, weights = [], [], []
    for j in range(len(target)):
        par = parents.get(mapping.get(len(source)+j), [])
        if len(par) == 1 and par[0] in parent_to_local:
            targets.append(j)
            labels.append(parent_to_local[par[0]])
            weights.append(2. if len(children.get(par[0], [])) == 2 else 1.)
    expected = dict(known_src=known, targets=np.array(targets, np.int64),
                    parent_index=np.array(labels, np.int64), weight=np.array(weights, np.float32))
    for key, value in expected.items():
        assert key in lab and np.array_equal(lab[key], value), ('original C037 label drift', key)
    assert len(known) >= 2 and len(targets), 'Original C052 eligibility must be retained'
    parent_source = known[lab['parent_index']]
    rounded = np.array([plain[i][1:] for i in range(len(source))], np.float64)
    gt_parent_xyz = np.array([gt_nodes[mapping[int(i)]][1:] for i in parent_source], np.float64)
    distance = np.linalg.norm((rounded[None]-gt_parent_xyz[:, None])*SCALE_UM, axis=-1)
    known_mask = np.zeros(len(source), bool)
    known_mask[known] = True
    allowed = known_mask[None] | (distance > IDENTITY_RADIUS_UM)
    ii = np.arange(len(targets))
    assert np.all(distance[ii, parent_source] <= IDENTITY_RADIUS_UM)
    assert np.all(allowed[ii, parent_source])
    # Keep every original known-source competitor regardless of this extra gate.
    assert np.all(allowed[:, known])
    assert not np.any(allowed & ~known_mask[None] & (distance <= IDENTITY_RADIUS_UM))
    lab.update(parent_source=parent_source.astype(np.int64), allowed_sources=allowed)
    metadata = dict(targets=len(targets), source_nodes=len(source), known_sources=len(known),
                    conditional_edge_negatives=int((allowed & ~known_mask[None]).sum()),
                    masked_identity_ambiguous=int((~allowed).sum()),
                    targets_with_conditional_negatives=int((allowed & ~known_mask[None]).any(axis=1).sum()),
                    positive_match_max_um=float(distance[ii, parent_source].max()),
                    identity_radius_um=IDENTITY_RADIUS_UM,
                    coordinate_convention='unchanged C037 integer emitted source coordinates',
                    unmatched_targets_labelled=0, unmatched_cells_labelled_as_false=0)
    return lab, metadata


def supervised_loss(current, lab):
    """Weighted target-conditional CE; masked logits have exactly zero gradient."""
    assert current.ndim == 2
    targets = torch.as_tensor(lab['targets'], device=current.device, dtype=torch.long)
    parent = torch.as_tensor(lab['parent_source'], device=current.device, dtype=torch.long)
    allowed = torch.as_tensor(lab['allowed_sources'], device=current.device, dtype=torch.bool)
    weights = torch.as_tensor(lab['weight'], device=current.device, dtype=current.dtype)
    assert allowed.shape == (len(targets), current.shape[0])
    assert torch.all(allowed[torch.arange(len(parent), device=current.device), parent])
    logits = current[:, targets].T.masked_fill(~allowed, -torch.inf)
    ce = F.cross_entropy(logits, parent, reduction='none')
    return (ce * weights).sum() / weights.sum()


def numerical_smoke(current, lab):
    """Real-packet objective contract checks without changing a model parameter."""
    device = current.device
    targets = torch.as_tensor(lab['targets'], device=device, dtype=torch.long)
    known = torch.as_tensor(lab['known_src'], device=device, dtype=torch.long)
    old_parent = torch.as_tensor(lab['parent_index'], device=device, dtype=torch.long)
    weights = torch.as_tensor(lab['weight'], device=device, dtype=current.dtype)
    original = current.detach().clone().requires_grad_(True)
    ce = F.cross_entropy(original[known][:, targets].T, old_parent, reduction='none')
    old_loss = (ce * weights).sum()/weights.sum()
    old_loss.backward()
    narrowed = dict(lab)
    only_known = torch.zeros((len(targets), current.shape[0]), device=device, dtype=torch.bool)
    only_known[:, known] = True
    narrowed['allowed_sources'] = only_known
    control = current.detach().clone().requires_grad_(True)
    control_loss = supervised_loss(control, narrowed)
    control_loss.backward()
    assert torch.allclose(old_loss, control_loss, rtol=0, atol=2e-6), 'Known-only loss mismatch'
    assert torch.allclose(original.grad, control.grad, rtol=0, atol=2e-6), 'Known-only gradient mismatch'
    actual = current.detach().clone().requires_grad_(True)
    actual_loss = supervised_loss(actual, lab)
    actual_loss.backward()
    allowed = torch.as_tensor(lab['allowed_sources'], device=device, dtype=torch.bool)
    grads = actual.grad[:, targets].T
    assert not torch.count_nonzero(grads[~allowed]), 'Masked competitors received gradient'
    other = torch.ones(current.shape[1], device=device, dtype=torch.bool)
    other[targets] = False
    assert not torch.count_nonzero(actual.grad[:, other]), 'Unlabelled targets received supervised gradient'
    added = allowed & ~only_known
    positive = torch.as_tensor(lab['parent_source'], device=device, dtype=torch.long)
    assert torch.all(grads[added] >= 0), 'Conditional negative gradient has wrong sign'
    assert torch.all(grads[torch.arange(len(targets), device=device), positive] <= 0)
    active_added = int(torch.count_nonzero(grads[added]))
    if torch.any(added):
        assert active_added > 0, 'All added conditional competitors had zero gradient'
    return dict(status='passed', known_only_loss_delta=float((old_loss-control_loss).abs()),
                known_only_gradient_max_delta=float((original.grad-control.grad).abs().max()),
                masked_gradient_nonzero=int(torch.count_nonzero(grads[~allowed])),
                unlabelled_target_gradient_nonzero=int(torch.count_nonzero(actual.grad[:, other])),
                conditional_negative_gradients_active=active_added,
                original_loss=float(old_loss.detach()), conditional_loss=float(actual_loss.detach()))
