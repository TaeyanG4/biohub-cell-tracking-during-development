"""Fixed equal raw-logit mean of two C052 models on every primary input.

No fitting, target labels, sample routing, probability calibration or new threshold.
The unchanged predictor applies its existing temporal/secondary fusion afterwards.
"""
from __future__ import annotations
import copy
import os
import torch
from c037_transformer_runtime import capture, state_digest


class OutputMean(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *args, **kwargs):
        a = self.first(*args, **kwargs)
        b = self.second(*args, **kwargs)
        return (a + b) * 0.5


def load_members(template, checkpoint):
    assert checkpoint['combination'] == 'equal_raw_logit_mean'
    assert checkpoint['base_transformer_sha256'] == state_digest(template.state_dict())
    assert len(checkpoint['members']) == 2
    models = []
    for member in checkpoint['members']:
        assert member['base_transformer_sha256'] == checkpoint['base_transformer_sha256']
        assert member['step'] == 600
        student = copy.deepcopy(template)
        student.load_state_dict(member['state_dict'], strict=True)
        student.eval()
        models.append(student)
    return OutputMean(*models).eval()


def apply_primary(model):
    path = os.environ.get('BIOHUB_C037_CHECKPOINT', '').strip()
    if not path:
        return
    assert float(os.environ.get('BIOHUB_C037_ALPHA', '1')) == 1.0
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.transformer = load_members(model.transformer, checkpoint)
    model.eval()
    print('C054_PRIMARY', dict(path=path, combination='equal_raw_logit_mean',
        members=2, alpha=1.0, step=600, train_embryo='fixed_both_no_routing',
        member_state_sha256=[state_digest(m['state_dict']) for m in checkpoint['members']]), flush=True)
