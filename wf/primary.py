"""WildfireTD4: the selected four-scale, three-branch custom segmentation model.

Uses the exact parameter layout of final_td4_w24_tversky_recall_seed0 so its
existing checkpoint can be loaded strictly, without retraining or key rewriting.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from .data import CHANNELS
from .models import Net

ARCHITECTURE = {
    'trunk': 'td', 'branches': ['ht', 'dct', 'conv'], 'width': 24,
    'sizes': [64, 32, 16, 8], 'depth': 1, 'fusion': 'gate',
    'attn': False, 'dec': 1, 'ds': False, 'hr': 0,
    'heads': ['main'], 'late': 0,
}
LOSS = {'name': 'focal_tversky', 'alpha': 0.4, 'beta': 0.6, 'gamma': 1.33}
REFERENCE_THRESHOLD = 0.35
REFERENCE_RUN = 'final_td4_w24_tversky_recall_seed0'


class WildfireTD4(Net):
    """33 x 64 x 64 inputs -> one logit per pixel; 5,973,133 parameters.

    Parallel Hadamard, DCT and convolution encoders have widths 24/48/96/192.
    Pixel gates combine their features at each scale; one skip decoder predicts
    next-day progression. External weather is not required by this selected run.
    """
    def __init__(self, mean, std):
        if len(mean) != len(CHANNELS) or len(std) != len(CHANNELS):
            raise ValueError('WildfireTD4 requires the 33-channel TS-SatFire cache')
        super().__init__({**copy.deepcopy(ARCHITECTURE), 'cin': len(CHANNELS)}, mean, std)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (33, 64, 64):
            raise ValueError('Expected input shape (batch, 33, 64, 64)')
        return super().forward(x)


def configuration(root: str | Path, seed=0):
    """The full-data finalist recipe, using the existing tuning partition."""
    return {
        'name': 'wildfire_td4', 'seed': seed,
        'data': {'root': str(root), 'in_memory': False, 'prev_dropout': 0.2},
        'model': {'trunk': 'wildfire_td4'}, 'loss': copy.deepcopy(LOSS),
        'train': {'epochs': 60, 'n_train': None, 'batch_size': 32,
                  'eval_batch_size': 64, 'workers': 2, 'eval_workers': 2,
                  'amp': 'fp16', 'lr': 0.002, 'patience': 12, 'warmup_epochs': 2,
                  'checkpoint_sweep': True, 'max_seconds': 2400},
    }


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    model = WildfireTD4(state['mean'].flatten().tolist(), state['std'].flatten().tolist())
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint
