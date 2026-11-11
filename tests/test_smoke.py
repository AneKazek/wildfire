"""Fast checks: vector-channel augmentation, model shapes, metrics, losses, TTA."""

import numpy as np
import torch

from wf.data import CHANNELS, _geo, dihedral, jac
from wf.final import _geo as tgeo, _inv
from wf.losses import lovasz_hinge, tversky, bce_dice
from wf.metrics import evaluate, sweep
from wf.models import Net

MEAN = [0.0] * len(CHANNELS)
STD = [1.0] * len(CHANNELS)


def test_dihedral_rotates_vector_channels():
    """A blob of 'east-pointing' vectors must move with the image and rotate by J."""
    n = 8
    m = np.zeros((n, n), np.float32)
    m[2:4, 5:7] = 1.0
    for k in range(8):
        J = jac(k)
        x = np.zeros((len(CHANNELS), n, n), np.float32)
        x[3], x[4] = m, np.zeros((n, n), np.float32)  # u=east, v=0
        out, _, _, _ = dihedral(x, np.zeros((n, n), np.float32), np.ones((n, n), np.float32),
                                np.zeros((n, n), np.float32), k)
        assert np.allclose(out[3], _geo(m, k) * J[0, 0]), k
        assert np.allclose(out[4], _geo(m, k) * J[1, 0]), k


def test_dihedral_target_matches_input():
    y = np.zeros((8, 8), np.float32)
    y[1:3, 5:7] = 1
    for k in range(8):
        _, out, _, _ = dihedral(np.zeros((len(CHANNELS), 8, 8), np.float32), y, np.ones((8, 8), np.float32),
                                np.zeros((8, 8), np.float32), k)
        assert np.array_equal(out, _geo(y, k))


def test_model_shapes():
    cfgs = [
        {"trunk": "fullres", "width": 16, "blocks": [[1, 1], [4, 1]], "heads": ["main", "boundary", "growth"]},
        {"trunk": "unet", "widths": [16, 32], "depths": [1, 1]},
        {"trunk": "hrnet", "width": 8, "stages": 2},
    ]
    x = torch.randn(2, len(CHANNELS), 64, 64)
    for c in cfgs:
        out = Net(c, MEAN, STD)(x)
        assert out["main"].shape == (2, 1, 64, 64), c


def test_metrics_perfect_and_threshold():
    rng = np.random.default_rng(0)
    tgt = rng.random((4, 64, 64)) > 0.99
    valid = np.ones_like(tgt)
    prob = tgt.astype(np.float32) * 0.9 + 0.05
    m = evaluate(prob, tgt, valid, 0.5)
    assert m["macro_dice"] > 0.99 and m["recall"] > 0.99
    thr, rows = sweep(prob, tgt, valid, thrs=[0.3, 0.5, 0.7])
    assert thr in (0.3, 0.5, 0.7) and len(rows) == 3


def test_losses_are_finite():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    y = (torch.rand(2, 1, 16, 16) > 0.9).float()
    v = torch.ones_like(y)
    for loss in (bce_dice(logits, y, v), tversky(logits, y, v), lovasz_hinge(logits, y, v)):
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)


def test_tta_inverse_undoes_transform():
    p = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    for k in range(8):
        assert torch.equal(_inv(tgeo(p, k), k), p), k
