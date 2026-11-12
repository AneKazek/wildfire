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


def test_spectral_block_is_identity_at_init():
    """Orthonormal basis + gain 1 + a threshold near zero must round-trip the input."""
    from wf.models import TDUNet, basis, Spectral

    for kind in ("ht", "dct"):
        B = basis(kind, 16)
        assert torch.allclose(B @ B.t(), torch.eye(16), atol=1e-5), kind

    x = torch.randn(2, 4, 16, 16)
    s = Spectral(4, 16, "ht")
    with torch.no_grad():
        s.thr.fill_(-20.0)  # softplus(-20) ~ 2e-9
        assert torch.allclose(s(x), x, atol=1e-4)

    net = TDUNet(23, width=4, branches=("ht", "dct"), depth=1, sizes=(16, 8, 4))
    out = net(torch.randn(2, 23, 16, 16))
    assert out.shape == (2, 4, 16, 16), out.shape
    assert torch.isfinite(out).all()


def test_pf_aug_keeps_mask_in_unit_range():
    from wf.data import PREV_FIRE, _pf_aug

    x = np.zeros((23, 64, 64), np.float16)
    x[PREV_FIRE][20:24, 20:24] = 1.0
    rng = np.random.default_rng(0)
    for mode in ("margin", "smooth", "both"):
        y = x.copy()
        _pf_aug(y, mode, rng)
        assert 0.0 <= y[PREV_FIRE].min() and y[PREV_FIRE].max() <= 1.0, mode
        assert y[PREV_FIRE].max() > 0.4, mode  # fire pixels stay fire-like
    y = x.copy()
    _pf_aug(y, "margin", rng)
    assert y[PREV_FIRE][0, 0] < 0.04 and y[PREV_FIRE][21, 21] > 0.7


def test_adaptive_threshold_beats_global_when_sizes_differ():
    """Big and small fires need different cut-offs; the fitted rule must find both."""
    from wf.athr import cv, fit, score
    from wf.metrics import dice_per_sample

    n = 2000
    prob = np.full((n, 64, 64), 0.20, np.float32)
    tgt = np.zeros((n, 64, 64), bool)
    for i in range(n):
        if i % 2 == 0:                       # 3 px fire, weak halo -> needs a high cut-off
            prob[i, 20:44, 20:44] = 0.55
            prob[i, 30:33, 30:33] = 0.75
            tgt[i, 30:33, 30:33] = True
        else:                                # 900 px fire, core 0.45 -> needs a low cut-off
            prob[i, 14:50, 14:50] = 0.42
            prob[i, 17:47, 17:47] = 0.45
            tgt[i, 17:47, 17:47] = True
    valid = np.ones_like(tgt)

    rule = fit(prob, tgt, valid)
    global_dice = float(dice_per_sample((prob > rule["global"]) & valid, tgt & valid).mean())
    assert score(prob, tgt, valid, rule) > global_dice + 0.2
    assert cv(prob, tgt, valid)["cv"] > global_dice + 0.2


def test_prep_channels_survive_dihedral():
    """Derived channels must be rotation scalars: aug-then-derive == derive-then-aug."""
    from wf.prep import PREPS, extras

    rng = np.random.default_rng(0)
    x = rng.random((len(CHANNELS), 64, 64)).astype(np.float32) - 0.5
    x[18] = (rng.random((64, 64)) > 0.8).astype(np.float32)   # prev_fire
    x[20] = np.clip(rng.normal(0.5, 0.4, (64, 64)), -0.3, 1)  # fire_dist
    x[10], x[11] = rng.normal(0, 1, (2, 64, 64))              # wind u, v

    for name in PREPS:
        e = extras(x, name)
        assert e.shape == (len(PREPS[name]), 64, 64), name
        assert np.isfinite(e).all(), name
        if not PREPS[name]:
            continue
        k = 5
        xa, _, _, _ = dihedral(x, np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32),
                               np.zeros((64, 64), np.float32), k)
        ea = _geo(extras(xa, name), k)
        assert np.allclose(e, ea, atol=2e-2), (name, np.abs(e - ea).max())


def test_soft_target_and_noscar_drop(tmp_path):
    """Soft labels keep the fire mass but lose the hard edge; `noscar` drops 2 channels."""
    from wf.data import FireDataset

    rng = np.random.default_rng(0)
    n = 2
    (tmp_path / "train").mkdir()
    fire = np.zeros((n, 64, 64), np.uint8)
    fire[:, 30:32, 30:33] = 8                       # 6 fire px per patch
    prev = np.zeros((n, 64, 64), np.uint8)
    prev[:, 20:22, 20:22] = 8
    (tmp_path / "train" / "features.f16").write_bytes(
        rng.random((n, len(CHANNELS), 64, 64)).astype(np.float16).tobytes())
    (tmp_path / "train" / "fire_class.u8").write_bytes(fire.tobytes())
    (tmp_path / "train" / "prev_class.u8").write_bytes(prev.tobytes())

    hard = FireDataset(tmp_path, "train", np.arange(n))[0]
    x, y, v, _ = FireDataset(tmp_path, "train", np.arange(n), soft=0.8)[0]
    assert x.shape[0] == len(CHANNELS), x.shape
    assert float(y.max()) < 1.0 and abs(float(y.sum()) - 6.0) < 0.5, float(y.sum())
    assert np.array_equal(np.asarray(v), np.ones((64, 64), np.float32))
    assert np.array_equal(np.asarray(y), np.asarray(hard[1])) is False
    assert FireDataset(tmp_path, "train", np.arange(n), drop="noscar")[0][0].shape[0] == len(CHANNELS) - 2


def test_aniso_cone_follows_wind():
    """A 90-degree rotation error would still pass the equivariance test, so check the axis."""
    from wf.prep import extras

    x = np.zeros((len(CHANNELS), 64, 64), np.float32)
    x[20] = 1.0
    x[20][30:34, 30:34] = 0.0                     # small scar blob near the centre
    x[18][30:34, 30:34] = 1.0                     # ... and it is the previous fire mask
    yy, xx = np.mgrid[0:64, 0:64]
    for wind, axis, want in [((-1, 0), 1, 1),   # wind from the west -> spread along +x
                             ((0, -1), 0, 1),   # wind from the north -> spread along +y
                             ((1, 0), 1, 1)]:   # wind from the east -> spread along -x
        x[10], x[11] = wind
        a = extras(x, "wind")[4].astype(np.float32)  # aniso
        var = [(a * (yy - 31.5) ** 2).sum() / a.sum(), (a * (xx - 31.5) ** 2).sum() / a.sum()]
        other = 1 - axis
        assert var[axis] > 2 * var[other], (wind, var)
