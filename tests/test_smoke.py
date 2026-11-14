"""Fast checks: vector-channel augmentation, model shapes, metrics, losses, TTA."""

import numpy as np
import torch

from wf.data import CHANNELS, PAIRS, _geo, dihedral, jac
from wf.final import _geo as tgeo, _inv
from wf.losses import lovasz_hinge, tversky, bce_dice, moment
from wf.metrics import evaluate, evaluate_area, sweep, sweep_area, topk_masks
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
        u, v = PAIRS[0]
        x[u], x[v] = m, np.zeros((n, n), np.float32)  # u=east, v=0
        out, _, _, _ = dihedral(x, np.zeros((n, n), np.float32), np.ones((n, n), np.float32),
                                np.zeros((n, n), np.float32), k)
        assert np.allclose(out[u], _geo(m, k) * J[0, 0]), k
        assert np.allclose(out[v], _geo(m, k) * J[1, 0]), k


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
    x = torch.randn(2, 23, 64, 64)
    m23, s23 = [0.0] * 23, [1.0] * 23
    for c in cfgs:
        out = Net(c, m23, s23)(x)
        assert out["main"].shape == (2, 1, 64, 64), c
    out = Net({"trunk": "td", "branches": ["ht", "dct"], "area": 16}, m23, s23)(x)
    assert out["area"].shape == (2,), out


def test_topk_decode_uses_the_area():
    """Top-k must put exactly k pixels per patch, in probability order; with the
    GT area it has to beat the best global threshold."""
    rng = np.random.default_rng(0)
    tgt = np.zeros((3, 64, 64), bool)
    for i in range(3):
        tgt[i, 20:20 + 2 + 3 * i, 28:28 + 2 + 3 * i] = True
    prob = tgt.astype(np.float32) * 0.5 + rng.random((3, 64, 64)).astype(np.float32) * 0.45
    valid = np.ones_like(tgt)
    k = tgt.reshape(3, -1).sum(1)
    masks = topk_masks(prob, k)
    assert np.array_equal(masks.reshape(3, -1).sum(1), k)
    assert np.array_equal(masks, tgt)
    assert evaluate_area(prob, k, tgt, valid)["macro_dice"] > 0.99
    g, rows = sweep_area(prob, k * 1.5, tgt, valid)
    assert len(rows) > 0 and g in [r[0] for r in rows]


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
    for loss in (bce_dice(logits, y, v), tversky(logits, y, v), lovasz_hinge(logits, y, v),
                 moment(logits, y, v)):
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)


def test_moment_prefers_the_right_place_and_size():
    """Moment term ~0 for the target itself; punished for a shifted or a
    diffuse blob of comparable mass."""
    y = torch.zeros(1, 1, 64, 64)
    y[0, 0, 30:33, 30:33] = 1.0
    v = torch.ones_like(y)
    same = torch.logit(y.clamp(1e-4, 1 - 1e-4))
    shifted = torch.zeros_like(y)
    shifted[0, 0, 40:43, 40:43] = 1.0
    diffuse = torch.full_like(y, 1e-4)
    diffuse[0, 0, 24:38, 24:38] = 0.1
    shifted = torch.logit(shifted.clamp(1e-4, 1 - 1e-4))
    diffuse = torch.logit(diffuse.clamp(1e-4, 1 - 1e-4))
    base = float(moment(same, y, v))
    assert base < 0.02, base
    assert float(moment(shifted, y, v)) > 10 * base
    assert float(moment(diffuse, y, v)) > 10 * base


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

    x = np.zeros((len(CHANNELS), 64, 64), np.float16)
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
    from wf.prep import DIST, PREPS, PREV, WINDS, extras

    rng = np.random.default_rng(0)
    x = rng.random((len(CHANNELS), 64, 64)).astype(np.float32) - 0.5
    x[PREV] = (rng.random((64, 64)) > 0.8).astype(np.float32)   # prev_fire
    x[DIST] = np.clip(rng.normal(0.5, 0.4, (64, 64)), -0.3, 1)  # fire_dist
    x[WINDS[0]], x[WINDS[1]] = rng.normal(0, 1, (2, 64, 64))    # wind direction

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
    from wf.prep import DIST, PREV, WINDS, extras

    x = np.zeros((len(CHANNELS), 64, 64), np.float32)
    x[DIST] = 1.0
    x[DIST][30:34, 30:34] = 0.0                   # small scar blob near the centre
    x[PREV][30:34, 30:34] = 1.0                   # ... and it is the previous fire mask
    yy, xx = np.mgrid[0:64, 0:64]
    for wind, axis, want in [((-1, 0), 1, 1),   # wind from the west -> spread along +x
                             ((0, -1), 0, 1),   # wind from the north -> spread along +y
                             ((1, 0), 1, 1)]:   # wind from the east -> spread along -x
        x[WINDS[0]], x[WINDS[1]] = wind
        a = extras(x, "wind")[4].astype(np.float32)  # aniso
        var = [(a * (yy - 31.5) ** 2).sum() / a.sum(), (a * (xx - 31.5) ** 2).sum() / a.sum()]
        other = 1 - axis
        assert var[axis] > 2 * var[other], (wind, var)


def test_tsf_patch_layout_and_centres():
    """The TS-SatFire converter must emit the channel layout the cache declares."""
    from wf.data import CHANNELS, PREV_FIRE
    from wf.tsf import _centres, _channels

    h = w = 128
    day = np.full((8, h, w), 300.0, np.float32)
    day[0:3] = 20.0
    day[6, 40:50, 40:60] = np.nan          # active-fire temperature, NaN off fire
    night = np.full((2, h, w), 290.0, np.float32)
    aux = np.zeros((19, h, w), np.float32)
    aux[10] = 180.0                         # aspect
    aux[4] = 270.0                          # wind direction
    aux[3] = 10.0                           # wind speed
    aux[13] = 5.0                           # landcover
    prev = np.zeros((h, w), bool)
    prev[50:70, 50:70] = True

    x = _channels(day, night, aux, prev, (30, 30))
    assert x.shape == (len(CHANNELS), 64, 64), x.shape
    assert np.isfinite(x).all()
    assert x[PREV_FIRE].sum() == 20 * 20
    assert abs(x[CHANNELS.index("wind_speed")].mean() - 10.0) < 1e-3
    assert abs(x[CHANNELS.index("wind_dir_sin")].mean() + 1.0) < 1e-3  # 270 deg

    from wf.tsf import _on_grid
    coarse = np.arange(2 * 10 * 12, dtype=np.float32).reshape(2, 10, 12)
    assert _on_grid(coarse, (10, 12), (16, 19)).shape == (2, 16, 19)

    new = np.zeros((h, w), bool)
    new[60:64, 5:9] = True
    boxes = _centres(new, k=4, rng=np.random.default_rng(0))
    assert len(boxes) == 1, boxes           # one bucket, four pixels
    r0, c0 = boxes[0]
    assert 0 <= r0 <= h - 64 and 0 <= c0 <= w - 64
    assert new[r0:r0 + 64, c0:c0 + 64].any()
