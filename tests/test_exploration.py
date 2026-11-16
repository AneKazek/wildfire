"""Run these checks on Kaggle, together with the experiment preflight."""
import numpy as np
import pandas as pd
import torch

from wf.data import CHANNELS, FireDataset
from wf.ext import build, scale_split
from wf.models import Net
from wf.train import _loader


def test_spatial_fusion_has_spatial_effect_and_gradients():
    torch.manual_seed(0)
    model = Net({'trunk': 'unet', 'widths': [8, 16], 'depths': [1, 1],
                 'cin': 33, 'late': 8, 'late_mode': 'spatial'}, [0.]*33, [1.]*33)
    x, ext = torch.randn(2, 33, 64, 64), torch.randn(2, 8)
    zero = model(x, ext)['main']
    zero.mean().backward()
    assert model.late[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.late[-1].weight.fill_(0.1)
    a, b = model(x, ext)['main'], model(x, torch.zeros_like(ext))['main']
    assert a.shape == (2, 1, 64, 64)
    assert (a - b).flatten(1).std(1).max() > 0


def test_missing_weather_does_not_erase_observed_column(tmp_path):
    raw = np.array([[1., 2., 3., 4., 5., 6., 0., 0.],
                    [3., 4., 5., 6., 7., 8., 0., 0.],
                    [np.nan]*7 + [1.]], np.float32)
    scale_split({'train': raw, 'eval': raw.copy()}, tmp_path)
    z = np.load(tmp_path / 'train/ext.npy')
    assert np.isfinite(z).all()
    assert z[0, 0] != z[1, 0]
    assert z[2, 0] == 0


def test_spherical_weather_join_and_integer_date():
    days = pd.date_range('2020-01-01', periods=2)
    mats = {key: np.array([[10., 12.], [20., 22.]], np.float32)
            for key in ('tmax', 'tmin', 'prcp')}
    meta = pd.DataFrame({'date': [20200102], 'lat': [40.], 'lon': [-120.]})
    out = build(meta, mats, np.array([[40., -120.], [0., 0.]]), days)
    assert out[0, 0] == 12.
    assert out[0, 6] == 0
    assert out[0, 7] == 0


def test_subset_and_loader_independent_of_model_rng(tmp_path):
    d = tmp_path / 'train'
    d.mkdir()
    np.zeros((4, len(CHANNELS), 64, 64), np.float16).tofile(d / 'features.f16')
    labels = np.full((4, 64, 64), 5, np.uint8)
    labels.tofile(d / 'fire_class.u8')
    labels.tofile(d / 'prev_class.u8')
    ext = np.arange(32, dtype=np.float32).reshape(4, 8)
    ds = FireDataset(tmp_path, indices=np.array([3, 1, 2]), in_memory=True, ext=ext)
    assert np.array_equal(ds[0][4].numpy(), ext[3])
    a = next(iter(_loader(ds, 2, True, 0, seed=42)))[4]
    torch.randn(300)
    b = next(iter(_loader(ds, 2, True, 0, seed=42)))[4]
    assert torch.equal(a, b)
