"""Derived input channels for the preprocessing sweep.

Everything here is computed from a patch that has *already* been dihedral
augmented (see `FireDataset.__getitem__`), so no channel needs its own
transform bookkeeping. Every channel is a rotation/flip scalar -- distance,
band, projection, dot product, isotropic blur -- which is why that works:
rotating the patch rotates the input to the function, and the output is the
rotated output.

Why these: `fire_dist` (channel 20) is a *clipped* signed distance,
`clip(px * 0.1, -0.3, 1.0)`, so a scar interior deeper than 3 px and a fire
far away are both flat, and `fire_front` is just a 1-px dilation of
`prev_fire`. The channels below add the information the clip throws away plus
wind anisotropy, which is the physics of spread.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PREV, FRONT, DIST = 18, 19, 20  # feature indices


def _pf(x: np.ndarray) -> np.ndarray:
    return x[PREV] > 0.5


def _unit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downwind unit vector: wind channels point where the wind comes from."""
    u, v = -x[10].astype(np.float32), -x[11].astype(np.float32)
    s = np.hypot(u, v) + 1e-6
    return u / s, v / s, s


_YY, _XX = np.mgrid[0:64, 0:64]
_XX = (_XX - 31.5).astype(np.float32)
_YY = (_YY - 31.5).astype(np.float32)


def ch_edt_in(x):
    return (ndi.distance_transform_edt(_pf(x)) / 8.0).astype(np.float32)


def ch_interior(x):
    return (ch_edt_in(x) * 8.0 > 1.5).astype(np.float32)


def ch_edt_out(x):
    return (ndi.distance_transform_edt(~_pf(x)) / 32.0).astype(np.float32)


def ch_near(x):
    return np.sqrt(np.clip(x[DIST], 0, 1)).astype(np.float32)


def _blur(x, sigma):
    return ndi.gaussian_filter(-x[DIST].astype(np.float32), sigma)


def ch_blur2(x):
    return _blur(x, 2.0)


def ch_blur5(x):
    return _blur(x, 5.0)


def ch_proj(x):
    u, v, _ = _unit(x)
    return ((_XX * u + _YY * v) / 16.0).astype(np.float32)


def ch_proj_pos(x):
    return np.maximum(ch_proj(x), 0).astype(np.float32)


def ch_proj_neg(x):
    return np.maximum(-ch_proj(x), 0).astype(np.float32)


def ch_gdot(x):
    """Approach rate: gradient of distance dotted with the downwind direction."""
    u, v, _ = _unit(x)
    gy, gx = np.gradient(-x[DIST].astype(np.float32))
    return (gx * u + gy * v).astype(np.float32)


def ch_aniso(x):
    """Closeness (1 at the scar, 0 beyond 10 px) blurred 6 px along the wind and 1.5 across."""
    if not _pf(x).any():  # nothing to spread from; the two rotations are the slowest ops here
        return np.zeros((64, 64), np.float32)
    u, v, _ = _unit(x)
    # rotate takes a scalar angle; wind is near-uniform inside one patch
    theta = float(np.degrees(np.arctan2(v.mean(), u.mean())))
    c = np.clip(1.0 - x[DIST].astype(np.float32), 0, 1)
    a = ndi.gaussian_filter(ndi.rotate(c, -theta, reshape=False, order=1), (1.5, 6.0))
    return ndi.rotate(a, theta, reshape=False, order=1).astype(np.float32)


def ch_ring2(x):
    pf = _pf(x)
    return (ndi.binary_dilation(pf, iterations=2) & ~ndi.binary_dilation(pf, iterations=1)).astype(np.float32)


def ch_pf_ndvi(x):
    return (_pf(x) * x[0]).astype(np.float32)


def ch_pf_slope(x):
    return (_pf(x) * x[2]).astype(np.float32)


def ch_pf_erc(x):
    return (_pf(x) * x[17]).astype(np.float32)


CH = {f.__name__[3:]: f for f in [
    ch_edt_in, ch_interior, ch_edt_out, ch_near, ch_blur2, ch_blur5,
    ch_proj, ch_proj_pos, ch_proj_neg, ch_gdot, ch_aniso, ch_ring2,
    ch_pf_ndvi, ch_pf_slope, ch_pf_erc,
]}

SCAR = ["edt_in", "interior", "ring2", "pf_ndvi", "pf_slope", "pf_erc"]
CTX = ["edt_out", "near", "blur2", "blur5"]
WIND = ["proj", "proj_pos", "proj_neg", "gdot", "aniso"]

PREPS: dict[str, list[str]] = {
    "base": [],
    "ctx": CTX,
    "scar": SCAR,
    "wind": WIND,
    "scarwind": SCAR + WIND,
    "all": CTX + SCAR + WIND,
    "dist": ["edt_in", "edt_out"],  # replaces the clipped fire_dist channel
}

# dropped base channels: near-constant or duplicated information
DROPS: dict[str, list[int]] = {"nopop": [5, 6], "nocoord": [21, 22], "lean": [5, 6, 21, 22],
                               "nodist": [DIST], "noscar": [FRONT, DIST]}


def extras(x: np.ndarray, prep: str) -> np.ndarray:
    e = np.stack([CH[c](x) for c in PREPS[prep]]) if PREPS[prep] else np.zeros((0, 64, 64), np.float32)
    # float16 across the worker boundary, same as the base channels
    return e.astype(np.float16)


def _key(prep: str, drop: str) -> str:
    return f"{prep}|{drop}"


def norm_for(root: str | Path, prep: str = "base", drop: str = "") -> tuple[list[float], list[float]]:
    """Mean/std matching the channel layout the dataset will produce."""
    d = json.loads((Path(root) / "norm.json").read_text())
    keep = [i for i in range(len(d["mean"])) if i not in DROPS.get(drop, [])]
    mean = [d["mean"][i] for i in keep]
    std = [d["std"][i] for i in keep]
    if PREPS[prep]:
        path = Path(root) / "norm_extra.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        k = _key(prep, drop)
        if k not in cache:
            cache[k] = _measure(root, prep)
            path.write_text(json.dumps(cache, indent=1))
        mean += cache[k]["mean"]
        std += cache[k]["std"]
    return mean, std


def _measure(root: str | Path, prep: str, n: int = 2000, seed: int = 0) -> dict:
    from .data import FireDataset

    total = (Path(root) / "train" / "fire_class.u8").stat().st_size // (64 * 64)
    ds = FireDataset(root, "train", np.random.default_rng(seed).choice(total, min(n, total), replace=False), aux=False)
    vals = np.stack([extras(np.asarray(ds.x[i]), prep).astype(np.float32) for i in range(n)])
    return {"mean": vals.mean((0, 2, 3)).tolist(),
            "std": np.maximum(vals.std((0, 2, 3)), 1e-3).tolist()}
