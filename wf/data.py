"""Cached 23-channel patches -> training tensors.

The heavy features (slope, fire_front, fire_dist, ...) are materialised once into
`features.npy` by the old pipeline; here we only read rows and augment, which is
what keeps the L4 busy instead of the CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = [
    "ndvi", "elevation", "slope", "aspect_sin", "aspect_cos", "population", "water",
    "th_sin", "th_cos", "wind_speed", "wind_u", "wind_v", "tmmx", "tmmn", "sph",
    "pr", "pdsi", "erc", "prev_fire", "fire_front", "fire_dist", "coord_x", "coord_y",
]
# (u, v) channel pairs that must rotate with the image during dihedral augmentation.
PAIRS = ((3, 4), (10, 11))
PREV_FIRE = 18
DYNAMIC = {"th_sin", "th_cos", "wind_speed", "wind_u", "wind_v", "tmmx", "tmmn", "sph", "pr", "pdsi", "erc"}
DYN_IDX = [CHANNELS.index(c) for c in CHANNELS if c in DYNAMIC]
STA_IDX = [i for i in range(len(CHANNELS)) if i not in DYN_IDX]


def _geo(a: np.ndarray, k: int) -> np.ndarray:
    if k & 1:
        a = a[..., ::-1]
    if k & 2:
        a = a[..., ::-1, :]
    if k & 4:
        a = np.rot90(a, 1, axes=(-2, -1))
    return a


_JAC: dict[int, np.ndarray] = {}


def jac(k: int) -> np.ndarray:
    """2x2 rotation part of transform k, for (u=east, v=south) vector channels."""
    if k not in _JAC:
        n = 5
        X = np.tile(np.arange(n, dtype=np.float32), (n, 1))
        Y = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, n))
        Xp, Yp = _geo(X, k), _geo(Y, k)
        _JAC[k] = np.array([
            [(Xp[2, 3] - Xp[2, 1]) / 2, (Xp[3, 2] - Xp[1, 2]) / 2],
            [(Yp[2, 3] - Yp[2, 1]) / 2, (Yp[3, 2] - Yp[1, 2]) / 2],
        ], dtype=np.float32)
    return _JAC[k]


def dihedral(x: np.ndarray, y: np.ndarray, valid: np.ndarray, prev: np.ndarray, k: int):
    x = _geo(x, k).copy()
    J = jac(k)
    for u, v in PAIRS:
        uu, vv = x[u].copy(), x[v].copy()
        x[u] = J[0, 0] * uu + J[0, 1] * vv
        x[v] = J[1, 0] * uu + J[1, 1] * vv
    return x, _geo(y, k).copy(), _geo(valid, k).copy(), _geo(prev, k).copy()


def boundary_of(y: np.ndarray) -> np.ndarray:
    p = np.pad(y, 1)
    core = p[0:-2, 1:-1] * p[1:-1, 1:-1] * p[2:, 1:-1] * p[1:-1, 0:-2] * p[1:-1, 2:]
    return (y - (core > 0.5)).astype(np.float32)


def split_indices(root: str | Path, split: str, n: int | None, seed: int = 0) -> np.ndarray:
    total = len(np.load(Path(root) / split / "fire_class.npy", mmap_mode="r"))
    if n is None or n >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n, replace=False))


class FireDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        indices: np.ndarray | None = None,
        augment: bool = False,
        prev_dropout: float = 0.0,
        seed: int = 0,
        aux: bool = True,
        in_memory: bool = False,
    ) -> None:
        base = Path(root) / split
        sel = np.arange(len(np.load(base / "fire_class.npy", mmap_mode="r"))) if indices is None else np.asarray(indices)
        if in_memory:
            # Screening subsets are scattered over a 19 GB memmap on a slow
            # overlay FS: re-reading them every epoch cost ~20 s/epoch.
            self.x = np.load(base / "features.npy", mmap_mode="r")[sel]
            self.fire = np.load(base / "fire_class.npy", mmap_mode="r")[sel]
            self.prev = np.load(base / "prev_class.npy", mmap_mode="r")[sel]
            self.idx = np.arange(len(sel))
        else:
            self.x = np.load(base / "features.npy", mmap_mode="r")
            self.fire = np.load(base / "fire_class.npy", mmap_mode="r")
            self.prev = np.load(base / "prev_class.npy", mmap_mode="r")
            self.idx = sel
        self.augment = augment
        self.prev_dropout = prev_dropout
        self.aux = aux
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        # float16 across the worker boundary: half the IPC, cast on GPU instead.
        x = np.array(self.x[j], dtype=np.float16)
        fire = self.fire[j].astype(np.int16)
        y = (fire >= 7).astype(np.float32)
        valid = ((fire != 4) & (fire != 6)).astype(np.float32)
        prev = (self.prev[j] >= 7).astype(np.float32)
        if self.augment:
            x, y, valid, prev = dihedral(x, y, valid, prev, int(self.rng.integers(8)))
            if self.prev_dropout and self.rng.random() < self.prev_dropout:
                x[PREV_FIRE] = 0.0
        if self.aux:
            aux = np.stack([boundary_of(y), y * (1.0 - prev), prev])
        else:
            aux = np.zeros((1, 1, 1), np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid), torch.from_numpy(aux)
