"""Post-processing sweep: coherent blob growth vs threshold.

The model under-predicts area (recall well below precision), and Dice punishes a
missing pixel exactly as hard as a false one only when the blob is big. Growing
the predicted mask by a small disk grows the blob without the scattered false
pixels that a lower threshold adds, so it can beat the threshold sweep at the
same recall.

`python -m wf.post --prob results/runs/<run>/eval_prob.npy` prints the grid.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage as ndi

from .metrics import dice_per_sample


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def grow(mask: np.ndarray, r: int) -> np.ndarray:
    if not r:
        return mask
    d = _disk(r)
    return ndi.binary_dilation(mask, structure=d.reshape((1,) * (mask.ndim - 2) + d.shape))


def score(prob: np.ndarray, tgt: np.ndarray, valid: np.ndarray, thr: float, r: int = 0) -> float:
    pred = grow((prob > thr) & valid, r)
    return float(dice_per_sample(pred, tgt & valid).mean())


def sweep(prob, tgt, valid, thrs=None, rs=(0, 1, 2)) -> list[list[float]]:
    thrs = np.arange(0.30, 0.86, 0.05) if thrs is None else thrs
    return [[score(prob, tgt, valid, float(t), r) for r in rs] for t in thrs]


def main() -> None:
    from .data import FireDataset
    from .final import _targets

    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--root", default="data/processed")
    args = ap.parse_args()

    ds = FireDataset(args.root, args.split, None, aux=False)
    tgt, valid = _targets(ds)
    prob = np.load(args.prob)
    rs = (0, 1, 2)
    rows = sweep(prob, tgt, valid, rs=rs)
    best = max(((t, r, v) for t, row in zip(np.arange(0.30, 0.86, 0.05), rows) for r, v in zip(rs, row)),
               key=lambda z: z[2])
    print("thr  " + "  ".join(f"r={r}" for r in rs))
    for t, row in zip(np.arange(0.30, 0.86, 0.05), rows):
        print(f"{t:.2f} " + "  ".join(f"{v:.4f}" for v in row))
    print(json.dumps({"best_thr": round(best[0], 3), "best_r": best[1], "best": round(best[2], 4)}))


if __name__ == "__main__":
    main()
