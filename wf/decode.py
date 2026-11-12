"""Turn eval/test probabilities into a mask: shape prior vs plain threshold.

Measured on td_gate3_s1: the model's fire centroid is already within 1.24 px
(median) and its probability mass is within 4% of the GT area, but its mask at
any threshold is fragmented, so per-patch Dice is 0.459. Replacing the mask by a
compact blob of the same mass at that centroid scores 0.50. This module sweeps
that family of decoders and reports a per-size breakdown, so a size-conditioned
rule can be read off the eval split.

    python -m wf.decode --run td_gate3_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import FireDataset
from .final import _targets
from .metrics import dice_per_sample, sweep

YS, XS = np.mgrid[0:64, 0:64]
YS = YS.astype(np.float32)
XS = XS.astype(np.float32)


def _mass_centroid(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = p.sum((1, 2))
    cy = (p * YS).sum((1, 2)) / np.maximum(m, 1e-6)
    cx = (p * XS).sum((1, 2)) / np.maximum(m, 1e-6)
    return cy, cx, m


def _disk(cy, cx, k) -> np.ndarray:
    d = (YS - cy) ** 2 + (XS - cx) ** 2
    flat = d.ravel()
    kk = int(np.clip(k, 1, 4096))
    return flat <= np.partition(flat, kk - 1)[kk - 1]


def _topk(p: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(p.size, bool)
    out[np.argsort(-p)[:max(int(k), 0)]] = True
    return out


def decode(p: np.ndarray, thr: float = 0.65, mode: str = "disk", alpha: float = 1.0,
           small: float = 1e9) -> np.ndarray:
    """`small` switches to the blob only where the probability mass is below it (px)."""
    cy, cx, mass = _mass_centroid(p)
    out = np.zeros(p.shape, bool)
    for i in range(len(p)):
        p_i, m = p[i], mass[i]
        k = int(round(alpha * m))
        blob = _disk(cy[i], cx[i], k)
        if mode == "hybrid" and m < small:
            out[i] = (blob | _topk(p_i.ravel(), k // 2)).reshape(64, 64)
        elif m < small:
            out[i] = blob.reshape(64, 64)
        else:
            out[i] = p_i > thr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--root", default="data/processed")
    args = ap.parse_args()

    out = Path(args.runs) / args.run
    p = np.load(out / f"{args.split}_prob.npy")
    ds = FireDataset(args.root, args.split, None, aux=False)
    tgt, valid = _targets(ds)
    g, v = tgt & valid, valid
    area = g.reshape(len(g), -1).sum(1)
    thr, _ = sweep(p, tgt, valid)

    def score(masks):
        return float(dice_per_sample(masks & v, g).mean())

    print(f"{args.run} {args.split}  base thr {thr:.2f}  dice {score(p > thr):.4f}")

    rows = {}
    for mode, alpha, small in [("disk", 1.0, 1e9), ("disk", 0.9, 1e9), ("disk", 1.1, 1e9),
                               ("disk", 1.2, 1e9), ("hybrid", 1.0, 1e9),
                               ("disk", 1.0, 10), ("disk", 1.0, 20), ("disk", 1.0, 40),
                               ("disk", 1.0, 80)]:
        masks = decode(p, float(thr), mode, alpha, small)
        rows[f"{mode} a={alpha} <{small:g}"] = round(score(masks), 4)
        if small == 1e9 and alpha == 1.0:
            best_masks = masks
    print(json.dumps(rows, indent=1))

    # per-size breakdown: where does the disk win, where does the mask win?
    buckets = [(0, 9), (10, 29), (30, 99), (100, 10 ** 6)]
    masks = decode(p, float(thr), "disk", 1.0)
    for lo, hi in buckets:
        sel = (area >= lo) & (area <= hi)
        if not sel.any():
            continue
        hi_s = "inf" if hi >= 10 ** 6 else str(hi)
        print(f"  area {lo:>3}-{hi_s:>3} n={sel.sum():>5}  base {score((p > thr)[sel]):.4f}  "
              f"disk {score(masks[sel]):.4f}")

    # how much of the gap is area vs place: oracle-area disk vs predicted-area disk
    cy, cx, mass = _mass_centroid(p)
    orc = np.stack([_disk(cy[i], cx[i], int(area[i])) for i in range(len(p))]).reshape(p.shape)
    print(f"  oracle-area disk at pred centroid: {score(orc):.4f}   "
          f"mass/area {mass.sum() / area.sum():.3f}")
    np.save(out / f"{args.split}_disk.npy", best_masks)


if __name__ == "__main__":
    main()
