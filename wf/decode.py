"""Turn eval/test probabilities into a mask: blob-at-centroid shape prior vs threshold.

Measured on td_gate3_s1: the centroid of the thresholded mask is within 1.24 px
(median) of the GT centroid, and total probability mass is within 4% of total GT
area, but the mask at any threshold is fragmented, so per-patch Dice is 0.459.
Replacing the mask by a compact blob at *that centroid* is worth ~+0.04.

    python -m wf.decode --run td_gate3_s1

`k` is the blob area. `mass` = sum of the probabilities (calibrated, carpet
included); `px` = number of pixels above the threshold (biased small).
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


def _blob(cy: float, cx: float, k: int) -> np.ndarray:
    d = ((YS - cy) ** 2 + (XS - cx) ** 2).ravel()
    kk = int(np.clip(k, 1, 4096))
    return d <= np.partition(d, kk - 1)[kk - 1]


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def decode(p: np.ndarray, thr: float = 0.65, mode: str = "maskdisk", alpha: float = 1.0,
           small: float = 1e9) -> np.ndarray:
    """`small` only affects `hybrid`: below it the blob, above it the threshold."""
    out = np.zeros(p.shape, bool)
    for i in range(len(p)):
        hard = p[i] > thr
        if mode == "base":
            out[i] = hard
            continue
        if not hard.any():
            continue
        mass = float(p[i].sum())
        if mode == "hybrid" and mass >= small:
            out[i] = hard
            continue
        cy, cx = _centroid(hard)
        k = round(alpha * (mass if mode != "maskdisk_px" else hard.sum()))
        out[i] = _blob(cy, cx, k).reshape(64, 64)
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
    base = p > thr

    def score(masks, sel=None):
        if sel is None:
            return float(dice_per_sample(masks & v, g).mean())
        return float(dice_per_sample(masks & v[sel], g[sel]).mean())

    print(f"{args.run} {args.split}  thr {thr:.2f}  base {score(base):.4f}")
    rows = {}
    for mode, alpha in [("maskdisk", 0.9), ("maskdisk", 1.0), ("maskdisk", 1.1), ("maskdisk", 1.2),
                        ("maskdisk_px", 1.2), ("maskdisk_px", 1.3), ("maskdisk_px", 1.4)]:
        rows[f"{mode} a={alpha}"] = round(score(decode(p, float(thr), mode, alpha)), 4)
    for small in (10, 20, 40, 80):
        rows[f"hybrid mass<{small}"] = round(score(decode(p, float(thr), "hybrid", 1.0, small)), 4)
    print(json.dumps(rows, indent=1))

    masks = decode(p, float(thr), "maskdisk", 1.0)
    for lo, hi in [(0, 9), (10, 29), (30, 99), (100, 10 ** 6)]:
        sel = (area >= lo) & (area <= hi)
        if not sel.any():
            continue
        hi_s = "inf" if hi >= 10 ** 6 else str(hi)
        print(f"  area {lo:>3}-{hi_s:>3} n={sel.sum():>5}  base {score(base, sel):.4f}  "
              f"blob {score(masks, sel):.4f}")

    cy = np.array([_centroid(m)[0] if m.any() else 31.5 for m in base])
    cx = np.array([_centroid(m)[1] if m.any() else 31.5 for m in base])
    orc = np.stack([_blob(cy[i], cx[i], int(area[i])) for i in range(len(p))]).reshape(p.shape)
    print(f"  oracle-area blob at predicted centroid: {score(orc):.4f}")

    best = decode(p, float(thr), "maskdisk", 1.0)
    np.save(out / f"{args.split}_blob.npy", best)
    (out / f"{args.split}_decode.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
