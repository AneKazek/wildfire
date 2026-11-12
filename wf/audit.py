"""Dataset audit: label classes, target size, scar overlap, trivial baselines.

Prints the numbers the cleaning/experiment decisions rest on. Cheap: pure
TFRecord reads, no cache needed.

    python -m wf.audit --split eval --limit 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .build import read_shard

ROOT = Path("datasets/next-day-wildfire-spread")


def _dice(pred: np.ndarray, tgt: np.ndarray) -> float:
    den = pred.sum() + tgt.sum()
    return 1.0 if den == 0 else float(2 * np.logical_and(pred, tgt).sum() / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    files = sorted((Path(args.root) / args.split).rglob("*.tfrecord"))
    fires, prevs, classes = [], [], []
    for f in files:
        for ex in read_shard(f):
            fires.append(ex["FireMask"].astype(np.uint8))
            prevs.append((ex["PrevFireMask"] >= 7).astype(bool))
            classes.append(ex["FireMask"].astype(np.uint8).ravel())
            if len(fires) >= args.limit:
                break
        if len(fires) >= args.limit:
            break

    fire = np.stack(fires)
    prev = np.stack(prevs)
    tgt = fire >= 7
    cls, cnt = np.unique(np.concatenate(classes), return_counts=True)
    area = tgt.sum((1, 2))

    print(f"{args.split}: {len(fire)} patches")
    print("class counts (px, %):",
          {int(c): f"{n} ({100 * n / fire.size:.2f}%)" for c, n in zip(cls, cnt)})
    print("valid-mask classes in use: target >=7, ignore {4,6}")
    print(f"empty GT: {(area == 0).mean():.1%}   area pct 10/25/50/75/90/99: "
          + "/".join(str(int(np.percentile(area, q))) for q in (10, 25, 50, 75, 90, 99)))
    print(f"patches with <10 GT px: {(area < 10).mean():.1%}   >=100: {(area >= 100).mean():.1%}")

    # how much of the target is copyable from the previous scar
    inter = np.logical_and(tgt, prev).sum((1, 2))
    print(f"GT px inside prev scar: {inter.sum() / max(tgt.sum(), 1):.1%}   "
          f"patches where GT == prev: {np.mean([np.array_equal(a, b) for a, b in zip(tgt, prev)]):.1%}")

    # 5 is background here; 4 and 6 are the ignore labels that leak into the loss
    soft = np.isin(fire, (4, 6))
    edge = tgt & ~ndi.binary_erosion(tgt)
    near = ndi.binary_dilation(soft, iterations=1)
    print(f"target edge px adjacent to class 4/6: {(edge & near).sum() / max(edge.sum(), 1):.1%}   "
          f"patches containing 4/6: {np.mean([s.any() for s in soft]):.1%}   all 4/6 px: {soft.sum()}")

    # how much of the metric is just "put a correctly sized blob at the right place"
    yy, xx = np.mgrid[0:64, 0:64]
    disk_d, disk_c, cent = [], [], []
    for t in tgt:
        k = int(t.sum())
        cy, cx = ndi.center_of_mass(t)
        d = (yy - cy) ** 2 + (xx - cx) ** 2
        near_px = d <= np.sort(d.ravel())[min(k, 4096) - 1]
        disk_d.append(_dice(near_px, t))
        c = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 2 * 2
        disk_c.append(_dice(c, t))
        cent.append(float(np.hypot(cy - 31.5, cx - 31.5)))
    print(f"dice of best-area disk at GT centroid: {np.mean(disk_d):.4f}   "
          f"GT centroid dist to patch centre: p50 {np.median(cent):.1f} p90 {np.percentile(cent, 90):.1f}")
    print(f"dice of fixed radius-2 centre disk: {np.mean(disk_c):.4f}")

    # trivial baselines, same metric as the model (macro Dice, empty = 1.0)
    out = {"copy_prev_scar": [], "prev_dilate1": [], "prev_both_wind": [], "empty": []}
    for t, p in zip(tgt, prev):
        out["copy_prev_scar"].append(_dice(p, t))
        out["prev_dilate1"].append(_dice(ndi.binary_dilation(p), t))
        out["prev_both_wind"].append(_dice(p | ndi.binary_dilation(p), t))
        out["empty"].append(_dice(np.zeros_like(t), t))
    print("trivial macro Dice:", {k: round(float(np.mean(v)), 4) for k, v in out.items()})
    # upper bound if only the area were right: top-k of the GT itself
    print(f"mean GT area on non-empty: {area[area > 0].mean():.1f} px")


if __name__ == "__main__":
    main()
