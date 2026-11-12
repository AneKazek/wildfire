"""Where the Dice is lost: rank vs location vs area vs diffuseness.

Loads a finished run, scores its eval probabilities once, and compares the
model mask against geometric proxies built from the same probabilities:

  base       p > thr (the number the search reports)
  topk       the |p| highest pixels per patch (same mass, sharpened)
  topk_gt    the GT-area highest pixels (perfect area, model ranking)
  disk_pre   GT-area disk at the predicted centroid (model location)
  disk_both  predicted-area disk at the predicted centroid (no shape at all)
  lcc/open   p > thr with specks removed / morphologically opened

    python -m wf.probe --run td_gate3_s1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import FireDataset
from .final import _targets, predict_tta
from .metrics import dice_per_sample, sweep
from .train import build

YY, XX = np.mgrid[0:64, 0:64]


def _disk_at(cy: float, cx: float, k: int) -> np.ndarray:
    d = (YY - cy) ** 2 + (XX - cx) ** 2
    cut = np.sort(d.ravel())[min(max(k, 1), 4096) - 1]
    return d <= cut


def _topk(flat: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(flat, dtype=bool)
    out[np.argsort(-flat)[: max(k, 0)]] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.runs) / args.run
    ck = torch.load(out / "best.pt", map_location="cpu")
    cfg = ck["cfg"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prep, drop = build(cfg, args.root)
    model = model.to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"], strict=False)

    ds = FireDataset(args.root, "eval", None, aux=False, prep=prep, drop=drop)
    tgt, valid = _targets(ds)
    p = predict_tta(model, torch.utils.data.DataLoader(ds, batch_size=512, num_workers=4),
                    dev, torch.bfloat16, args.tta)[0]
    np.save(out / "eval_prob.npy", p)
    thr, rows = sweep(p, tgt, valid)
    g = tgt & valid
    area = g.reshape(len(g), -1).sum(1)
    mass = p.reshape(len(p), -1).sum(1)

    def score(masks: np.ndarray) -> float:
        return float(dice_per_sample(masks & valid, g).mean())

    res = {"thr": round(float(thr), 3), "base": round(score(p > thr), 4)}
    base_masks = p > thr
    tk = np.stack([_topk(p[i].ravel(), int(round(mass[i]))) for i in range(len(p))]).reshape(p.shape)
    tk_gt = np.stack([_topk(p[i].ravel(), int(area[i])) for i in range(len(p))]).reshape(p.shape)
    res["topk"] = round(score(tk), 4)
    res["topk_gt"] = round(score(tk_gt), 4)

    disk_pre = np.zeros_like(base_masks)
    disk_both = np.zeros_like(base_masks)
    disk_peak = np.zeros_like(base_masks)
    cen_err = []
    for i in range(len(p)):
        m = base_masks[i]
        if not m.any():
            continue
        cy, cx = ndi.center_of_mass(m)
        disk_pre[i] = _disk_at(cy, cx, int(area[i]))
        disk_both[i] = _disk_at(cy, cx, int(round(m.sum())))
        py, px = np.unravel_index(int(np.argmax(ndi.gaussian_filter(p[i], 1.5))), p[i].shape)
        disk_peak[i] = _disk_at(py, px, int(round(m.sum())))
        gy, gx = ndi.center_of_mass(g[i])
        cen_err.append(float(np.hypot(cy - gy, cx - gx)))
    res["disk_pre"] = round(score(disk_pre), 4)
    res["disk_both"] = round(score(disk_both), 4)
    res["disk_peak"] = round(score(disk_peak), 4)
    res["miss"] = round(float(np.mean([not m.any() for m in base_masks])), 4)
    res["cen_err_p50"] = round(float(np.median(cen_err)), 2)
    res["cen_err_p90"] = round(float(np.percentile(cen_err, 90)), 2)
    res["area_ratio"] = round(float(mass.sum() / area.sum()), 3)

    lcc = np.zeros_like(base_masks)
    opened = np.zeros_like(base_masks)
    for i, m in enumerate(base_masks):
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            lcc[i] = lab == int(np.argmax(sizes)) + 1
        opened[i] = ndi.binary_opening(m, np.ones((3, 3)))
    res["lcc"] = round(score(lcc), 4)
    res["open3"] = round(score(opened), 4)
    res["best_thr"] = round(float(max(r[1] for r in rows)), 4)
    print(args.run, res, flush=True)


if __name__ == "__main__":
    main()
