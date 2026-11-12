"""Final ensemble vs ground truth on the test split: per-sample stats + visual grid.

Reads the same 5 checkpoints and settings as `ensemble.py`, so `metrics` here
should reproduce the single reported test pass exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import FireDataset
from .ensemble import _load_run
from .final import _targets, predict_tta
from .metrics import dice_per_sample, evaluate
from .train import _loader

RUNS = "smp_unetpp_r18_s3,td_gate3_s3"
WEIGHTS = "0.6,0.4"


def _rgb(x: np.ndarray) -> np.ndarray:
    """Per-patch stretched elevation+NDVI relief with prev_fire in blue."""

    def n(i):
        a = x[i].astype(np.float32)
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    base = 0.6 * n(1) + 0.4 * n(0)
    img = np.repeat(base[..., None], 3, axis=2) * 0.85
    img[..., 2] = np.clip(img[..., 2] + 0.9 * n(18), 0, 1)
    return img


def _overlay(img: np.ndarray, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    out = img * 0.45
    out[gt] = [0.15, 0.85, 0.35]
    out[pr] = [0.95, 0.25, 0.20]
    out[gt & pr] = [1.0, 0.95, 0.20]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/test_vs_gt.png")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    root = "data/processed"
    names = args.runs.split(",")
    w = np.array([float(v) for v in args.weights.split(",")])
    assert len(w) == len(names)

    head = torch.load(Path("results/runs") / names[0] / "best.pt", map_location="cpu")["cfg"]["data"]
    prep, drop = head.get("prep", "base"), head.get("drop", "")

    ds = FireDataset(root, args.split, None, aux=False, prep=prep, drop=drop)
    tg, va = _targets(ds)
    ld = _loader(ds, 512, False, 4)

    cache = Path(f"results/{args.split}_blend.npy")
    if cache.exists():
        blend = np.load(cache)
        print("blend (cached)", flush=True)
    else:
        probs = []
        for name in names:
            model, _, _ = _load_run(name, dev, root)
            probs.append(predict_tta(model, ld, dev, dtype, args.tta)[0])
            del model
            torch.cuda.empty_cache()
        blend = np.average(probs, axis=0, weights=w)
        del probs
        np.save(cache, blend.astype(np.float32))

    metrics = evaluate(blend, tg, va, args.thr, full=True)
    print(json.dumps(metrics, indent=1), flush=True)

    pred = (blend > args.thr) & va
    gt = tg & va
    d = dice_per_sample(pred, gt)
    fp = np.logical_and(pred, ~gt).sum((1, 2))
    tp = np.logical_and(pred, gt).sum((1, 2))
    fn = np.logical_and(~pred, gt).sum((1, 2))

    order = np.argsort(d)
    picks, seen = [], set()
    for tag, i in [
        ("worst", order[0]),
        ("2nd worst", order[1]),
        ("median", order[len(d) // 2]),
        ("best", order[-1]),
        ("largest GT", int(np.argmax(gt.sum((1, 2))))),
        ("most false fire", int(np.argmax(fp - tp))),
    ]:
        if i not in seen:
            seen.add(int(i))
            picks.append((tag, int(i), float(d[i])))

    stats = {
        "split": args.split, "thr": args.thr, "tta": args.tta, "runs": names, "weights": w.tolist(),
        "metrics": metrics,
        "per_sample": {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90)),
            "share_ge_0.5": float((d >= 0.5).mean()), "share_zero": float((d == 0).mean()),
            "share_pred_empty_gt_not": float(((pred.sum((1, 2)) == 0) & (gt.sum((1, 2)) > 0)).mean()),
        },
        "pixels": {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
                   "recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
                   "precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1))},
        "cases": [{"tag": t, "index": i, "dice": dd,
                   "gt_px": int(gt[i].sum()), "pred_px": int(pred[i].sum()),
                   "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i])} for t, i, dd in picks],
    }
    Path("results/test_vs_gt.json").write_text(json.dumps(stats, indent=1))

    fig, axes = plt.subplots(len(picks), 5, figsize=(11.5, 2.3 * len(picks)), facecolor="white")
    for r, (tag, i, dd) in enumerate(picks):
        img = _rgb(np.asarray(ds.x[i]))
        g, p = gt[i], pred[i]
        grey = np.repeat(img.mean(2, keepdims=True), 3, 2) * 0.35
        panels = [
            (img, "input (elevation · ndvi · prev_fire)"),
            (blend[i], f"P(fire) — max {blend[i].max():.2f}"),
            (grey + g[..., None] * np.array([0.15, 0.85, 0.35]), "ground truth"),
            (grey + p[..., None] * np.array([0.95, 0.25, 0.20]), f"prediction (thr {args.thr})"),
            (_overlay(img, g, p), "overlay: hit / miss / false"),
        ]
        for c, (im, title) in enumerate(panels):
            ax = axes[r, c]
            if c == 1:
                ax.imshow(im, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(
            f"{tag}\ndice {dd:.3f}\nGT {int(gt[i].sum())}px · pred {int(pred[i].sum())}px", fontsize=8
        )
    fig.suptitle(
        f"test split: {len(names)}-model blend, TTA-{args.tta}, thr {args.thr} — macro Dice {metrics['macro_dice']:.4f} "
        f"(median {np.median(d):.3f}, {100 * (d >= 0.5).mean():.0f}% of patches >= 0.50)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=140)
    print(json.dumps(stats["per_sample"], indent=1), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
