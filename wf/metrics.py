"""Metrics: macro/micro Dice, IoU, P/R, PR-AUC, boundary F1, hard Dice.

`hard_dice` only scores pixels outside the radius-2 disk at the patch centre,
which removes the trivially centred blob and measures actual spread.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _disk(shape=(64, 64), r=2):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - shape[0] // 2) ** 2 + (xx - shape[1] // 2) ** 2 <= r * r


OUTSIDE = ~_disk()


def dice_per_sample(pred: np.ndarray, tgt: np.ndarray, empty_one: bool = True) -> np.ndarray:
    inter = np.logical_and(pred, tgt).sum(axis=(1, 2), dtype=np.float64)
    den = pred.sum(axis=(1, 2), dtype=np.float64) + tgt.sum(axis=(1, 2), dtype=np.float64)
    empty = 1.0 if empty_one else 0.0
    return np.where(den > 0, 2 * inter / np.maximum(den, 1), empty)


def _edge(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1)
    core = p[:-2, 1:-1] & p[1:-1, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~core


def boundary_f1(pred: np.ndarray, tgt: np.ndarray, tol: float = 2.0) -> float:
    ps, ts = [], []
    for p, t in zip(pred, tgt):
        if p.sum() == 0 or t.sum() == 0:
            continue
        ep, et = _edge(p), _edge(t)
        dp, dt = distance_transform_edt(~ep), distance_transform_edt(~et)
        ps.append((ep & (dt <= tol)).sum() / max(ep.sum(), 1))
        ts.append((et & (dp <= tol)).sum() / max(et.sum(), 1))
    if not ps:
        return 0.0
    ps, ts = np.array(ps), np.array(ts)
    f = 2 * ps * ts / np.maximum(ps + ts, 1e-9)
    return float(f.mean())


def average_precision(prob: np.ndarray, tgt: np.ndarray, valid: np.ndarray, max_neg: int = 4_000_000,
                      seed: int = 0) -> float:
    p = prob.ravel()
    t = tgt.ravel().astype(bool)
    v = valid.ravel().astype(bool)
    p, t = p[v], t[v]
    neg = np.flatnonzero(~t)
    rng = np.random.default_rng(seed)
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    keep = np.concatenate([neg, np.flatnonzero(t)])
    p, t = p[keep], t[keep]
    order = np.argsort(-p)
    t = t[order]
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    rec = tp / max(t.sum(), 1)
    ap = np.sum(np.diff(np.concatenate([[0.0], rec])) * prec)
    return float(ap)


def topk_masks(prob: np.ndarray, k) -> np.ndarray:
    """Per-sample top-`k` pixels of `prob`: the decode an area estimate implies.

    Measured on td_gate3_s1: with the GT area this scores 0.554 versus 0.459 for
    the best global threshold, so area — not ranking — is the binding constraint.
    """
    n, m = prob.shape[0], prob[0].size
    order = np.argsort(-prob.reshape(n, m), axis=1)
    k = np.broadcast_to(np.asarray(k, np.float64), (n,))
    out = np.zeros((n, m), bool)
    for i in range(n):
        out[i, order[i, : int(np.clip(k[i], 1, m))]] = True
    return out.reshape(prob.shape)


def _score(pred: np.ndarray, tgt: np.ndarray, valid: np.ndarray, prob: np.ndarray,
           full: bool) -> dict:
    tg = tgt & valid
    d = dice_per_sample(pred, tg)
    inter = np.logical_and(pred, tg).sum()
    union = np.logical_or(pred, tg).sum()
    tp = inter
    out = {
        "macro_dice": float(d.mean()),
        "micro_dice": float(2 * inter / max(pred.sum() + tg.sum(), 1)),
        "iou": float(inter / max(union, 1)),
        "precision": float(tp / max(pred.sum(), 1)),
        "recall": float(tp / max(tg.sum(), 1)),
    }
    if full:
        hm = np.logical_and(pred, OUTSIDE)
        ht = np.logical_and(tg, OUTSIDE)
        # empty-outside counts as 0: this metric is about real spread, not the blob
        out["hard_dice"] = float(dice_per_sample(hm, ht, empty_one=False).mean())
        out["boundary_f1"] = boundary_f1(pred, tg)
        out["ap"] = average_precision(prob, tgt, valid)
    return out


def evaluate(prob: np.ndarray, tgt: np.ndarray, valid: np.ndarray, thr: float = 0.5,
             full: bool = True) -> dict:
    thr = np.asarray(thr, np.float32)
    return _score((prob > (thr[:, None, None] if thr.ndim else thr)) & valid, tgt, valid, prob, full)


def evaluate_area(prob: np.ndarray, k, tgt: np.ndarray, valid: np.ndarray,
                  full: bool = True) -> dict:
    return _score(topk_masks(prob, k) & valid, tgt, valid, prob, full)


def sweep_area(prob: np.ndarray, k, tgt: np.ndarray, valid: np.ndarray,
               gammas=None) -> tuple[float, list]:
    """Global scale on the predicted area, the counterpart of a threshold sweep."""
    gammas = np.arange(0.60, 1.46, 0.05) if gammas is None else gammas
    rows = [(float(g), float(dice_per_sample(topk_masks(prob, np.asarray(k, np.float64) * g) & valid,
                                            tgt & valid).mean())) for g in gammas]
    best = max(rows, key=lambda r: r[1])
    return best[0], rows


def sweep(prob: np.ndarray, tgt: np.ndarray, valid: np.ndarray, thrs=None) -> tuple[float, list]:
    thrs = np.arange(0.20, 0.86, 0.05) if thrs is None else thrs
    rows = [(float(t), float(dice_per_sample((prob > t) & valid, tgt & valid).mean())) for t in thrs]
    best = max(rows, key=lambda r: r[1])
    return best[0], rows
