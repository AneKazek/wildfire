"""Per-sample thresholds fitted on eval.

A single global threshold is wrong for this data: a 3-px fire and a 300-px fire
want very different cut-offs (the eval oracle for a per-patch threshold scores
0.54 vs 0.46 for the best global one). Predicted area at 0.5 is known without
any extra inference, so bin on it and pick the best cut-off per bin.

Bin edges are fixed rather than fitted so they carry over to `test` unchanged;
only the five thresholds are estimated. Fitted on eval only, with a 2-fold check
so the reported gain is not the fit itself.
`python -m wf.athr --prob results/runs/<run>/eval_prob.npy`
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .metrics import dice_per_sample

GRID = np.arange(0.30, 0.86, 0.025)
# predicted-fire-size bins: 0-3, 4-9, 10-24, 25-59, 60+ pixels
EDGES = np.array([0.0, 4.0, 10.0, 25.0, 60.0, 1e9])
MIN_BIN = 100


def _area(prob: np.ndarray) -> np.ndarray:
    return (prob > 0.5).sum(axis=(1, 2))


def _bin(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(EDGES, _area(prob), side="right") - 1, 0, len(EDGES) - 2)


def fit(prob: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> dict:
    d = np.stack([dice_per_sample((prob > g) & valid, tgt & valid) for g in GRID], 1)  # (n, |GRID|)
    b = _bin(prob)
    g = float(GRID[int(np.argmax(d.mean(0)))])
    thr = []
    for i in range(len(EDGES) - 1):
        m = b == i
        thr.append(float(GRID[int(np.argmax(d[m].mean(0)))]) if m.sum() >= MIN_BIN else g)
    return {"edges": EDGES.tolist(), "thr": thr, "global": g, "bin_n": np.bincount(b, minlength=5).tolist()}


def apply(prob: np.ndarray, rule: dict) -> np.ndarray:
    return np.asarray(rule["thr"], np.float32)[_bin(prob)]


def score(prob, tgt, valid, rule) -> float:
    return float(dice_per_sample((prob > apply(prob, rule)[:, None, None]) & valid, tgt & valid).mean())


def cv(prob, tgt, valid, folds: int = 2, seed: int = 0) -> dict:
    """Fit on one part, score the other; the honest number for the gain."""
    idx = np.random.default_rng(seed).permutation(len(prob))
    parts = np.array_split(idx, folds)
    scores = []
    for i in range(folds):
        held = parts[i]
        rest = np.concatenate([parts[j] for j in range(folds) if j != i])
        rule = fit(prob[rest], tgt[rest], valid[rest])
        thr = apply(prob[held], rule)
        scores.append(float(dice_per_sample((prob[held] > thr[:, None, None]) & valid[held],
                                            tgt[held] & valid[held]).mean()))
    return {"cv": float(np.mean(scores)), "folds": scores}


def main() -> None:
    from .data import FireDataset
    from .metrics import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True, help="eval_prob.npy written by wf.final")
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = FireDataset(args.root, "eval", None, aux=False)
    tgt = np.stack([np.asarray(ds.fire[i]) >= 7 for i in range(len(ds))])
    valid = np.stack([~np.isin(np.asarray(ds.fire[i]), (4, 6)) for i in range(len(ds))])
    prob = np.load(args.prob)

    rule = fit(prob, tgt, valid)
    res = {"global_thr": rule["global"],
           "global_dice": float(dice_per_sample((prob > rule["global"]) & valid, tgt & valid).mean()),
           "rule": rule, "fit_dice": score(prob, tgt, valid, rule), **cv(prob, tgt, valid),
           "full": evaluate(prob, tgt, valid, apply(prob, rule))}
    print(json.dumps(res, indent=1), flush=True)
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
