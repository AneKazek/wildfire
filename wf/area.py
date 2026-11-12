"""Can the per-patch fire size be predicted at all? Ridge feasibility check.

The area head on the TD decoder learns almost nothing beyond the mask's own
probability mass (corr 0.557 vs 0.541), which says the decoder representation
has thrown the size information away. This asks the prior question: is the size
in the *inputs*? Per-patch statistics of the 23 cached channels (weather,
terrain, vegetation, previous scar) plus the mask features are regressed on
log1p(area) with a closed-form ridge, fitted on train and scored on eval.

If this cannot beat the mask's own mass, no head on this data will, and the
area route is closed.

    python -m wf.area --n 40000
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import FireDataset, cache, split_indices


def _features(x: np.ndarray) -> np.ndarray:
    """Patch-level summaries of the input channels."""
    x = np.asarray(x, np.float32)
    return np.concatenate([x.mean((1, 2)), x.std((1, 2)), x.max((1, 2))])


def _fit(A: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    A = np.hstack([A, np.ones((len(A), 1), np.float32)])
    reg = lam * np.eye(A.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    xin = cache(args.root, "train")[0]
    fire = cache(args.root, "train")[1]
    idx = split_indices(args.root, "train", args.n, seed=0)
    X, A = [], []
    for i in idx:
        X.append(_features(np.asarray(xin[i])[None])[0])
        f = np.asarray(fire[i])
        A.append(np.log1p((f >= 7).sum()))
    X, A = np.stack(X), np.stack(A)
    print(f"train {len(X)} features {X.shape[1]}  log-area mean {A.mean():.2f} sd {A.std():.2f}")

    w = _fit(X, A, args.lam)
    pred_tr = np.hstack([X, np.ones((len(X), 1), np.float32)]) @ w
    print(f"train med relerr {np.median(np.abs(np.expm1(pred_tr) / np.expm1(A) - 1)):.2f} "
          f"corr {np.corrcoef(pred_tr, A)[0, 1]:.3f}")

    ds = FireDataset(args.root, "eval", None, aux=False)
    fe, ae = [], []
    for i in range(len(ds)):
        fe.append(_features(np.asarray(ds.x[i])[None])[0])
        ae.append(np.log1p(int((np.asarray(ds.fire[i]) >= 7).sum())))
    fe, ae = np.stack(fe), np.stack(ae)
    pe = np.hstack([fe, np.ones((len(fe), 1), np.float32)]) @ w
    ke, ka = np.expm1(pe), np.expm1(ae)
    print(f"eval  med relerr {np.median(np.abs(ke / np.maximum(ka, 1) - 1)):.2f} "
          f"corr {np.corrcoef(pe, ae)[0, 1]:.3f}  mean k {ke.mean():.1f} vs area {ka.mean():.1f}")


if __name__ == "__main__":
    main()
