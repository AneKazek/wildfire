"""Finalist: full-data training, 8-way dihedral TTA, threshold on eval, one test pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .candidates import make_cfg
from .data import PAIRS, FireDataset, jac, split_indices
from .metrics import evaluate, evaluate_area, sweep, sweep_area
from .train import _loader, build, norm_stats, run


def _geo(x: torch.Tensor, k: int) -> torch.Tensor:
    if k & 1:
        x = torch.flip(x, [-1])
    if k & 2:
        x = torch.flip(x, [-2])
    if k & 4:
        x = torch.rot90(x, 1, dims=(-2, -1))
    return x


def _vec(x: torch.Tensor, k: int) -> torch.Tensor:
    J = torch.tensor(jac(k), device=x.device, dtype=x.dtype)
    x = x.clone()
    for u, v in PAIRS:
        uu, vv = x[:, u].clone(), x[:, v].clone()
        x[:, u] = J[0, 0] * uu + J[0, 1] * vv
        x[:, v] = J[1, 0] * uu + J[1, 1] * vv
    return x


def _inv(p: torch.Tensor, k: int) -> torch.Tensor:
    if k & 4:
        p = torch.rot90(p, -1, dims=(-2, -1))
    if k & 2:
        p = torch.flip(p, [-2])
    if k & 1:
        p = torch.flip(p, [-1])
    return p


@torch.no_grad()
def predict_tta(model, loader, device, dtype, tta: int = 8):
    """Mean probability over the dihedral group, plus the area prediction if the
    model has an area head (invariant under the group, so the plain mean)."""
    model.eval()
    ks = list(range(tta)) if tta > 1 else [0]
    probs = [None] * len(ks)
    areas = []
    for i, k in enumerate(ks):
        acc = []
        for x, _, _, _ in loader:
            x = x.to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            xk = _geo(x, k)
            if k:
                xk = _vec(xk, k)
            xk = xk.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                o = model(xk)
            acc.append(_inv(torch.sigmoid(o["main"].float()), k).cpu().numpy()[:, 0])
            if i == 0 and "area" in o:
                areas.append(o["area"].float().cpu().numpy())
        probs[i] = np.concatenate(acc)
    area = np.concatenate(areas) if areas else None
    return np.mean(probs, axis=0), area


def _targets(ds):
    tgt = np.stack([np.asarray(ds.fire[i]) >= 7 for i in range(len(ds))])
    val = np.stack([~np.isin(np.asarray(ds.fire[i]), (4, 6)) for i in range(len(ds))])
    return tgt, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to fine-tune from, e.g. a stage-1 best.pt")
    ap.add_argument("--tag", default="", help="suffix for the run dir, e.g. _seed1")
    ap.add_argument("--lr", type=float, default=0.0, help="override the protocol lr (fine-tuning)")
    ap.add_argument("--epochs", type=int, default=0, help="override the protocol epochs (fine-tuning)")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    root = "data/processed"
    cfg = make_cfg(args.name, args.stage)
    cfg["seed"] = args.seed
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.epochs:
        cfg["train"].update(epochs=args.epochs, patience=args.epochs)
    run_name = f"{args.name}_s{args.stage}{args.tag}"
    out = Path(args.runs) / run_name

    if not args.skip_train:
        run(cfg, run_name, args.runs, resume_from=args.resume)

    ckpt = torch.load(out / "best.pt", map_location=dev)
    model, prep, drop = build(cfg, root)
    model = model.to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"], strict=False)

    ev_ds = FireDataset(root, "eval", None, aux=False, prep=prep, drop=drop)
    tg, va = _targets(ev_ds)
    ld = _loader(ev_ds, 512, False, 4)
    prob, ar = predict_tta(model, ld, dev, dtype, args.tta)
    if ar is not None:
        thr, rows = sweep_area(prob, np.expm1(ar), tg, va)
        ev = evaluate_area(prob, np.expm1(ar) * thr, tg, va, full=True)
        res_decode = "area"
    else:
        thr, rows = sweep(prob, tg, va)
        ev = evaluate(prob, tg, va, thr, full=True)
        res_decode = "thr"
    res = {"name": args.name, "run": run_name, "seed": args.seed, "stage": args.stage, "tta": args.tta,
           "decode": res_decode, "thr": thr, "sweep": rows, "eval": ev, "prep": prep, "drop": drop}
    np.save(out / "eval_prob.npy", prob)
    if ar is not None:
        np.save(out / "eval_area.npy", ar)

    if args.test:
        te_ds = FireDataset(root, "test", None, aux=False, prep=prep, drop=drop)
        ttg, tva = _targets(te_ds)
        tprob, tar = predict_tta(model, _loader(te_ds, 512, False, 4), dev, dtype, args.tta)
        res["test"] = (evaluate_area(tprob, np.expm1(tar) * thr, ttg, tva, full=True) if tar is not None
                       else evaluate(tprob, ttg, tva, thr, full=True))

    (out / "final.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("name", "thr", "eval")}, indent=1), flush=True)
    if args.test:
        print(json.dumps({"test": res["test"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
