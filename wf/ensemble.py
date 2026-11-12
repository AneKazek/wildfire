"""Blend finished runs with TTA, tune the threshold on eval, test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import athr
from .data import FireDataset
from .final import _targets, predict_tta
from .metrics import dice_per_sample, evaluate, sweep
from .train import _loader, build


def _load_run(name: str, dev, root: str):
    ckpt = torch.load(Path("results/runs") / name / "best.pt", map_location=dev)
    model, prep, drop = build(ckpt["cfg"], root)
    model = model.to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"], strict=False)
    return model, prep, drop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma separated run dir names under results/runs")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--weights", default=None, help="comma separated blend weights")
    ap.add_argument("--athr", action="store_true", help="per-sample thresholds from predicted area")
    ap.add_argument("--out", default="results/ensemble.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    root = "data/processed"
    names = args.runs.split(",")
    w = None if args.weights is None else np.array([float(x) for x in args.weights.split(",")])
    assert w is None or len(w) == len(names)

    head = torch.load(Path("results/runs") / names[0] / "best.pt", map_location="cpu")["cfg"]["data"]
    prep, drop = head.get("prep", "base"), head.get("drop", "")

    ds = FireDataset(root, args.split, None, aux=False, prep=prep, drop=drop)
    tg, va = _targets(ds)
    ld = _loader(ds, 512, False, 4)

    probs = []
    for name in names:
        cached = Path("results/runs") / name / f"{args.split}_prob.npy"
        if cached.exists():
            p = np.load(cached)
            print(name, "(cached)", flush=True)
        else:
            model, _, _ = _load_run(name, dev, root)
            p = predict_tta(model, ld, dev, dtype, args.tta)
            del model
            torch.cuda.empty_cache()
        probs.append(p)
        print(name, "macro@0.5", round(evaluate(p, tg, va, 0.5, full=False)["macro_dice"], 4), flush=True)

    blend = np.average(probs, axis=0, weights=w)
    thr, rows = sweep(blend, tg, va)
    rule = None
    if args.athr:
        rule = athr.fit(blend, tg, va)
        honest = athr.cv(blend, tg, va)["cv"]
        global_dice = float(dice_per_sample((blend > thr) & va, tg & va).mean())
        print(f"athr: global {global_dice:.4f} fit {athr.score(blend, tg, va, rule):.4f} cv {honest:.4f}",
              flush=True)
        if honest <= global_dice + 0.002:  # the gain must survive held-out fitting
            print("athr: no honest gain, keeping the global threshold", flush=True)
            rule = None
    res = {"runs": names, "split": args.split, "tta": args.tta, "thr": thr, "sweep": rows,
           "athr": rule, "eval": evaluate(blend, tg, va, athr.apply(blend, rule) if rule else thr, full=True)}

    if args.test:
        te = FireDataset(root, "test", None, aux=False, prep=prep, drop=drop)
        ttg, tva = _targets(te)
        tld = _loader(te, 512, False, 4)
        tp = []
        for name in names:
            model, _, _ = _load_run(name, dev, root)
            tp.append(predict_tta(model, tld, dev, dtype, args.tta))
            del model
            torch.cuda.empty_cache()
        tblend = np.average(tp, axis=0, weights=w)
        res["test"] = evaluate(tblend, ttg, tva, athr.apply(tblend, rule) if rule else thr, full=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("runs", "thr", "eval")}, indent=1), flush=True)
    if args.test:
        print(json.dumps({"test": res["test"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
