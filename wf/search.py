"""Search driver: runs candidates in ORDER, skips finished ones, writes a CSV row each."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

from .candidates import ORDER, make_cfg
from .train import run

COLS = ["name", "stage", "params", "best_epoch", "best_thr", "macro_dice", "micro_dice", "iou",
        "precision", "recall", "ap", "boundary_f1", "hard_dice", "minutes", "error"]


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {r["name"] for r in csv.DictReader(f)}


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma separated candidate names")
    ap.add_argument("--deadline", type=float, default=0.0, help="minutes for this whole invocation")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--runs", default="results/runs")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else ORDER
    csv_path = Path(args.csv or f"results/search_s{args.stage}.csv")
    done = _done(csv_path)
    t0 = time.time()
    for name in names:
        if name in done:
            print("skip", name, flush=True)
            continue
        if args.deadline and (time.time() - t0) / 60 > args.deadline:
            print("deadline reached, stopping", flush=True)
            break
        cfg = make_cfg(name, args.stage)
        print("START", name, json.dumps(cfg["model"]), flush=True)
        row = {"name": name, "stage": args.stage}
        try:
            s = run(cfg, f"{name}_s{args.stage}", args.runs)
            v = s["val_at_best_thr"]
            row.update(params=s["params"], best_epoch=s["best_epoch"], best_thr=s["best_thr"],
                       macro_dice=round(v["macro_dice"], 4), micro_dice=round(v["micro_dice"], 4),
                       iou=round(v["iou"], 4), precision=round(v["precision"], 4), recall=round(v["recall"], 4),
                       ap=round(v["ap"], 4), boundary_f1=round(v["boundary_f1"], 4),
                       hard_dice=round(v["hard_dice"], 4), minutes=s["minutes"])
        except Exception as exc:  # keep the campaign going
            traceback.print_exc()
            row["error"] = repr(exc)[:200]
        _append(csv_path, row)
        print("ROW", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
