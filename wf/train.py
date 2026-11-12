"""One training run: config dict in, history + best checkpoint out."""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .data import CHANNELS, FireDataset, split_indices
from .losses import make_loss, moment
from .metrics import evaluate, evaluate_area, sweep, sweep_area
from .models import build_model
from .prep import norm_for


def build(cfg: dict, root: str | Path = "data/processed"):
    """Model + the data-preprocessing options it was trained with."""
    d = cfg["data"]
    prep, drop = d.get("prep", "base"), d.get("drop", "")
    mean, std = norm_for(root, prep, drop)
    model = build_model({**cfg["model"], "cin": len(mean)}, mean, std)
    return model, prep, drop


def norm_stats(root: str | Path, n: int = 20000, seed: int = 0) -> tuple[list[float], list[float]]:
    path = Path(root) / "norm.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d["mean"], d["std"]
    from .data import cache

    x = cache(root, "train")[0]
    idx = np.random.default_rng(seed).integers(0, len(x), min(n, len(x)))
    blocks = np.stack([np.asarray(x[i], dtype=np.float32) for i in idx])
    mean = blocks.mean(axis=(0, 2, 3)).tolist()
    std = np.maximum(blocks.std(axis=(0, 2, 3)), 1e-3).tolist()
    path.write_text(json.dumps({"channels": CHANNELS, "mean": mean, "std": std}, indent=1))
    return mean, std


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int = 10**9) -> None:
        # Adaptive warmup: with a fixed 0.999 the shadow is still the init
        # weights after a 12-epoch screen, which made short runs look dead.
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup)


def _loader(ds, bs: int, shuffle: bool, workers: int) -> DataLoader:
    # Persistent workers: re-spawning 10 forks every epoch cost ~25 s per epoch,
    # which dominated short screening runs.
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=False,
                      drop_last=shuffle, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers else None)


@torch.no_grad()
def predict(model, loader, device, dtype):
    """Probabilities plus, when the model has one, the predicted log1p(area)."""
    model.eval()
    probs, tgts, valids, areas = [], [], [], []
    for x, y, v, _ in loader:
        x = x.to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            o = model(x)
        probs.append(torch.sigmoid(o["main"].float()).cpu().numpy()[:, 0])
        if "area" in o:
            areas.append(o["area"].float().cpu().numpy())
        tgts.append(y.numpy() > 0.5)
        valids.append(v.numpy() > 0.5)
    area = np.concatenate(areas) if areas else None
    return np.concatenate(probs), np.concatenate(tgts), np.concatenate(valids), area


def run(cfg: dict, run_name: str | None = None, out_dir: str | Path = "results/runs",
        resume_from: str | Path | None = None) -> dict:
    rng = int(cfg.get("seed", 0))
    torch.manual_seed(rng)
    np.random.seed(rng)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    d, m, l, t = cfg["data"], cfg["model"], cfg["loss"], cfg["train"]
    root = d.get("root", "data/processed")
    name = run_name or cfg.get("name", "run")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if t.get("amp", "bf16") in (True, "bf16") else torch.float16
    use_amp = dev.type == "cuda" and bool(t.get("amp", True))

    model, prep, drop = build(cfg, root)
    model = model.to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    train_idx = split_indices(root, "train", t.get("n_train"), seed=rng)
    criterion = make_loss(l)
    wb, wg = l.get("boundary_weight", 0.0), l.get("growth_weight", 0.0)
    wm = l.get("moment_weight", 0.0)
    wa = l.get("area_weight", 0.0)
    # Fork + copy-on-write makes an in-RAM copy per worker: a 100k subset is
    # 3.8 GB, i.e. ~38 GB across 10 workers, which the cgroup OOM-kills.
    in_mem = d.get("in_memory")
    if in_mem is None:
        in_mem = len(train_idx) <= 30000
    tr_ds = FireDataset(root, "train", train_idx, augment=True, prev_dropout=d.get("prev_dropout", 0.2),
                        seed=rng, aux=bool(wb or wg), in_memory=in_mem, prep=prep, drop=drop,
                        pf_aug=d.get("pf_aug", ""), soft=float(d.get("soft", 0.0)))
    ev_idx = split_indices(root, "eval", t.get("eval_n"), seed=0)
    ev_ds = FireDataset(root, "eval", ev_idx, augment=False, aux=False, in_memory=len(ev_idx) <= 30000,
                        prep=prep, drop=drop)
    bs = int(t.get("batch_size", 256))
    tr_ld = _loader(tr_ds, bs, True, int(t.get("workers", 8)))
    ev_ld = _loader(ev_ds, 512, False, int(t.get("eval_workers", 4)))
    print(f"RUN {name} params={nparam/1e6:.2f}M n_train={len(train_idx)} n_eval={len(ev_idx)} "
          f"epochs={int(t.get('epochs', 10))} bs={bs}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 2e-3)),
                            weight_decay=float(t.get("weight_decay", 1e-4)))
    epochs = int(t.get("epochs", 10))
    warm = int(t.get("warmup_epochs", 1))
    steps_per_epoch = max(1, len(tr_ds) // bs)
    total_steps = epochs * steps_per_epoch
    warm_steps = warm * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warm_steps:
            return (step + 1) / max(warm_steps, 1)
        p = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(model, float(t.get("ema_decay", 0.999)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype is torch.float16)
    clip = float(t.get("grad_clip", 1.0))

    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    best = {"macro_dice": -1.0, "epoch": -1}
    history = []
    t_start = time.time()
    step = 0
    patience = int(t.get("patience", epochs))

    for epoch in range(epochs):
        model.train()
        t0, seen, losses, t_data = time.time(), 0, [], 0.0
        it = iter(tr_ld)
        for _ in range(steps_per_epoch):
            t_wait = time.time()
            x, y, v, aux = next(it)
            t_data += time.time() - t_wait
            x = x.to(dev, non_blocking=True).float().to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True).unsqueeze(1)
            v = v.to(dev, non_blocking=True).unsqueeze(1)
            aux = aux.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                o = model(x)
                loss = criterion(o["main"].float(), y, v)
                if wm:
                    loss = loss + wm * moment(o["main"].float(), y, v)
                if wa and "area" in o:
                    # L1 on log area = relative error; a 0.3 loss is ~30% off.
                    at = torch.log1p((y * v).sum(dim=(1, 2, 3)))
                    loss = loss + wa * F.l1_loss(o["area"].float(), at)
                if wb and "boundary" in o:
                    loss = loss + wb * criterion(o["boundary"].float(), aux[:, 0:1], v)
                if wg and "growth" in o:
                    loss = loss + wg * criterion(o["growth"].float(), aux[:, 1:2], v)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model, step)
            losses.append(float(loss.detach()))
            seen += x.shape[0]
            step += 1
        dt = time.time() - t0
        t_eval = time.time()
        with ema.apply(model):
            prob, tg, va, ar = predict(model, ev_ld, dev, dtype)
        mt = (evaluate_area(prob, np.expm1(ar), tg, va, full=False) if ar is not None
              else evaluate(prob, tg, va, 0.5, full=False))
        t_eval = time.time() - t_eval
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "train_sps": int(seen / dt),
               "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in mt.items()},
               "eval_s": round(t_eval, 1), "data_s": round(t_data, 1), "loop_s": round(dt, 1),
               "minutes": round((time.time() - t_start) / 60, 1)}
        history.append(row)
        if mt["macro_dice"] > best["macro_dice"]:
            best = {**mt, "epoch": epoch, "thr": 0.5}
            t_save = time.time()
            torch.save({"model": ema.shadow, "cfg": cfg, "metrics": best, "epoch": epoch,
                        "params": nparam}, out / "best.pt")
            row["save_s"] = round(time.time() - t_save, 1)
        print(json.dumps(row), flush=True)
        if epoch + 1 - best["epoch"] >= patience and best["epoch"] >= 0:
            print("early stop", flush=True)
            break
        if t.get("abort_below") and epoch + 1 >= int(t.get("abort_after", 3)) \
                and best["macro_dice"] < float(t["abort_below"]):
            print("abort: below bar", flush=True)
            break
        if t.get("max_seconds") and time.time() - t_start > float(t["max_seconds"]):
            print("time budget reached", flush=True)
            break

    del tr_ld, ev_ld
    ckpt = out / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"], strict=False)
    full_ds = FireDataset(root, "eval", None, aux=False, prep=prep, drop=drop)
    prob, tg, va, ar = predict(model, _loader(full_ds, 512, False, 4), dev, dtype)
    if ar is not None:
        thr, rows = sweep_area(prob, np.expm1(ar), tg, va)
        full = evaluate_area(prob, np.expm1(ar) * thr, tg, va, full=True)
        decode = "area"
    else:
        thr, rows = sweep(prob, tg, va)
        full = evaluate(prob, tg, va, thr, full=True)
        decode = "thr"
    summary = {"name": name, "params": nparam, "best_epoch": best["epoch"], "best_thr": thr,
               "decode": decode, "best_macro_dice": best["macro_dice"], "val_at_best_thr": full,
               "minutes": round((time.time() - t_start) / 60, 1), "history": history, "sweep": rows,
               "cfg": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("FINAL", json.dumps({k: summary[k] for k in ("name", "params", "best_epoch", "best_thr", "minutes")}),
          json.dumps(full), flush=True)
    return summary
