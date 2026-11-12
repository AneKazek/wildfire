"""TFRecord -> the 23-channel float16 cache the trainers read.

The shards store 14 raw float fields; the old `data/processed` cache was built
from them by a pipeline that is not in this repo, so this rebuild re-derives the
23 channels and calibrates the per-channel scales against the statistics that
were recorded from the old cache (`OLD_STATS` below). Channels 0-10 must land on
those numbers; the rest use the derivations measured earlier (fire_dist =
clip(signed EDT * 0.1, -0.3, 1), fire_front = 1-px dilation of prev_fire).

Usage (Colab):
    python -m wf.build --shards 10 --out data/processed
    python -m wf.build --calibrate --shards 2      # stats only, no files written
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

RAW = ["elevation", "population", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx",
       "vs", "erc", "PrevFireMask", "waterOccurrence", "FireMask"]

# mean/sd/min/max/nonzero of channels 0-10, measured on the old cache (3000 samples).
OLD_STATS = {
    "ndvi": (0.601, 0.219, -0.200, 1.000, 0.99),
    "elevation": (0.404, 0.639, 0.000, 4.121, 0.98),
    "slope": (0.025, 0.053, 0.000, 0.780, 0.94),
    "aspect_sin": (0.032, 0.670, None, None, 0.88),
    "aspect_cos": (-0.060, 0.698, None, None, 0.88),
    "population": (0.177, 0.193, 0.000, 1.082, 0.78),
    "water": (0.152, 0.276, 0.000, 1.000, 0.41),
    "th_sin": (-0.101, 0.714, None, None, 0.99),
    "th_cos": (-0.368, 0.587, None, None, 1.00),
    "wind_speed": (0.344, 0.126, 0.000, 3.000, 0.99),
    "wind_u": (-0.037, 0.261, -2.992, 1.647, 0.99),
}

# Scales fitted by matching the old cache's channel statistics on 3k raw shards:
#   water        clip(w,0,100)/100      -> mean 0.152 sd 0.276 max 1.0 (old 0.152/0.276/1.0)
#   population   log10(1+pop)/4.46      -> mean 0.173 sd 0.188 max 1.082 (old 0.177/0.193/1.082)
#   wind_speed   vs*0.1 (clip 0..30)    -> mean 0.344 sd 0.126 max 3.0  (old 0.344/0.126/3.0)
#   elevation    m/1000                 -> mean 0.404 sd 0.638            (old 0.404/0.639)
NDVI_SCALE = 1e4
ELEV_SCALE = 1000.0
POP_LOG = 4.46
WATER_SCALE = 100.0
WIND_SCALE = 0.1
SPACING = 1000.0  # metres between patch pixels
SLOPE_MAX = 0.78  # the old cache capped here


def _clean(ex: dict) -> dict:
    """Kill the int16/projection sentinels the raw export contains."""
    c = dict(ex)
    c["NDVI"] = np.clip(ex["NDVI"], -2000, NDVI_SCALE)
    c["elevation"] = np.clip(ex["elevation"], 0, 5000)
    c["th"] = np.where((ex["th"] >= 0) & (ex["th"] <= 360), ex["th"], 0.0)
    c["vs"] = np.clip(ex["vs"], 0, 30)
    c["waterOccurrence"] = np.clip(ex["waterOccurrence"], 0, WATER_SCALE)
    c["tmmx"] = np.clip(ex["tmmx"], 200, 350)
    c["tmmn"] = np.clip(ex["tmmn"], 190, 340)
    c["sph"] = np.clip(ex["sph"], 0, 0.06)
    c["pr"] = np.clip(ex["pr"], 0, 100)
    c["pdsi"] = np.clip(ex["pdsi"], -10, 10)
    c["erc"] = np.clip(ex["erc"], 0, 150)
    return c


def _varint(b: bytes, i: int):
    shift = val = 0
    while True:
        x = b[i]
        i += 1
        val |= (x & 0x7F) << shift
        if not x & 0x80:
            return val, i
        shift += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        tag, i = _varint(b, i)
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"wire type {wt}")
        yield fno, wt, v


def read_shard(path: Path):
    """Yield {field: 64x64 float32} per example. 1 GFLOP-free, no tensorflow."""
    with path.open("rb", buffering=1 << 20) as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (ln,) = struct.unpack("<Q", head)
            f.read(4)
            payload = f.read(ln)
            f.read(4)
            if len(payload) < ln:
                return
            for fno, _, feats in _fields(payload):
                if fno != 1:
                    continue
                ex = {}
                for _, _, entry in _fields(feats):
                    key, raw = None, None
                    for kfno, _, kf in _fields(entry):
                        if kfno == 1:
                            key = kf.decode()
                        elif kfno == 2:
                            for g, _, gv in _fields(kf):
                                if g == 2:
                                    raw = gv
                    if key in RAW and raw is not None:
                        ex[key] = np.frombuffer(raw[:4096 * 4], "<f4").reshape(64, 64)
                if len(ex) == len(RAW):
                    yield ex


def derive(ex: dict) -> np.ndarray:
    """14 raw fields -> 23 model channels (float32)."""
    ex = _clean(ex)
    elev_m = ex["elevation"]
    gy, gx = np.gradient(elev_m / SPACING)
    th = np.deg2rad(ex["th"])
    vs, r = ex["vs"], np.hypot(gx, gy)
    prev = (ex["PrevFireMask"] >= 7).astype(np.float32)
    inside = ndi.distance_transform_edt(prev > 0.5)
    outside = ndi.distance_transform_edt(prev <= 0.5)
    dist = np.clip(np.where(prev > 0.5, -inside, outside) * 0.1, -0.3, 1.0)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    ang = np.arctan2(gy, gx)
    x = np.stack([
        ex["NDVI"] / NDVI_SCALE,
        elev_m / ELEV_SCALE,
        np.minimum(r, SLOPE_MAX),
        np.sin(ang),
        np.cos(ang),
        np.log10(1.0 + ex["population"]) / POP_LOG,
        ex["waterOccurrence"] / WATER_SCALE,
        np.sin(th),
        np.cos(th),
        vs * WIND_SCALE,
        vs * WIND_SCALE * np.sin(th),
        vs * WIND_SCALE * np.cos(th),
        ex["tmmx"], ex["tmmn"], ex["sph"], ex["pr"], ex["pdsi"], ex["erc"],
        prev,
        ndi.binary_dilation(prev > 0.5).astype(np.float32),
        dist,
        (xx - 31.5) / 31.5,
        (yy - 31.5) / 31.5,
    ]).astype(np.float32)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/next-day-wildfire-spread")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--shards", type=int, default=0, help="train shards to use (0 = all)")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    train = sorted((root / "train").rglob("*.tfrecord"))
    ev = sorted((root / "eval").rglob("*.tfrecord"))
    splits = {"train": train[: args.shards or None],
              # a small --shards build must still fit its split ratios
              "eval": ev[: max(2, args.shards // 4)] if args.shards else ev,
              "test": sorted((root / "test").rglob("*.tfrecord"))}
    if args.calibrate:
        splits.pop("test")  # 3k examples are enough to fit the scales

    acc = {k: [] for k in OLD_STATS}
    for split, files in splits.items():
        xs, fires, prevs = [], [], []
        for f in files:
            for ex in read_shard(f):
                if args.calibrate:
                    if len(acc["ndvi"]) < 3000:
                        x = derive(ex)
                        for i, name in enumerate(["ndvi", "elevation", "slope", "aspect_sin", "aspect_cos",
                                                  "population", "water", "th_sin", "th_cos", "wind_speed",
                                                  "wind_u"]):
                            acc[name].append(x[i])
                    continue
                xs.append(derive(ex).astype(np.float16))
                fires.append(ex["FireMask"].astype(np.uint8))
                prevs.append(ex["PrevFireMask"].astype(np.uint8))
                if len(xs) >= 20000:
                    _flush(args.out, split, xs, fires, prevs)
                    xs, fires, prevs = [], [], []
            print(f"{split} {f.name}: total {_count(args.out, split) + len(xs)}", flush=True)
        if xs:
            _flush(args.out, split, xs, fires, prevs)

    if args.calibrate:
        for name, old in OLD_STATS.items():
            a = np.stack(acc[name])
            print(f"{name:12s} mean {a.mean():7.3f} (old {old[0]:7.3f})  sd {a.std():6.3f} (old {old[1]:6.3f})  "
                  f"min {a.min():7.3f}  max {a.max():7.3f} (old {old[3] if old[3] is not None else '?'})")
    else:
        _write_norm(args.out)


def _flush(out, split: str, xs, fires, prevs) -> None:
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    n0 = _count(out, split)
    with (d / "features.f16").open("ab") as f:
        f.write(np.stack(xs).astype(np.float16).tobytes())
    with (d / "fire_class.u8").open("ab") as f:
        f.write(np.stack(fires).astype(np.uint8).tobytes())
    with (d / "prev_class.u8").open("ab") as f:
        f.write(np.stack(prevs).astype(np.uint8).tobytes())
    print(f"  wrote {len(xs)} rows -> {n0 + len(xs)}", flush=True)


def _count(out, split: str) -> int:
    p = Path(out) / split / "fire_class.u8"
    return p.stat().st_size // (64 * 64) if p.exists() else 0


def _write_norm(out) -> None:
    from .data import cache

    x = cache(out, "train")[0]
    idx = np.random.default_rng(0).choice(len(x), min(20000, len(x)), replace=False)
    block = np.stack([np.asarray(x[i], dtype=np.float32) for i in idx])
    mean = block.mean((0, 2, 3)).tolist()
    std = np.maximum(block.std((0, 2, 3)), 1e-3).tolist()
    (Path(out) / "norm.json").write_text(json.dumps({"mean": mean, "std": std}, indent=1))
    print("norm.json written", flush=True)


if __name__ == "__main__":
    main()
