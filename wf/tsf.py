"""TS-SatFire (HuggingFace `SamuelWu318/ts-satfire`) -> the float16 patch cache.

Per event (ROI) the dataset stores one directory per day:

  VIIRS_Day    8 bands   I1 I2 I3 I4 I5 M11, active-fire temperature (NaN off fire),
                         burned-area product (>0 where burned)
  VIIRS_Night  2 bands   night I4, I5
  FirePred    19 bands   NDVI_last, EVI2_last, precip, wind speed/dir, tmin, tmax,
                         ERC, sph, slope, aspect, elevation, PDSI, LC_Type1,
                         precip sum, forecast wind speed/dir, forecast temp/sph

None of the tifs carry a usable transform (scale 0), only matching bounds, so
layers are resampled onto the day grid by bounds and then cropped.

Target: next-day *progression*, i.e. the pixels that are burned in the
cumulative mask of day t+1 but not in that of day t — the label TS-SatFire
defines for this task. Crops are centred on progression pixels, so every row
has fire, exactly like the Kaggle cache this repo was built on.

Usage (Colab):
    python -m wf.tsf --download                 # ~40 GB, parallel
    python -m wf.tsf --out data/tsf --per-day 16
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = "SamuelWu318/ts-satfire"
CACHE = "/content/tsf"
PATTERNS = [f"ts-satfire/*/{d}/*.tif" for d in ("VIIRS_Day", "VIIRS_Night", "FirePred")] + ["roi/*.csv"]

# official split (dataset_gen_pred.py): 2017-2020 fires minus these ids, 2021 held out
VAL = {"20568194", "20701026", "20562846", "20700973", "24462610", "24462788", "24462753",
       "24103571", "21998313", "21751303", "22141596", "21999381", "23301962", "22712904",
       "22713339"}

# aux band index -> our channel (0-based within the 19)
AUX = dict(ndvi=0, evi2=1, pr=2, ws=3, wdir=4, tmin=5, tmax=6, erc=7, sph=8, slope=9,
           aspect=10, elev=11, pdsi=12, lulc=13, pr_sum=14, fws=15, fdir=16, ftmp=17, fsph=18)
CLIP = {"pr": (0, 100), "pr_sum": (0, 300), "ws": (0, 60), "fws": (0, 60), "tmin": (180, 340),
        "tmax": (180, 350), "ftmp": (0, 60), "erc": (0, 150), "sph": (0, 0.06),
        "fsph": (0, 0.06), "slope": (0, 60), "elev": (0, 5000), "pdsi": (-10, 10),
        "lulc": (1, 17)}  # slope is degrees, ftmp is Celsius in this copy
SIZE = 64
STRIDE = 16  # centre-bucket size when picking distinct crops, in pixels


def _read(path: Path):
    import rasterio as rio  # only the converter needs it; the trainers do not

    with rio.open(path) as r:
        return r.read().astype(np.float32), r.bounds, r.shape


def _on_grid(a, src_shape, dst_shape):
    """Resample a layer onto the day grid. The layers share bounds, so a plain
    resize is the right transform; the tifs carry no usable affine to warp with."""
    if src_shape == dst_shape:
        return a
    zy, zx = dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1]
    out = np.stack([ndi.zoom(b, (zy, zx), order=1) for b in a])
    padded = np.zeros((a.shape[0], *dst_shape), np.float32)
    h, w = min(out.shape[1], dst_shape[0]), min(out.shape[2], dst_shape[1])
    padded[:, :h, :w] = out[:, :h, :w]
    return padded


def _fill(a: np.ndarray) -> np.ndarray:
    """NaN-safe per-band fill: the masked-off pixels are a big fraction of a layer."""
    out = np.empty_like(a)
    for i, b in enumerate(a):
        ok = np.isfinite(b)
        out[i] = np.where(ok, b, np.median(b[ok]) if ok.any() else 0.0)
    return out


def _band(a: np.ndarray, name: str) -> np.ndarray:
    b = a[AUX[name]]
    lo, hi = CLIP.get(name, (-1e9, 1e9))
    return np.clip(b, lo, hi)


def _sincos(deg: np.ndarray):
    t = np.deg2rad(deg)
    return np.sin(t).astype(np.float32), np.cos(t).astype(np.float32)


def _channels(day, night, aux, prev, box):
    """One 33 x SIZE x SIZE patch. `prev` is the cumulative fire mask of day t."""
    r0, c0 = box
    sl = (slice(r0, r0 + SIZE), slice(c0, c0 + SIZE))
    d = _fill(day)[:, sl[0], sl[1]]
    n = _fill(night)[:, sl[0], sl[1]]
    a = _fill(aux)[:, sl[0], sl[1]]
    p = prev[sl].astype(np.float32)

    i1, i2, i3 = d[0] / 100.0, d[1] / 100.0, d[2] / 100.0
    ndvi = (i2 - i1) / (i2 + i1 + 1e-3)
    evi2 = 2.5 * (i2 - i1) / (i2 + 2.4 * i1 + 1.0)
    asp_s, asp_c = _sincos(_band(a, "aspect"))
    wd_s, wd_c = _sincos(_band(a, "wdir"))
    fd_s, fd_c = _sincos(_band(a, "fdir"))
    ws = _band(a, "ws")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    inside = ndi.distance_transform_edt(p > 0.5)
    outside = ndi.distance_transform_edt(p <= 0.5)
    dist = np.clip(np.where(p > 0.5, -inside, outside) * 0.1, -0.3, 1.0)
    return np.stack([
        d[0], d[1], d[2], d[3], d[4], d[5], n[0], n[1],
        ndvi, evi2, _band(a, "pr_sum"), ws, wd_s, wd_c,
        _band(a, "tmin"), _band(a, "tmax"), _band(a, "erc"), _band(a, "sph"),
        _band(a, "slope"), asp_s, asp_c, _band(a, "elev"), _band(a, "pdsi"), _band(a, "lulc"),
        _band(a, "fws"), fd_s, fd_c, _band(a, "ftmp"),
        p, ndi.binary_dilation(p > 0.5).astype(np.float32), dist,
        (xx - 31.5) / 31.5, (yy - 31.5) / 31.5,
    ]).astype(np.float32)


def _centres(new: np.ndarray, k: int, rng) -> list[tuple[int, int]]:
    """Up to k crop origins on distinct STRIDExSTRIDE buckets of progression pixels."""
    ys, xs = np.nonzero(new)
    if len(ys) == 0:
        return []
    order = rng.permutation(len(ys))
    seen, out = set(), []
    for i in order:
        by, bx = ys[i] // STRIDE, xs[i] // STRIDE
        if (by, bx) in seen:
            continue
        seen.add((by, bx))
        out.append((int(np.clip(ys[i] - SIZE // 2, 0, new.shape[0] - SIZE)),
                    int(np.clip(xs[i] - SIZE // 2, 0, new.shape[1] - SIZE))))
        if len(out) >= k:
            break
    return out


def _by_date(d: Path) -> dict[str, Path]:
    return {p.name.split("_")[0]: p for p in d.glob("*.tif")}


def convert(roi_dir: Path, per_day: int, rng) -> tuple[list, list, list]:
    days = sorted((roi_dir / "VIIRS_Day").glob("*.tif"))
    nights = _by_date(roi_dir / "VIIRS_Night")
    preds = _by_date(roi_dir / "FirePred")
    days = [p for p in days if p.name.split("_")[0] in preds]
    if len(days) < 3:
        return [], [], []
    xs, fires, prevs = [], [], []
    cum = None
    cur = None
    for i in range(len(days) - 1):
        f0, f1 = days[i], days[i + 1]
        day0, b, shape = _read(f0) if cur is None else cur
        day1, b1, s1 = _read(f1)
        m1 = (~np.isnan(day1[6])) | (np.nan_to_num(day1[7]) > 0)
        nxt = m1 if cum is None else (cum | m1)
        new = nxt if cum is None else (nxt & ~cum)
        cur = (day1, b1, s1)
        if new.sum() < 8:
            cum = nxt
            continue
        key = f0.name.split("_")[0]
        night = nights.get(key)
        if night is None:
            night = np.zeros((2, *shape), np.float32)
        else:
            night, nb, ns = _read(night)
            night = night[-2:] if night.shape[0] > 2 else night
            night = _on_grid(night, ns, shape)
        aux = _read(preds[key])
        aux = _on_grid(aux[0], aux[2], shape)
        prev = np.zeros(shape, bool) if cum is None else cum
        for box in _centres(new, per_day, rng):
            r0, c0 = box
            sl = (slice(r0, r0 + SIZE), slice(c0, c0 + SIZE))
            xs.append(_channels(day0, night, aux, prev, box))
            fires.append(np.where(new[sl], 7, 5).astype(np.uint8))
            prevs.append(np.where(prev[sl], 7, 5).astype(np.uint8))
        cum = nxt
    return xs, fires, prevs


def _split_ids(roi_csv: Path, cache: Path) -> dict[str, list[str]]:
    def ids(name):
        with (roi_csv / name).open(newline="") as f:
            return [r["Id"] for r in csv.DictReader(f)]

    test = ids("fire_all_test.csv")
    # every ROI on disk that is not held out, so the fires missing from the csv
    # lists (the card says 179 events, the lists cover 162) still train
    on_disk = sorted(p.name for p in (cache / "ts-satfire").iterdir() if p.is_dir())
    train = [i for i in on_disk if i not in VAL and i not in set(test)]
    return {"train": train, "eval": sorted(VAL), "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch every VIIRS/FirePred tif (~40 GB)")
    ap.add_argument("--out", default="data/tsf")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--roi-csv", default=None, help="dir with fire_all.csv (default <cache>/roi)")
    ap.add_argument("--per-day", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only this many ROIs per split")
    ap.add_argument("--keep", action="store_true", help="do not delete tifs after converting")
    args = ap.parse_args()

    cache = Path(args.cache)
    if args.download or not (cache / "roi").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, repo_type="dataset", local_dir=str(cache),
                          allow_patterns=PATTERNS, max_workers=16)
        print("download complete", flush=True)
        if args.download:
            return

    from .build import _count, _flush, _write_norm

    rng = np.random.default_rng(0)
    splits = _split_ids(Path(args.roi_csv or cache / "roi"), cache)
    for split, ids in splits.items():
        if args.limit:
            ids = ids[: args.limit]
        n0 = _count(args.out, split)
        for roi in ids:
            d = cache / "ts-satfire" / roi
            if not (d / "VIIRS_Day").exists():
                print("missing", split, roi, flush=True)
                continue
            try:
                xs, fires, prevs = convert(d, args.per_day, rng)
            except Exception as exc:  # a truncated tif should cost one ROI, not the run
                print("fail", split, roi, repr(exc)[:140], flush=True)
                continue
            if xs:
                _flush(args.out, split, xs, fires, prevs)
            if not args.keep:
                shutil.rmtree(d, ignore_errors=True)
            print(f"{split} {roi}: {len(xs)} rows, total {_count(args.out, split)}", flush=True)
        print(f"{split}: {_count(args.out, split) - n0} rows from {len(ids)} ROIs", flush=True)

    _write_norm(args.out)


if __name__ == "__main__":
    main()
