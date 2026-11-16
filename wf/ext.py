"""External daily weather per cached row, for late fusion.

The cache's weather block already comes from the FirePred (GRIDMET) bands, so the
only honest way to test "does more weather data help" is a different observation
network. Source: Kaggle `dianatofficial/global-daily-weather-1950-2026-analysis-layer`
(GHCN-Daily core network, one wide parquet per year). That copy ships no station
coordinates, so they come from NOAA's `ghcnd-stations.txt`.

Per row we take the K nearest stations that reported on the crop's date and
combine them by inverse distance. Eight features: tmax, tmin, prcp, 3-day prcp,
tmax anomaly vs its own trailing week, diurnal range, nearest-station distance,
and a missing flag. Scaled with train statistics and written as `ext.npy`
alongside each split, aligned row-for-row with `meta.csv` from `wf.tsf`.

Usage (Colab):
    python -m wf.ext --ghcn x_ghcn --stations ghcnd-stations.txt --out data/tsf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K = 4              # nearest reporting stations
MAX_KM = 150.0     # beyond this a row counts as missing
COLS = ["tmax", "tmin", "prcp", "prcp_3d", "tmax_anom7", "dtr", "nearest_km", "missing"]


def _station_coords(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix in (".parquet", ".csv"):
        d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        names = {c.lower(): c for c in d.columns}
        lat = next((names[k] for k in ("latitude", "lat") if k in names), None)
        lon = next((names[k] for k in ("longitude", "lon", "lng") if k in names), None)
        station = next((names[k] for k in ("station_id", "id", "station") if k in names), None)
        if not all((lat, lon, station)):
            raise ValueError(f"Station coordinate schema not recognized: {list(d.columns)}")
        d = d.dropna(subset=[lat, lon, station])
        return d[station].astype(str).tolist(), d[[lat, lon]].to_numpy(np.float64)
    ids, ll = [], []
    for line in path.read_text().splitlines():
        ids.append(line[:11])
        ll.append((float(line[12:20]), float(line[21:30])))
    return ids, np.asarray(ll, np.float64)


def _grid(ghcn: Path, years: range, ids: list[str], index: dict[str, int],
          days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """station x day matrices of the raw fields, NaN where a station did not report."""
    n = len(ids)
    mats = {c: np.full((n, len(days)), np.nan, np.float32) for c in ("tmax", "tmin", "prcp")}
    for y in years:
        f = ghcn / f"ghcnd_wide_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["station_id", "date", "tmax_c", "tmin_c", "prcp_mm"])
        st = d.station_id.map(index)
        day = (pd.to_datetime(d.date) - days[0]).dt.days
        ok = st.notna() & day.between(0, len(days) - 1)
        for name, col in (("tmax", "tmax_c"), ("tmin", "tmin_c"), ("prcp", "prcp_mm")):
            mats[name][st[ok].astype(int), day[ok].astype(int)] = d[col][ok].to_numpy(np.float32)
    return mats


def _rolling(a: np.ndarray, w: int, how: str) -> np.ndarray:
    df = pd.DataFrame(a.T).rolling(w, min_periods=1)
    return (df.sum() if how == "sum" else df.mean()).to_numpy(np.float32).T


def build(meta: pd.DataFrame, mats: dict[str, np.ndarray], ll: np.ndarray,
          days: pd.DatetimeIndex) -> np.ndarray:
    if len(ll) == 0:
        raise ValueError("No reporting weather stations overlap the crops")
    tmax, tmin, prcp = mats["tmax"], mats["tmin"], mats["prcp"]
    fields = np.stack([tmax, tmin, prcp, _rolling(prcp, 3, "sum"),
                       tmax - _rolling(tmax, 7, "mean"), tmax - tmin])
    def xyz(coords):
        lat, lon = np.deg2rad(coords).T
        return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    tree = cKDTree(xyz(ll))
    dist, nb = tree.query(xyz(np.c_[meta.lat, meta.lon]), k=min(32, len(ll)))
    # query collapses to 1-D when there is a single station
    dist, nb = dist.reshape(len(meta), -1), nb.reshape(len(meta), -1)
    day = (pd.to_datetime(meta.date.astype(str)) - days[0]).dt.days.to_numpy()
    out = np.full((len(meta), len(COLS)), np.nan, np.float32)
    for i in range(len(meta)):
        t = day[i]
        if not (0 <= t < len(days)):
            continue
        v = fields[:, nb[i], t]                    # (features, k)
        d = 2 * 6371.0 * np.arcsin(np.clip(dist[i] / 2, 0, 1))
        available = np.isfinite(v) & (d <= MAX_KM)
        available &= np.cumsum(available, axis=1) <= K
        w = np.where(available, 1.0 / np.maximum(d, 0.5), 0.0)
        s = w.sum(1)
        good = s > 0
        out[i, :6][good] = (np.nan_to_num(v) * w).sum(1)[good] / s[good]
        near = np.isfinite(v).any(0)
        out[i, 6] = d[near].min() if near.any() else MAX_KM
    out[:, 7] = ~np.isfinite(out[:, :6]).all(1)
    out[:, 6] = np.minimum(out[:, 6], MAX_KM) / 100.0
    return out


def scale_split(raw: dict[str, np.ndarray], out: Path) -> None:
    train = np.where(np.isfinite(raw["train"]), raw["train"], np.nan)
    count = np.isfinite(train).sum(0)
    mean = np.nansum(train, axis=0) / np.maximum(count, 1)
    var = np.nansum((train - mean) ** 2, axis=0) / np.maximum(count, 1)
    std = np.maximum(np.sqrt(var), 1e-6)
    for split, a in raw.items():
        z = np.nan_to_num((a - mean) / std, nan=0.0)
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "ext.npy", z.astype(np.float32))
        print(f"{split}: ext {z.shape} missing={float(np.mean(a[:, 7] > 0)):.1%}", flush=True)
    (out / "ext_norm.json").write_text(json.dumps({"cols": COLS, "mean": mean.tolist(),
                                                   "std": std.tolist()}, indent=1))
    audit = {f"{s}_complete_fraction": float(np.mean(a[:, 7] == 0)) for s, a in raw.items()}
    audit['feature_observed_fraction'] = {s: np.isfinite(a[:, :6]).mean(0).tolist() for s, a in raw.items()}
    (out / 'weather_audit.json').write_text(json.dumps(audit, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghcn", default="x_ghcn")
    ap.add_argument("--stations", default="ghcnd-stations.txt")
    ap.add_argument("--out", default="data/tsf")
    ap.add_argument("--year0", type=int, default=2017)
    ap.add_argument("--year1", type=int, default=2021)
    ap.add_argument("--splits", default="train,eval,test")
    args = ap.parse_args()

    out, ghcn = Path(args.out), Path(args.ghcn)
    splits = {s: pd.read_csv(out / s / "meta.csv") for s in args.splits.split(',')}
    meta = pd.concat(splits.values(), ignore_index=True)
    ids_all, ll_all = _station_coords(Path(args.stations))
    # only stations near our crops and present in the core network
    near = ((ll_all[:, 0] > meta.lat.min() - 2) & (ll_all[:, 0] < meta.lat.max() + 2)
            & (ll_all[:, 1] > meta.lon.min() - 2) & (ll_all[:, 1] < meta.lon.max() + 2))
    days = pd.date_range(f"{args.year0}-01-01", f"{args.year1}-12-31", freq="D")
    ids = [i for i, k in zip(ids_all, near) if k]
    ll = ll_all[near]
    if not any(ghcn.glob("ghcnd_wide_*.parquet")):
        raise FileNotFoundError(f"No ghcnd_wide_*.parquet under {ghcn}")
    mats = _grid(ghcn, range(args.year0, args.year1 + 1), ids,
                 {s: i for i, s in enumerate(ids)}, days)
    keep = np.isfinite(mats["tmax"]).any(1) | np.isfinite(mats["prcp"]).any(1)
    print(f"{len(ids)} stations in the crop bbox, {int(keep.sum())} with data", flush=True)
    mats = {k: v[keep] for k, v in mats.items()}
    ll = ll[keep]

    raw = {s: build(d, mats, ll, days) for s, d in splits.items()}
    scale_split(raw, out)


if __name__ == "__main__":
    main()
