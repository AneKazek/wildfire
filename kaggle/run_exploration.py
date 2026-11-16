"""Run from the repository directory INSIDE Kaggle. No local training."""
from pathlib import Path
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np


def run(module, *args):
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)


def main():
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('This runner is for Kaggle only')
    os.environ['OMP_NUM_THREADS'] = '2'
    run('pip', 'install', '-q', 'segmentation-models-pytorch', 'timm', 'rasterio', 'pyarrow')
    import torch
    import rasterio
    assert torch.cuda.is_available(), 'Enable GPU T4'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    sat = Path('/kaggle/input/datasets/z789456sx/ts-satfire/ts-satfire')
    if not sat.is_dir():
        matches = [p for p in Path('/kaggle/input').rglob('ts-satfire') if any(p.glob('*/VIIRS_Day'))]
        if len(matches) != 1:
            raise RuntimeError(f'Attach TS-SatFire before Save & Run All: {matches}')
        sat = matches[0]
    weather = list(Path('/kaggle/input').rglob('ghcnd_wide_2017.parquet'))
    if len(weather) != 1:
        raise RuntimeError('Attach the GHCN analysis layer before Save & Run All')
    weather_root = weather[0].parent
    root = Path('/kaggle/working/wildfire')
    root.mkdir(exist_ok=True)
    roi = root / 'roi'
    roi.mkdir(exist_ok=True)
    for name in ['fire_all_test.csv'] + [f'us_fire_{y}_out_new.csv' for y in range(2017, 2021)]:
        target = roi / name
        if not target.exists():
            urllib.request.urlretrieve('https://huggingface.co/datasets/SamuelWu318/ts-satfire/resolve/main/roi/' + name, target)

    # A small raw-data audit runs on Kaggle, before allocating a large cache.
    audit = []
    for event in sorted(p for p in sat.iterdir() if p.is_dir())[:6]:
        for folder in ('VIIRS_Day', 'FirePred'):
            files = sorted((event / folder).glob('*.tif'))
            if not files:
                continue
            with rasterio.open(files[0]) as src:
                a = src.read().astype(np.float32)
                bands = []
                for band in a:
                    finite = band[np.isfinite(band)]
                    bands.append({'finite': float(np.isfinite(band).mean()),
                                  'quantiles': np.percentile(finite, [1, 50, 99]).tolist() if finite.size else []})
                audit.append({'path': str(files[0]), 'shape': list(a.shape), 'crs': str(src.crs),
                              'bounds': list(src.bounds), 'bands': bands})
    (root / 'raw_audit.json').write_text(json.dumps(audit, indent=2))

    # Check new spatial fusion in the requested GPU environment, never locally.
    from wf.models import Net
    m = Net({'trunk': 'unet', 'widths': [8, 16], 'depths': [1, 1], 'cin': 33,
             'late': 8, 'late_mode': 'spatial'}, [0.]*33, [1.]*33).cuda()
    result = m(torch.randn(2, 33, 64, 64, device='cuda'), torch.randn(2, 8, device='cuda'))['main']
    assert result.shape == (2, 1, 64, 64)
    result.mean().backward()
    assert torch.isfinite(m.late[-1].weight.grad).all()
    del m, result
    torch.cuda.empty_cache()

    cache = Path('/kaggle/working/tsf_full')
    if not (cache / 'norm.json').exists():
        run('wf.tsf', '--cache', sat.parent, '--roi-csv', roi, '--out', cache,
            '--keep', '--per-day', 16, '--workers', 2, '--splits', 'train,eval')
    try:
        station = weather_root / 'ghcnd_wide_stations.parquet'
        if not station.exists():
            station = root / 'ghcnd-stations.txt'
            urllib.request.urlretrieve('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt', station)
        run('wf.ext', '--ghcn', weather_root, '--stations', station, '--out', cache,
            '--splits', 'train,eval')
    except Exception as exc:
        # Preserve usable vision experiments when an external schema/join fails.
        (cache / 'weather_audit.json').write_text(json.dumps({'train_complete_fraction': 0,
                                                            'error': str(exc)}, indent=2))
        print('WEATHER_DISABLED:', repr(exc), flush=True)
    run('wf.explore', '--source', cache, '--out', '/kaggle/working/exploration', '--hours', 8)


if __name__ == '__main__':
    main()
