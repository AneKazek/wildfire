# Attach both datasets through Kaggle Add Input. No TS-SatFire download.
from pathlib import Path
import subprocess, sys, urllib.request
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[external]'], check=True)
roots = list(Path('/kaggle/input').rglob('ts-satfire'))
roots = [p.parent for p in roots if p.is_dir() and (p.parent / 'roi').is_dir()]
if len(roots) != 1:
    raise RuntimeError(f'Expected one ts-satfire/ + roi/ dataset layout; found {roots}. Inspect attached dataset paths.')
weather = list(Path('/kaggle/input').rglob('ghcnd_wide_2017.parquet'))
if len(weather) != 1:
    raise RuntimeError('Attach Global Daily Weather analysis layer with ghcnd_wide_2017.parquet')
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Select GPU T4 x2 in Kaggle session options')
print('GPU:', torch.cuda.get_device_name(0), '(single GPU used)')
# Start with a bounded screening run; remove --limit for all events.
subprocess.run([sys.executable, '-m', 'wf.tsf', '--cache', str(roots[0]),
                '--out', '/kaggle/working/tsf', '--keep', '--limit', '8',
                '--per-day', '8', '--workers', '2'], check=True)
station = Path('/kaggle/working/ghcnd-stations.txt')
if not station.exists():
    urllib.request.urlretrieve('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt', station)
subprocess.run([sys.executable, '-m', 'wf.ext', '--ghcn', str(weather[0].parent),
                '--stations', str(station), '--out', '/kaggle/working/tsf'], check=True)
subprocess.run([sys.executable, '-m', 'wf.latefusion', '--root', '/kaggle/working/tsf',
                '--out', '/kaggle/working/results/latefusion'], check=True)
