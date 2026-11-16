# Kaggle workflows

## Primary model: WildfireTD4

The selected default is the weather-free TD4 wide model with focal Tversky
recall loss. Use `python -m wf train`, `python -m wf evaluate` or
`python -m wf predict` inside Kaggle. See the repository README and
`docs/MODEL.md` for the exact checkpoint, channel order and threshold.
Keep an existing cache's tuning/confirmation event partition unchanged.

The workflows below are preserved historical comparisons; they are not the
default model-training route.

## Historical late-fusion pilot

Training and dataset conversion run in Kaggle. Do not run the experiment locally.

Attach these inputs without downloading TS-SatFire to your computer:

- `z789456sx/ts-satfire`
- `dianatofficial/global-daily-weather-1950-2026-analysis-layer`

The second dataset's parquet layout must be verified when attached. The runner
expects `ghcnd_wide_2017.parquet` through `ghcnd_wide_2021.parquet`, with columns
`station_id`, `date`, `tmax_c`, `tmin_c`, `prcp_mm`. It fails if inputs are absent.

Upload the modified repository as a private Kaggle code input (including `wf/`
and `pyproject.toml`). In a notebook with Internet enabled and GPU T4 selected:

```python
from pathlib import Path
import shutil, os, runpy
sources = list(Path('/kaggle/input').rglob('pyproject.toml'))
sources = [p.parent for p in sources if (p.parent / 'wf' / 'latefusion.py').exists()]
assert len(sources) == 1, sources
shutil.copytree(sources[0], '/kaggle/working/wildfire', dirs_exist_ok=True)
os.chdir('/kaggle/working/wildfire')
runpy.run_path('kaggle/run_latefusion.py', run_name='__main__')
```

`run_latefusion.py` reads the two attached datasets, converts a bounded screening
subset (8 events per split, 8 crops per day), fetches NOAA station coordinates,
joins same-day and trailing weather by location, and trains two models using
the same seed and budget. Only one T4 is used. All outputs go under
`/kaggle/working/`; attached data are retained. Internet is needed for dependencies
and NOAA station coordinates.

The external MLP adds a zero-initialized scalar correction to the segmentation
logits. This is weather-conditioned output fusion, not concatenated image inputs.
Train statistics normalize weather; missing values become zero after scaling.

Read `/kaggle/working/results/latefusion/comparison.json` for eval Dice difference
and chosen thresholds. Neither model is selected on test. The runner does not
evaluate test. Remove `--limit` and increase the training budget for a full run.
Screening results must not be compared directly to earlier full-data results.

`kernel-metadata.json` is a template, not a submitted run: replace its username,
provide the code input, and request the available T4 accelerator when submitting.
Dataset contents, T4 allocation, and actual performance remain unverified until
the Kaggle session runs successfully.
