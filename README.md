# Next-day wildfire spread — staged architecture search

Best result found on an L4: **macro Dice 0.4823 / micro 0.4353 / hard Dice 0.2620** on
the held-out test split (single pass), versus 0.4726 / 0.4280 / 0.126 for the previous
best. Full numbers and every ablation: [`docs/RESULTS.md`](docs/RESULTS.md).
Prediction-vs-ground-truth panels for the final ensemble:
[`docs/test_vs_gt.png`](docs/test_vs_gt.png).

## Layout

```
wf/data.py        cached 23-channel patches -> tensors, dihedral aug with vector fix
wf/models.py      trunks (smp / unet / hrnet / dual-branch CAFIM) + multi-head decoder
wf/losses.py      bce+dice, tversky, focal tversky, lovasz
wf/metrics.py     macro/micro dice, iou, P/R, PR-AUC, boundary F1, hard dice
wf/train.py       one run: EMA, bf16, channels_last, cosine, early abort
wf/candidates.py  candidate space + stage protocols
wf/search.py      driver, resume-safe, writes results/search_s<stage>.csv
wf/final.py       finalist: full data, TTA, threshold on eval, optional test
wf/ensemble.py    blends finished runs, threshold on eval, one test pass
docs/RESULTS.md   measured screening / finals / ensemble tables
docs/ARCHITECTURE.md  diagram (mermaid + svg/png) and why each piece survived
docs/test_vs_gt.png   final ensemble vs ground truth on 6 test patches
```

## Protocol

* Never select on `test`; the threshold and blend weights are tuned on `eval`, and
  `test` is evaluated once with the final ensemble.
* Screening compares candidates at an identical budget (same subset seed, epochs, lr,
  batch), aborting anything below 0.33 macro Dice after epoch 4. Eval bootstrap CI is
  ±0.0036, so sub-0.004 gaps are treated as noise.
* Reported metrics: macro Dice (primary), micro Dice, IoU, precision/recall, PR-AUC,
  boundary F1, and `hard_dice` (outside the radius-2 centre disk = real spread).

## Commands (`PYTHONPATH=/content/wildfire`, cwd `/content/wildfire`)

```bash
python -m wf.search  --stage 1 --deadline 240
python -m wf.final   --name smp_unetpp_r18 --stage 3 --tta 8
python -m wf.final   --name smp_unet_mitb0 --stage 3 --tta 4 --seed 1 --tag _seed1
python -m wf.ensemble --runs A,B,C,D,E --weights 0.25,0.20,0.20,0.20,0.15 --tta 4 --test
```

Final ensemble: `smp_unetpp_r18` (0.25) + `smp_unet_mitb0` seeds 0/1/2 (0.20 each) +
`smp_unet_r34` (0.15), threshold 0.45.
