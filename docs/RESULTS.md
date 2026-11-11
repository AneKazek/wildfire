# Results — next-day wildfire spread

All numbers are on the cached 23-channel 64×64 patches, positive class `FireMask >= 7`,
pixels of class 4/6 excluded. Primary metric: **macro Dice per sample** at a threshold
tuned on `eval`. `test` was evaluated once, with the final ensemble only.

## Protocol

| stage | data | epochs | lr | batch | purpose |
|---|---|---|---|---|---|
| 1 | 50k | 12 | 3e-3 | 384 | screen every family, abort below 0.33 after epoch 4 |
| 3 | all 100,904 | 36 (patience 8) | 2.2e-3 | 384 | finalists, EMA, bf16, channels_last |

Screening and finalists share seed 0. Eval bootstrap CI is ±0.0036 (n = 10,512), so
gaps below ~0.004 in the screen are noise — architecture choice was made on family
diversity, not on small deltas.

## Stage 1 screen (50k, 12 epochs, full-eval scoring)

| candidate | params | macro Dice | micro Dice | hard Dice | min |
|---|---|---|---|---|---|
| smp_unetpp_r18 | 16.0M | **0.4589** | 0.3845 | 0.2516 | 6.6 |
| smp_unet_mitb0 | 5.6M | 0.4563 | 0.3864 | 0.2597 | 3.8 |
| smp_unet_r34 | 24.5M | 0.4561 | 0.3851 | 0.2547 | 3.9 |
| smp_unet_r18 | 14.4M | 0.4543 | 0.3830 | 0.2685 | 3.5 |
| dual32 (fuel/weather + CAFIM) | 4.1M | 0.4499 | 0.3677 | 0.2391 | 15.3 |
| smp_deeplab_r18 | 12.4M | 0.3521 | 0.3443 | 0.2192 | 1.3 |

Rejected quickly: DeepLabV3+ (output stride too coarse for 1–3 px blobs),
dual-branch CAFIM (converges far too slowly at this budget),
2× input upsampling (0.4501 vs 0.4507 at epoch 5 — no gain, 4× the compute).

## Finalists (full train set, 36 epochs, TTA)

| run | eval macro | eval micro | eval hard | thr | min |
|---|---|---|---|---|---|
| smp_unetpp_r18_s3 | 0.4686 | 0.3900 | 0.2473 | 0.65 | 43.6 |
| smp_unet_mitb0_s3 (seed 0) | 0.4685 | 0.3965 | 0.2539 | 0.50 | 31.4 |
| smp_unet_mitb0_s3 (seed 2) | 0.4685 | 0.3928 | 0.2464 | 0.65 | 32.3 |
| smp_unet_mitb0_s3 (seed 1) | 0.4668 | 0.3862 | 0.2455 | 0.65 | 32.3 |
| smp_unet_r34_s3 | 0.4643 | 0.3880 | 0.2433 | 0.60 | 26.9 |

## Ensemble (eval)

| blend | macro Dice |
|---|---|
| unetpp 0.5 + mitb0 0.5 | 0.4707 |
| unetpp 0.45 + mitb0 0.45 + r34 0.10 | 0.4709 |
| equal thirds | 0.4694 |
| **unetpp 0.25 + mitb0 seeds 0.20 x3 + r34 0.15** | **0.4710** |

Weights were swept on eval only. Extra mitb0 seeds added +0.0003 macro (noise) but the
largest micro Dice (0.3994) and hard Dice (0.2568), so the 5-model blend was kept: it
is also the least dependent on a single checkpoint.

## Test (single pass, final ensemble, thr 0.45, TTA-4)

| metric | this run | previous best |
|---|---|---|
| macro Dice | **0.4823** | 0.4726 |
| micro Dice | 0.4353 | 0.4280 |
| IoU | 0.2782 | — |
| precision / recall | 0.4083 / 0.4661 | — |
| hard Dice (outside r=2) | **0.2620** | 0.126 |
| boundary F1 | 0.8413 | — |
| PR-AUC | 0.7774 | — |

## Test set, prediction vs ground truth

Reproduce with `python -m wf.inspect_test` (same 5 checkpoints, TTA-4, thr 0.45); it
rebuilds the blend and should print the exact numbers above. Nothing was picked using
`test` here either — checkpoints and threshold were fixed by eval, so this is a
reproducibility check plus a visual, not a second selection pass. Outputs:
`docs/test_vs_gt.png` (6 cases x input / P(fire) / truth / prediction / overlay) and
`docs/test_vs_gt.json`.

Per-patch Dice on the 12,467 test patches: mean 0.4823, median 0.4615, p10 0.25,
p90 0.75; 47.2% of patches score >= 0.50, none score 0 and no patch is predicted
empty, so the model always puts fire somewhere. Pixel sums: 103,375 TP / 149,835 FP /
118,429 FN (precision 0.408, recall 0.466) — it over-fires slightly more than it
misses, which is the right side to err on for early warning.

![test predictions vs ground truth](test_vs_gt.png)

What the picture shows:

* Small fires are close to solved: the best case is 3 GT px and 3 predicted px (Dice
  1.0), and the median case (6 GT px) overlaps on 3 of them (Dice 0.46).
* The failure mode is shape, not location: on larger fires it predicts a lump on the
  perimeter of the previous-day burn scar and leaves the interior green (case 5,
  recall 25% on 521 GT px), or draws a ring around the scar that is mostly false
  alarm (case 6, 738 predicted px vs 121 true).
* Very large GT patches (`largest GT`, dice 0.33) are where the remaining macro Dice
  is lost; a loss that weights recall on big fires is the obvious next move, not a
  bigger backbone.

## What did not work (measured, not guessed)

* DeepLabV3+ / coarse-decoder models: −0.10 macro Dice.
* 2× upsampled input (finer feature grid, 4× compute): no gain at equal budget.
* Shift TTA (±1, ±2 px): −0.02 to −0.13, blurs the centred blob.
* Masking predictions outside a centre radius: −0.005.
* `prev_fire` is a weak prior: target is a superset of the previous-day fire in only
  6% of samples, and its pixels sit at median radius 12 vs 1.6 for the target.
* 8-way dihedral TTA: macro-neutral (0.4558 vs 0.4561), kept only because it slightly
  improves micro Dice and recall.

## Data facts that shaped the design

* All targets are non-empty; median 7 fire px, mean 16, p95 79.
* 90% of target pixels lie within radius 3.2 of the patch centre (the patch is cropped
  on the next-day fire), so this is a small-blob problem, not a large-region one.
* A fixed disk of radius 2 already scores 0.351 macro Dice — the learned models add
  +0.12 over that prior.
