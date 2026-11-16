# Kaggle architecture exploration

Training, validation checks, cache conversion and weather joins run on Kaggle.
Local actions only edit/read source files. The earlier 727-patch experiment is
a screening control, not the target performance for the larger search.

The pipeline uses all available official 2017–2020 training events, 16 crops
per day, and all available validation events. Test events are neither converted
nor evaluated by this search. Downloaded annual ROI CSVs restrict training to
the official training years. Raw band quantiles, shapes, CRS and bounds are
recorded in `wildfire/raw_audit.json` for six events.

The existing positive-progression-centered patch task is retained. Patch
locations use labels to build the dataset, so this is a conditional patch
benchmark, not an unbiased deployment evaluation or the official full-frame
TS-SatFire benchmark. Compare scores only within the same cache/protocol.

Before search, whole validation events are reserved for confirmation using a
fixed seed. The remaining events supply checkpoint/architecture selection,
threshold calibration, TTA choice and ensemble selection. The final recipe is
written before confirmation is evaluated once. A confirmation event-level
bootstrap interval is reported; with few events it is only a rough uncertainty
estimate. The ordinary patch Dice is also reported for continuity.

## Experiment stages

1. Screen 17 recipes: TD four-scale control; attention, wider decoder, wider
   encoder, deeper encoder and deep-supervision variants; plain U-Net; HRNet;
   fuel/weather dual encoder; pretrained U-Net++ ResNet18, U-Net ResNet34,
   MiT-B0, MiT-B1, EfficientNet-B1 and ConvNeXt; scalar and spatial weather
   fusion. Weather recipes run only if complete train weather coverage >=10%.
   Each uses 18 epochs, the same 12k training subset, and at most 15 minutes.
2. Test four loss alternatives on the strongest architecture: lower BCE positive
   weight, recall/precision focal Tversky and Lovasz.
3. Train the best three recipes on all training patches for up to 60 epochs,
   at most 40 minutes per run, repeating seeds 0/1/2 when budget permits.
4. Compare TTA 1/4 and equal-weight ensembles of the top one through six
   finalists, using tuning events only. Fix the recipe, evaluate confirmation,
   and leave test untouched.

Eight hours is the training/search scheduling budget; preprocessing and final
inference are additional. Actual completed candidates depend on GPU throughput.
The Kaggle session timeout should be 12 hours. The code uses one T4 even when
Kaggle allocates two. Failures are recorded, not assigned invented scores.

Weather comes from the attached GHCN analysis layer. Joins use unit-sphere
nearest neighbors, inverse-distance weighting within 150 km, same-day and
trailing observations only. Complete coverage and per-feature observed rates
are saved in `tsf_full/weather_audit.json`. Existing station coordinates from
the attached dataset are preferred. If coordinates/schema fail or coverage is
poor, the image-only experiments continue and weather recipes are excluded.

Spatial late fusion predicts weights for the final decoder feature channels
and combines them into a spatial logit correction plus a scalar bias. It starts
at zero correction. This expands the earlier scalar-only fusion. Both variants
retain the ordinary image inputs and target. Loader RNG is independent of model
initialization, and workers receive separate augmentation seeds.

## Outputs

- `exploration/leaderboard.json`: every successful run and exact configuration.
- `exploration/runs/jobs/*.log`: per-run logs, including failures.
- `exploration/runs/failures.jsonl`: failed candidates.
- `exploration/runs/*/best.pt` and `summary.json`: checkpoints and histories.
- `exploration/search_data/partition.json`: fixed tuning/confirmation event IDs.
- `exploration/frozen_recipe.json`: selected members, threshold and TTA.
- `exploration/report.json`: tuning and confirmation metrics, event scores/CI.
- `exploration/confirm_probability.npy`: selected confirmation probabilities.

No score improvement is claimed until the Kaggle experiment completes.

## Sources

Official event-split implementation:
https://github.com/zhaoyutim/TS-SatFire/blob/main/dataset_gen_pred.py

Official progression-label implementation (this repo still uses its own
positive-crop cache and does not reproduce the official temporal benchmark):
https://github.com/zhaoyutim/TS-SatFire/blob/main/satimg_dataset_processor/satimg_dataset_processor.py
