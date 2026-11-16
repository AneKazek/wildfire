# Preserved experiments and comparisons

The default workflow is now WildfireTD4. Historical implementations and
artifacts are retained at their existing paths to preserve imports, notebooks
and reproducibility. No prior results, tests or architectures were deleted.

| Material | Preserved location |
| --- | --- |
| Earlier project entry point | `docs/HISTORICAL_README.md` |
| NDWS architecture/loss/ensemble results | `docs/RESULTS.md`, `docs/reports1.1.md` |
| Prior architecture documentation and figures | `docs/ARCHITECTURE.md`, `docs/architecture.*`, `docs/td_architecture.*` |
| Prediction panels and measured metrics | `docs/test_vs_gt.*`, `docs/results/`, `results/` |
| Scalar-weather pilot | `docs/LATE_FUSION.md`, `wf/latefusion.py` |
| Broad Kaggle search | `docs/EXPLORATION.md`, `wf/explore.py`, `kaggle/wildfire_deep_search.ipynb` |
| Candidate recipes and historical trunks | `wf/candidates.py`, `wf/models.py` |
| Search, finals, ensemble, postprocessing | existing `wf/search.py`, `wf/final.py`, `wf/ensemble.py`, other `wf/` runners |
| Earlier tests | `tests/test_smoke.py`, `tests/test_exploration.py` |
| Transcribed available Kaggle numbers | `docs/comparisons/available_results.json` |

`available_results.json` contains only values present in the user-supplied log;
it is not the full search leaderboard. Both completed finalists listed there
use full training; screening rows use 12k training patches. All these rows use
the same 1,248-patch tuning subset. Confirmation/test are not reported.
The 727-patch weather pilot and older NDWS test scores are separate protocols.

Existing Kaggle runs and their checkpoint binaries are not overwritten by this
repo refactor. Remote checkpoint retention still depends on saved Kaggle
outputs. Existing local `results/` and checkpoint files remain untouched and
remain ignored by Git; this index does not claim missing binaries were fetched.

## Recorded Kaggle comparison tables

These are measured results from the user-supplied log excerpt, recorded on
2026-11-16. Duplicate stdout lines in that log count as one run. Exact numbers
are retained in [available_results.json](available_results.json). The excerpt
does not contain the complete search history. No missing score is inferred.

### Full-data finalists: 21,719 training / 1,248 tuning patches

Both runs used seed 0, batch size 32, a maximum of 60 epochs, patience 12,
FP16 and T4. Checkpoints and thresholds were selected on tuning. Neither row
is a confirmation or test result. Epoch indices are zero-based.

| Run | Architecture / loss | Macro Dice | Micro Dice | IoU | Threshold | Best epoch index |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_td4_w24_tversky_recall_seed0` | TD4 width 24; focal Tversky recall | 0.449657 | 0.457701 | 0.296765 | 0.35 | 20 |
| `final_td4_w24_tversky_precision_seed0` | TD4 width 24; focal Tversky precision | 0.433120 | 0.418257 | 0.264428 | 0.20 | 24 |

The selected model is **WildfireTD4**, matching the recall-Tversky finalist
above, with 5,973,133 parameters. Its recorded operating point is threshold
0.35, epoch index 20. It stopped at epoch index 31 after 31 minutes.
The precision finalist stopped at epoch index 35 after 34.9 minutes.
These are full-data training results, not equal-budget comparisons with the
screening table below.

### Screening and loss ablations: 12,000 training / 1,248 tuning patches

All listed runs used seed 0, batch size 32 and up to 18 epochs. The architecture
screen used BCE + Dice with positive weight 5; loss rows changed the stated
loss on TD4 width 24. Shared training inputs include the cache's existing
weather channels. Only the two explicitly named external-weather variants add
GHCN features. Their result alone does not establish a gain over a matched
weather-free TD4 control, whose score is absent from this excerpt.

| Run | Architecture / loss | Macro Dice | Micro Dice | IoU | Threshold | Best epoch index |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `loss_td4_w24_tversky_recall` | TD4 width 24; alpha 0.4 / beta 0.6 / gamma 1.33 | 0.434107 | 0.444772 | 0.285985 | 0.85 | 10 |
| `loss_td4_w24_tversky_precision` | TD4 width 24; alpha 0.7 / beta 0.3 / gamma 1.33 | 0.420953 | 0.392042 | 0.243814 | 0.20 | 10 |
| `loss_td4_w24_bce_pw1` | TD4 width 24; BCE + Dice, positive weight 1 | 0.415920 | 0.407160 | 0.255619 | 0.55 | 15 |
| `screen_td4_weather_scalar` | TD4 width 16; external GHCN scalar logit correction | 0.398317 | 0.427477 | 0.271842 | 0.65 | 17 |
| `screen_td4_weather_spatial` | TD4 width 16; external GHCN spatial logit correction | 0.376648 | 0.405699 | 0.254468 | 0.50 | 17 |
| `screen_unet` | Custom U-Net; widths 32/64/128/256 | 0.370079 | 0.430256 | 0.274093 | 0.75 | 17 |
| `screen_hrnet` | HRNet; width 24, three stages | 0.368832 | 0.392291 | 0.244006 | 0.75 | 17 |
| `screen_unet_effb1` | U-Net / EfficientNet-B1 | 0.364912 | 0.423022 | 0.268248 | 0.75 | 12 |
| `screen_unet_r34` | U-Net / ResNet34 | 0.361311 | 0.418907 | 0.264948 | 0.75 | 17 |
| `screen_unetpp_r18` | U-Net++ / ResNet18 | 0.361134 | 0.415962 | 0.262596 | 0.70 | 16 |
| `screen_dual_fuel_weather` | Dual fuel/weather encoders; widths 32/64/128 | 0.346850 | 0.416183 | 0.262772 | 0.75 | 17 |
| `screen_unet_convnext` | U-Net / ConvNeXt Tiny | 0.338986 | 0.383687 | 0.237384 | 0.70 | 16 |
| `screen_unet_mitb1` | U-Net / MiT-B1 | 0.329426 | 0.378058 | 0.233090 | 0.75 | 16 |
| `screen_unet_mitb0` | U-Net / MiT-B0 | 0.327159 | 0.391161 | 0.243133 | 0.80 | 16 |
| `loss_td4_w24_lovasz` | TD4 width 24; Lovasz hinge | 0.238837 | 0.298398 | 0.175363 | 0.50 | 13 |

Recall Tversky outperformed precision Tversky by 0.013154 macro Dice in the
matched width-24 screening comparison. The BCE-positive-weight-1 variant scored
0.415920, while Lovasz scored 0.238837. Scalar external-weather fusion scored
0.398317 versus 0.376648 for spatial external-weather fusion. Those two
weather variants use width 16 and should not be used to isolate the effect of
weather against the width-24 loss finalists.

### Earlier external-weather pilot: separate dataset budget and eval subset

Source: [full pilot report](../LATE_FUSION.md).
Both rows used 727 training / 1,016 eval patches, seed 0, 12 epochs, batch 32,
TD4 width 16, and BCE + Dice. This pilot used the earlier loader/join code, so
it is a separate protocol from the larger search. Do not compute improvements
across these tables.

| Run | External input | Macro Dice | Micro Dice | IoU | Threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_seed0` | None | 0.182445 | 0.233982 | 0.132491 | 0.35 |
| `weather_late_seed0` | GHCN scalar correction | 0.177887 | 0.209093 | 0.116752 | 0.30 |

Weather reduced pilot macro Dice by 0.004558. Both rows were measured on eval;
test was not evaluated.

## Experiments whose final results are not in the available excerpt

These rows document coverage gaps, not failed experiments. The broad search
may have executed them; the supplied excerpt begins after their expected
screening positions. Its loss-run names indicate that width 24 was promoted,
but do not reveal that architecture's original screen score.

| Run / group | Configuration or purpose | Evidence status |
| --- | --- | --- |
| `screen_td4` | Width-16 four-scale control | Configured in search; final score not supplied |
| `screen_td4_attention` | Control plus SE attention | Configured; final score not supplied |
| `screen_td4_wide_decoder` | Decoder width multiplier 2 | Configured; final score not supplied |
| `screen_td4_w24` | Encoder widths 24/48/96/192 | Promoted by search; original screen score not supplied |
| `screen_td4_d2` | Two blocks per encoder scale | Configured; final score not supplied |
| `screen_td4_deep_supervision` | Auxiliary decoder supervision, weight 0.3 | Configured; final score not supplied |
| `final_td4_w24_bce_pw1_seed0` | Full-data BCE + Dice finalist | RUN line supplied; excerpt ends before epochs / FINAL |
| Finalists with seeds 1 and 2 | Repeat the three promoted recipes | Planned when budget permits; outcomes not supplied |
| TTA 1 versus 4 | Tune inference augmentation | Planned; comparison not supplied |
| Ensembles of top 1–6 finalists | Equal-weight blends, tuning-only selection | Planned; scores / frozen recipe not supplied |
| Reserved confirmation events | Evaluate the frozen recipe once | Confirmation report not supplied |
| Test split | Final held-out performance | Not evaluated by the search runner |

## Earlier NDWS experiments — preserved measured comparisons

The complete historical tables remain in [RESULTS.md](../RESULTS.md), with
their cache-generation caveats. They cover all previously documented families:
TD sum/gate/three-branch/deep/wide/multihead, SMP encoders, dual CAFIM, loss and
augmentation variants, attention/decoder-depth changes, resolution/deep
supervision, area heads, threshold/shape postprocessing, seed repeats,
ensembles, and alternative-dataset checks. Rejected experiments and oracle
probes remain documented; oracle scores are not deployable model results.

| Historical block | What is recorded | Markdown source |
| --- | --- | --- |
| Old-cache architecture screen | SMP, TD and dual-encoder comparisons | [RESULTS.md](../RESULTS.md#stage-1-screen-old-cache-50k-12-epochs-full-eval-scoring) |
| Rebuilt-cache TD screen | TD variants versus U-Net++ control | [RESULTS.md](../RESULTS.md#td-fusionunet-line-stage-1-50k-rebuilt-cache) |
| Single-model finalists and seeds | Full-data runs under each cache generation | [RESULTS.md](../RESULTS.md) |
| Ensemble sweeps and test | Blend weights, thresholds, old/rebuilt test results | [RESULTS.md](../RESULTS.md) |
| Rejected loss / preprocessing / postprocessing | Measured negative results and tradeoffs | [RESULTS.md](../RESULTS.md#what-did-not-work-measured-not-guessed) |
| Resolution and deep supervision | Round-7 comparisons | [RESULTS.md](../RESULTS.md) |
| Area / centroid / morphology probes | Decoder controls and GT-area oracle | [RESULTS.md](../RESULTS.md#where-the-dice-actually-goes-wfprobe-td_gate3_s1-thr-065-base-0459) |
| External dataset review | Compatibility and alignment checks | [RESULTS.md](../RESULTS.md#alternative-datasets-reviewed-not-used) |
| Model diagrams and rationale | Prior architecture descriptions and figures | [ARCHITECTURE.md](../ARCHITECTURE.md) |

Historical NDWS rebuilt-cache ensemble test macro Dice **0.4826** belongs to a
different task/cache and split from WildfireTD4 tuning macro Dice **0.449657**.
They are retained for historical context, not ranked against each other.

## Validation runs — software correctness, not model scores

| Kaggle check | Recorded result | Scope |
| --- | --- | --- |
| [Exploration preflight](https://www.kaggle.com/code/anekazek/wildfire-exploration-preflight), version 1 | Four focused tests and 11 custom-model forward/backward checks passed | Weather normalization/join, row alignment, loader RNG, model shapes |
| Same preflight, version 2 | Four tests plus six SMP forward/backward checks passed | Shape/backward checks with random weights; not pretrained training scores |
| [WildfireTD4 contract check](https://www.kaggle.com/code/anekazek/wildfire-td4-contract-check) | Six tests passed; CLI help, compilation and TOML parsing passed | Exact parameter count, fixed recipe, and prediction equality with a synthetic legacy-format EMA checkpoint |

The actual winning checkpoint binary has not been downloaded and independently
loaded by the refactor. No new accuracy measurement is attributed to these
software checks.
