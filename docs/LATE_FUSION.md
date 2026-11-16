# External weather late-fusion screening

Run: https://www.kaggle.com/code/anekazek/wildfire-weather-late-fusion

Results below are transcribed from the completed Kaggle output supplied by the
user on 2026-11-16. They were not independently retrieved from the output API.
The API reported RUNNING with empty logs while the user saw completed output.

Inputs: TS-SatFire (`z789456sx/ts-satfire`) and GHCN weather analysis layer
(`dianatofficial/global-daily-weather-1950-2026-analysis-layer`), attached in
Kaggle. GPU: Tesla T4. No real-dataset processing or training was run locally.

Screening: limit 8 events per split, 8 crops per day; 727 training patches,
1016 eval patches; seed 0, 12 epochs, batch size 32. Both models use the
same TD trunk, budget and split. Weather fusion adds an MLP correction to the
final segmentation logits. Thresholds were selected on eval; test was not
evaluated.

| Eval metric | Baseline | Weather late fusion |
| --- | ---: | ---: |
| Macro Dice | 0.18244475 | 0.17788683 |
| Micro Dice | 0.23398213 | 0.20909278 |
| IoU | 0.13249137 | 0.11675243 |
| Precision | 0.15510159 | 0.14115680 |
| Recall | 0.47612837 | 0.40309393 |
| Hard Dice | 0.17018546 | 0.16447119 |
| Boundary F1 | 0.38242331 | 0.39060531 |
| AP | 0.16817768 | 0.12874271 |
| Threshold | 0.35 | 0.30 |
| Parameters | 2826269 | 2826590 |

Macro Dice difference: **-0.00455792** (about -2.5% relative).
Baseline remains preferable for the primary metric in this screening.
The slight boundary-F1 increase does not establish better overall segmentation.

This is not a full-data result or evidence that external weather cannot help.
Only one seed was tested, training was short, and both models achieved their
best fixed-threshold checkpoint at the last epoch. Earlier repository scores
use different data/protocols and cannot be compared directly to this table.
Weather coverage/missingness was not included in the user-supplied excerpt.

Before a larger experiment, inspect weather coverage and joins. A stronger
comparison should also reset DataLoader RNG independently of model construction
(the extra MLP consumes initialization RNG), repeat seeds, and use identical
training events and budgets. Keep test untouched until selection is complete.

Kaggle artifacts: `/kaggle/working/results/latefusion/comparison.json`, plus
`baseline_seed0/` and `weather_late_seed0/` summaries and best checkpoints.
The MCP save operation dropped dataset attachments in initial attempts; the
user attached both inputs and started version 4 through Kaggle's editor.
