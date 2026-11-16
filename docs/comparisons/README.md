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
