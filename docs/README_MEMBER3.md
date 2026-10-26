# Supply Chain project: Member 3 package (LightGBM, Linear, SHAP)

A ready-to-open folder: the files Member 3 needs from the GitHub repo
(https://github.com/MohamedMagdy-203/Supply_Chain), the processed data, and the
Member 3 code. Nothing has been run yet: `predictions/`, `models/` and `reports/`
are empty and fill up as you run the steps.

## What is inside

| Path | From | Purpose |
|---|---|---|
| `data/processed/` | repo | features.parquet, weekly.parquet, series_type.csv, splits.json, sample_items.csv |
| `src/config.py`, `src/data.py`, `src/features.py` | repo (unchanged) | constants, data loading + `save_predictions`, feature names |
| `tests/conftest.py`, `tests/test_leakage.py`, `pytest.ini` | repo (unchanged) | synthetic test data and the leakage test |
| `docs/` | repo (unchanged) | modeling guide, contracts, data dictionary |
| `src/models/common.py` | Member 3 | model feature list, 3 time-ordered CV folds, MLflow helper |
| `src/models/linear.py` | Member 3 | Ridge -> `predictions/linear.parquet`, Quantile Regression -> `predictions/quantreg.parquet` |
| `src/models/lgbm.py` | Member 3 | LightGBM -> `predictions/lgbm.parquet`, `models/lgbm_q*.txt` |
| `src/explain.py` | Member 3 | SHAP -> `reports/shap/` |
| `scripts/member3_report.py` | Member 3 | summary -> `reports/member3_models.md` |
| `tests/test_models.py` | Member 3 | quick checks on synthetic data (optional) |
| `requirements.txt` | repo + Member 3 | repo file plus `mlflow-skinny` and `shap` |
| `.gitignore` | repo + Member 3 | also ignores `/models/` and `mlflow.db` |
| `.vscode/` | Member 3 | run buttons (F5) and import path for VS Code |

## Setup (once)

1. VS Code: File > Open Folder > this folder (the one that contains `src` and `data`).
2. Terminal > New Terminal:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. `pip install -r requirements.txt` (Python 3.10+)

## Run order (from this folder)

| # | Command | Produces |
|---|---|---|
| 1 | `python -m src.models.linear` | predictions/linear.parquet, predictions/quantreg.parquet |
| 2 | `python -m src.models.lgbm` | predictions/lgbm.parquet, models/lgbm_q10.txt ... q99.txt |
| 3 | `python -m src.explain` | reports/shap/ (needs step 2) |
| 4 | `python scripts/member3_report.py` | reports/member3_models.md |

Optional checks: `python -m pytest tests/test_leakage.py tests/test_models.py`

Or in VS Code: Run and Debug (Ctrl+Shift+D) > choose "1 Linear models", "2 LightGBM", "3 SHAP", "4 Report" > F5.

## What to check

- Every "Saved predictions" line: 21581 rows (val=10775, test=10806).
- Steps 1 and 2 print the 3 fold scores and their mean for every setting tried; lower is better.
- Step 3 prints an additivity error close to 0 for each quantile.
- In the report, coverage_80 should be near 0.80 and coverage_50 near 0.50.
- "MLflow logging skipped: ..." is only a warning; results are still saved.
- MLflow page (optional): `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
  (needs the full `mlflow` package; see the earlier note about pyarrow).

## Putting it back into the repo

Copy `src/models/common.py`, `src/models/linear.py`, `src/models/lgbm.py`, `src/explain.py`,
`scripts/member3_report.py`, `tests/test_models.py`, and the extra lines in
`requirements.txt` and `.gitignore` into the repo. The other files here are the repo's own, unchanged.
