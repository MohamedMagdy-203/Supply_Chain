# Modeling Guide (read this before writing any model)

Applies to Members 2, 3 and 4. All data preparation is finished; you only read
`data/processed/features.parquet` through `src/data.py`.

## 1. Rules

1. Never re-split. Use the `split` column (train / val / test, defined by target week).
2. Fit scalers, encoders and imputers on `train` rows only.
3. Choose hyperparameters on `val`. Evaluate on `test` once, at the end.
4. Do not add features outside `src/features.py`. If you need a new one, add it there and re-run `pytest tests/test_leakage.py`.
5. Every model writes `predictions/{model}.parquet` through `save_predictions` (never `to_parquet` directly).

## 2. Vocabulary

- Target week `T`: the week being forecast. `y_next` is its sales.
- `origin_week = T - 1`: the last week whose sales are known. This is the key used everywhere (features, predictions, API).
- One row = one `(series_id, origin_week)`.

## 3. Reading the data

```python
from src.data import load_xy, load_series_type

X_train, y_train, keys_train = load_xy("train")
X_val,   y_val,   keys_val   = load_xy("val")
X_test,  y_test,  keys_test  = load_xy("test")
```

- `X` has the numeric features plus `store_id`, `dept_id`, `cat_id` as pandas `category` (levels fixed from train).
- LightGBM: pass `X` directly (handles NaN and categories).
- Linear / deep: impute NaN and scale inside your own pipeline, fitted on train only.
  Known NaN: `lag_52` (about 20 percent, series younger than one year; the flag `has_lag52` marks them).
- `on_sale` is constant (always 1) in the current data; ignore it or drop it.
- Linear models: `roll_mean_4` is exactly the mean of `lag_1..lag_4` (perfect collinearity). Use Ridge or regularized quantile regression, or drop one group.

## 4. Writing predictions

```python
from src.data import save_predictions

preds = keys_test.copy()                      # series_id, origin_week
preds["q10"], preds["q25"], ... = ...         # 7 columns: q10 q25 q50 q75 q90 q95 q99
save_predictions(preds, "lgbm")               # validates, then writes predictions/lgbm.parquet
```

`save_predictions` checks: required columns, no NaN, no duplicate keys, keys exist in val/test,
**all test rows present**, quantiles non-decreasing (crossing is fixed by sorting and reported),
values clipped at 0. Include the val rows too if you can (the evaluation harness uses them for tuning).

## 5. Evaluation harness (Member 4)

```python
from src.data import load_features, load_series_type
truth = load_features("test")[["series_id", "origin_week", "y_next", "dept_id", "store_id"]]
# join predictions/{model}.parquet on (series_id, origin_week), then compute metrics
# join load_series_type() on series_id for the intermittent / lumpy breakdown
```

All models are scored on the same test rows with the same code.

## 6. MLflow

One experiment name for everybody. Tags: `model_family` (baseline | linear | lgbm | deep) and `fold` (fold number or `final`).