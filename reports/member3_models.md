# Member 3: Linear and LightGBM quantile models

## Scope

- Stores: 3 (CA_1, CA_2, CA_3), items: 301, series: 903
- Granularity: weekly; horizon: 1 week ahead
- Validation: weeks 253-264 (10775 rows); test: weeks 265-276 (10806 rows, held out)
- Quantiles: 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99
- Costs: Cu = 30% and Co = 10% of unit price -> critical ratio tau = 0.75 -> order up to q75

How to read the tables:
- **Pinball loss**: lower is better (units sold per series-week).
- **Service level**: share of weeks where demand was at or below the forecast; should match the quantile level (target row).
- **Ordering at tau**: stock up to q75 for one week. Stock-out rate should be near 0.25; lower cost is better. This is a quick check under the README cost assumptions; the full policy comparison is Member 1's.

## Val split (model selection)

### Pinball loss per quantile

| model | q10 | q25 | q50 | q75 | q90 | q95 | q99 | mean |
|---|---|---|---|---|---|---|---|---|
| linear | 0.832 | 1.578 | 2.157 | 1.931 | 1.231 | 0.807 | 0.285 | 1.260 |
| quantreg | 0.813 | 1.538 | 2.130 | 1.909 | 1.206 | 0.793 | 0.274 | 1.238 |
| lgbm | 0.846 | 1.495 | 2.026 | 1.814 | 1.172 | 0.786 | 0.301 | 1.206 |

### Service level per quantile

| model | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---|---|---|---|---|---|---|
| target | 0.100 | 0.250 | 0.500 | 0.750 | 0.900 | 0.950 | 0.990 |
| linear | 0.239 | 0.329 | 0.510 | 0.742 | 0.900 | 0.950 | 0.991 |
| quantreg | 0.234 | 0.336 | 0.524 | 0.737 | 0.892 | 0.941 | 0.982 |
| lgbm | 0.225 | 0.323 | 0.529 | 0.756 | 0.894 | 0.939 | 0.982 |

### Intervals and median error

| model | coverage q25-q75 | coverage q10-q90 | width q25-q75 | width q10-q90 | MAE of q50 |
|---|---|---|---|---|---|
| target | 0.500 | 0.800 |  |  |  |
| linear | 0.528 | 0.822 | 6.666 | 13.212 | 4.314 |
| quantreg | 0.498 | 0.801 | 6.696 | 13.402 | 4.260 |
| lgbm | 0.469 | 0.834 | 6.603 | 12.958 | 4.053 |

### Ordering at tau = 0.75 (q75)

| model | stock-out rate | units short / week | units left over / week | cost $ / series-week | total cost $ |
|---|---|---|---|---|---|
| linear | 0.26 | 1.29 | 3.86 | 2.40 | 25,880.82 |
| quantreg | 0.26 | 1.23 | 3.94 | 2.38 | 25,641.89 |
| lgbm | 0.24 | 1.18 | 3.72 | 2.26 | 24,297.86 |

## Test split (one-time check)

### Pinball loss per quantile

| model | q10 | q25 | q50 | q75 | q90 | q95 | q99 | mean |
|---|---|---|---|---|---|---|---|---|
| linear | 0.901 | 1.638 | 2.154 | 1.858 | 1.138 | 0.737 | 0.259 | 1.241 |
| quantreg | 0.867 | 1.605 | 2.134 | 1.826 | 1.119 | 0.719 | 0.235 | 1.215 |
| lgbm | 0.839 | 1.522 | 1.996 | 1.749 | 1.088 | 0.700 | 0.243 | 1.162 |

### Service level per quantile

| model | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---|---|---|---|---|---|---|
| target | 0.100 | 0.250 | 0.500 | 0.750 | 0.900 | 0.950 | 0.990 |
| linear | 0.207 | 0.318 | 0.506 | 0.748 | 0.897 | 0.951 | 0.989 |
| quantreg | 0.192 | 0.305 | 0.502 | 0.744 | 0.894 | 0.949 | 0.989 |
| lgbm | 0.184 | 0.302 | 0.523 | 0.752 | 0.896 | 0.940 | 0.985 |

### Intervals and median error

| model | coverage q25-q75 | coverage q10-q90 | width q25-q75 | width q10-q90 | MAE of q50 |
|---|---|---|---|---|---|
| target | 0.500 | 0.800 |  |  |  |
| linear | 0.504 | 0.805 | 6.628 | 13.195 | 4.307 |
| quantreg | 0.500 | 0.807 | 6.933 | 13.732 | 4.267 |
| lgbm | 0.475 | 0.836 | 6.577 | 13.307 | 3.992 |

### Ordering at tau = 0.75 (q75)

| model | stock-out rate | units short / week | units left over / week | cost $ / series-week | total cost $ |
|---|---|---|---|---|---|
| linear | 0.25 | 1.11 | 4.09 | 2.46 | 26,558.44 |
| quantreg | 0.26 | 1.04 | 4.19 | 2.41 | 26,076.66 |
| lgbm | 0.25 | 1.02 | 3.92 | 2.35 | 25,434.27 |

## Test mean pinball by demand type

| demand_type | linear | quantreg | lgbm |
|---|---|---|---|
| erratic | 1.498 | 1.454 | 1.374 |
| intermittent | 0.745 | 0.718 | 0.677 |
| lumpy | 0.941 | 0.905 | 0.839 |
| smooth | 1.439 | 1.416 | 1.362 |

## SHAP: most important features (LightGBM)

| feature (mean |SHAP|, units) | q10 | q50 | q75 | q90 | q99 |
|---|---|---|---|---|---|
| roll_mean_4 | 1.195 | 6.567 | 9.906 | 10.545 | 4.316 |
| lag_1 | 2.183 | 4.343 | 3.105 | 2.199 | 5.785 |
| roll_mean_13 | 0.509 | 0.952 | 0.839 | 1.231 | 2.885 |
| roll_mean_8 | 1.099 | 0.518 | 0.789 | 1.762 | 1.564 |
| lag_2 | 0.864 | 0.500 | 0.268 | 0.185 | 0.201 |
| weeks_since_sale | 0.884 | 0.482 | 0.380 | 0.277 | 0.221 |
| roll_std_13 | 0.037 | 0.322 | 0.485 | 1.114 | 4.048 |
| lag_52 | 0.118 | 0.307 | 0.470 | 0.641 | 1.358 |
| snap_days | 0.000 | 0.305 | 0.380 | 0.374 | 0.210 |
| week_of_year | 0.014 | 0.188 | 0.221 | 0.258 | 0.205 |

Plots: `reports/shap/shap_summary_q50.png`, `shap_summary_q90.png`, `shap_importance.png`, and the one-item explanation `local_*.png`.

## Model selection

- Selected on **val mean pinball**: **lgbm** (1.206).
- Best on val pinball at the ordering quantile q75: **lgbm** (1.814).
- Test check: lgbm scores 1.162; the best test score is lgbm (the same model, so the choice holds).
- Runs and per-fold scores are in MLflow: experiment `m5-inventory-forecasting` (`mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`).
