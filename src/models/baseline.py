"""Baselines (Pillar 4a): the simplest forecasts a model has to beat.

naive : seasonal naive. Demand of next week = demand of the same week last year
        (lag_52). When there is no year of history, fall back to last week (lag_1).
ma4   : moving average of the last 4 weeks (roll_mean_4).

Both give a POINT forecast. To get the 7 quantiles we add the empirical quantiles
of the baseline's own errors on the TRAIN rows:

    q_tau = max(0, forecast + tau-quantile of (y - forecast) on train)

The error distribution is taken per series; a series with fewer than MIN_OBS train
rows uses its department, and then all series. Nothing from val or test is used
to build the quantiles.

Run:  python -m src.models.baseline
"""
import numpy as np
import pandas as pd

from src.config import KEY_COLS, QUANTILE_COLS, QUANTILE_LEVELS
from src.data import load_features, load_xy, save_predictions

BASELINES = {"naive": "Seasonal naive (lag_52, fallback lag_1)",
             "ma4": "Moving average of the last 4 weeks"}
MIN_OBS = 20
LEVELS = list(QUANTILE_LEVELS.values())


def point_forecast(X: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "naive":
        return X["lag_52"].fillna(X["lag_1"]).fillna(0).to_numpy(float)
    if kind == "ma4":
        return X["roll_mean_4"].fillna(X["lag_1"]).fillna(0).to_numpy(float)
    raise ValueError(f"Unknown baseline '{kind}'. Choose from {list(BASELINES)}")


def _quantile_table(errors: pd.Series, by: pd.Series) -> pd.DataFrame:
    """Rows = group, columns = QUANTILE_COLS, values = error quantiles."""
    t = errors.groupby(by).quantile(LEVELS).unstack()
    t.columns = QUANTILE_COLS
    return t


def fit_error_quantiles(keys: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                        forecast: np.ndarray, min_obs: int = MIN_OBS):
    """Error quantiles on TRAIN rows: per series, per department, and overall."""
    err = pd.Series(y.to_numpy(float) - forecast)
    series = keys["series_id"].reset_index(drop=True)
    dept = X["dept_id"].astype(str).reset_index(drop=True)

    by_series = _quantile_table(err, series)
    by_series = by_series[series.value_counts().reindex(by_series.index) >= min_obs]
    by_dept = _quantile_table(err, dept)
    overall = err.quantile(LEVELS).to_numpy()
    return by_series, by_dept, overall


def predict_quantiles(keys: pd.DataFrame, X: pd.DataFrame, forecast: np.ndarray,
                      tables) -> pd.DataFrame:
    by_series, by_dept, overall = tables
    series = keys["series_id"].to_numpy()
    dept = X["dept_id"].astype(str).to_numpy()

    E = np.array(by_series.reindex(series), dtype=float)   # writable copy
    missing = np.isnan(E[:, 0])
    if missing.any():
        E[missing] = by_dept.reindex(dept[missing]).to_numpy(float)
    missing = np.isnan(E[:, 0])
    if missing.any():
        E[missing] = overall

    q = np.sort(np.maximum(forecast[:, None] + E, 0.0), axis=1)
    out = keys.reset_index(drop=True).copy()
    out[QUANTILE_COLS] = q
    return out


def _mean_pinball(y: np.ndarray, preds: pd.DataFrame) -> float:
    return float(np.mean([np.mean(np.maximum(tau * (y - preds[c]), (tau - 1) * (y - preds[c])))
                          for c, tau in QUANTILE_LEVELS.items()]))


def run(features: pd.DataFrame | None = None, save: bool = True) -> dict:
    feats = features if features is not None else load_features()
    X_tr, y_tr, k_tr = load_xy("train", feats)
    X_va, y_va, k_va = load_xy("val", feats)
    X_te, y_te, k_te = load_xy("test", feats)

    scores = {}
    for name, label in BASELINES.items():
        print(f"\n{name}: {label}")
        tables = fit_error_quantiles(k_tr, X_tr, y_tr, point_forecast(X_tr, name))
        print(f"  series with their own error distribution: {len(tables[0])}")

        val = predict_quantiles(k_va, X_va, point_forecast(X_va, name), tables)
        test = predict_quantiles(k_te, X_te, point_forecast(X_te, name), tables)
        scores[name] = {"val": _mean_pinball(y_va.to_numpy(float), val),
                        "test": _mean_pinball(y_te.to_numpy(float), test)}
        print(f"  mean pinball  val={scores[name]['val']:.4f}  test={scores[name]['test']:.4f}")

        if save:
            save_predictions(pd.concat([val, test], ignore_index=True), name, features=feats)
    return scores


if __name__ == "__main__":
    run()