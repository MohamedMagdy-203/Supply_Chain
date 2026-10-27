from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import QUANTILE_LEVELS
from src.data import save_predictions
from src.models.common import load_splits, log_to_mlflow, time_folds

RIDGE_NAME = "linear"
QUANTREG_NAME = "quantreg"
QUANTREG_DROP = ["roll_mean_4", "cat_id"]
RIDGE_ALPHAS = (10.0, 100.0, 1000.0, 10000.0, 30000.0)


#shared helpers
def pinball_loss(y_true, y_pred, alpha):
    """Mean quantile (pinball) loss for one quantile level."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def score_validation(y_val, predictions):
    """Average pinball loss across all required quantiles."""
    y = y_val.to_numpy(dtype=float)
    return float(np.mean([pinball_loss(y, predictions[name], q) for name, q in QUANTILE_LEVELS.items()]))


def make_preprocessor(X, drop_first=False):
    """Median-impute + scale numeric columns, one-hot encode categorical columns."""
    categorical = list(X.select_dtypes(include=["category"]).columns)
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(drop="first" if drop_first else None,
                                      sparse_output=False), categorical),
    ])


def per_quantile_scores(y, predictions):
    y = y.to_numpy(dtype=float)
    return {f"val_pinball_{name}": pinball_loss(y, predictions[name], q) for name, q in QUANTILE_LEVELS.items()}


def to_frame(keys_val, val_predictions, keys_test, test_predictions):
    """Stack val + test predictions in the shared format."""
    val_frame, test_frame = keys_val.copy(), keys_test.copy()
    for quantile in QUANTILE_LEVELS:
        val_frame[quantile] = val_predictions[quantile]
        test_frame[quantile] = test_predictions[quantile]
    return pd.concat([val_frame, test_frame], ignore_index=True)


#Ridge
def residual_scale(X):
    """Per-row scale for the residuals: the item's recent volatility."""
    return X["roll_std_13"].fillna(0).to_numpy(dtype=float) + 1.0


def fit_ridge(X, y, alpha):
    """Fit Ridge and compute the scaled residual quantiles on the same rows."""
    model = Pipeline([("preprocess", make_preprocessor(X)), ("regressor", Ridge(alpha=alpha))])
    model.fit(X, y)
    residuals = (y.to_numpy(dtype=float) - model.predict(X)) / residual_scale(X)
    offsets = {name: float(np.quantile(residuals, q)) for name, q in QUANTILE_LEVELS.items()}
    return model, offsets


def predict_ridge(model, offsets, X):
    center, scale = model.predict(X), residual_scale(X)
    return {name: np.maximum(center + scale * offsets[name], 0.0) for name in QUANTILE_LEVELS}


def ridge_cv_score(X_tv, y_tv, folds, alpha):
    """Per-fold scores for one alpha, plus the last fold's predictions (= val rows)."""
    scores, last_predictions = [], None
    for train_rows, valid_rows in folds:
        model, offsets = fit_ridge(X_tv.iloc[train_rows], y_tv.iloc[train_rows], alpha)
        last_predictions = predict_ridge(model, offsets, X_tv.iloc[valid_rows])
        scores.append(score_validation(y_tv.iloc[valid_rows], last_predictions))
    return scores, last_predictions


def run_ridge(X_tv, y_tv, folds, y_val, keys_val, X_test, keys_test):
    best = {"score": float("inf")}
    for i, alpha in enumerate(RIDGE_ALPHAS, start=1):
        fold_scores, val_predictions = ridge_cv_score(X_tv, y_tv, folds, alpha)
        score = float(np.mean(fold_scores))
        print(f"Ridge {i}/{len(RIDGE_ALPHAS)} alpha={alpha}: "
              f"folds={[round(s, 4) for s in fold_scores]} mean={score:.6f}", flush=True)
        if score < best["score"]:
            best = {"score": score, "alpha": alpha, "fold_scores": fold_scores,
                    "val_predictions": val_predictions}
    print(f"Ridge selected alpha={best['alpha']}, CV pinball={best['score']:.6f} "
          f"(val fold: {best['fold_scores'][-1]:.6f})")

    # refit on train + val, then predict test
    model, offsets = fit_ridge(X_tv, y_tv, best["alpha"])
    test_predictions = predict_ridge(model, offsets, X_test)
    save_predictions(to_frame(keys_val, best["val_predictions"], keys_test, test_predictions), RIDGE_NAME)

    params = {"model": "ridge", "alpha": best["alpha"]}
    for k, fold_score in enumerate(best["fold_scores"]):
        log_to_mlflow(f"{RIDGE_NAME}_fold{k}", "linear", params, {"pinball_mean": fold_score}, fold=k)
    log_to_mlflow(RIDGE_NAME, "linear", params,
                  {"cv_pinball_mean": best["score"], "val_pinball_mean": best["fold_scores"][-1],
                   **per_quantile_scores(y_val, best["val_predictions"])})


#Quantile Regression
def fit_quantreg(X, y):
    """One statsmodels QuantReg per quantile on a shared preprocessor."""
    import statsmodels.api as sm
    X = X.drop(columns=QUANTREG_DROP)
    preprocessor = make_preprocessor(X, drop_first=True)
    A = sm.add_constant(preprocessor.fit_transform(X), has_constant="add")
    params = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        for name, q in QUANTILE_LEVELS.items():
            params[name] = sm.QuantReg(y.to_numpy(dtype=float), A).fit(q=q, max_iter=2000).params
    return preprocessor, params


def predict_quantreg(preprocessor, params, X):
    A = preprocessor.transform(X.drop(columns=QUANTREG_DROP))
    A = np.column_stack([np.ones(len(A)), A])
    return {name: np.maximum(A @ p, 0.0) for name, p in params.items()}


def run_quantreg(X_train, y_train, X_val, y_val, keys_val, X_tv, y_tv, X_test, keys_test):
    preprocessor, params = fit_quantreg(X_train, y_train)
    val_predictions = predict_quantreg(preprocessor, params, X_val)
    score = score_validation(y_val, val_predictions)
    print(f"Quantile regression: validation pinball={score:.6f}", flush=True)

    preprocessor, params = fit_quantreg(X_tv, y_tv)
    test_predictions = predict_quantreg(preprocessor, params, X_test)

    save_predictions(to_frame(keys_val, val_predictions, keys_test, test_predictions), QUANTREG_NAME)
    log_to_mlflow(QUANTREG_NAME, "linear", {"model": "quantreg"},
                  {"val_pinball_mean": score, **per_quantile_scores(y_val, val_predictions)})


#main
def main():
    splits = load_splits()
    X_train, y_train, _, weeks_train = splits["train"]
    X_val, y_val, keys_val, weeks_val = splits["val"]
    X_test, _, keys_test, _ = splits["test"]          # test targets are never used here

    # train + val stacked (train first); the folds index into these rows
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    folds = time_folds(weeks_train, weeks_val)

    run_ridge(X_tv, y_tv, folds, y_val, keys_val, X_test, keys_test)
    run_quantreg(X_train, y_train, X_val, y_val, keys_val, X_tv, y_tv, X_test, keys_test)


if __name__ == "__main__":
    main()