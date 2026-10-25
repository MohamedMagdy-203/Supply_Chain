
"""Ridge linear-regression baseline with empirical residual quantiles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import QUANTILE_LEVELS
from src.data import load_xy, save_predictions


@dataclass(frozen=True)
class Candidate:
    alpha: float


CANDIDATES = (
    Candidate(alpha=1.0),
    Candidate(alpha=10.0),
    Candidate(alpha=100.0),
)


def pinball_loss(y_true, y_pred, alpha):
    error = y_true - y_pred
    return float(
        np.mean(np.maximum(alpha * error, (alpha - 1.0) * error))
    )


def build_model(X_train, alpha):
    categorical = list(
        X_train.select_dtypes(include=["category", "object"]).columns
    )
    numeric = [
        column for column in X_train.columns
        if column not in categorical
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])


def residual_offsets(y_train, train_predictions):
    residuals = y_train.to_numpy(dtype=float) - train_predictions
    return {
        name: float(np.quantile(residuals, quantile))
        for name, quantile in QUANTILE_LEVELS.items()
    }


def make_quantile_predictions(model, X_eval, offsets):
    center = model.predict(X_eval)
    return {
        name: np.maximum(center + offsets[name], 0.0)
        for name in QUANTILE_LEVELS
    }


def score_validation(y_val, predictions):
    y = y_val.to_numpy(dtype=float)
    losses = [
        pinball_loss(y, predictions[name], quantile)
        for name, quantile in QUANTILE_LEVELS.items()
    ]
    return float(np.mean(losses))


def main():
    X_train, y_train, _ = load_xy("train")
    X_val, y_val, keys_val = load_xy("val")
    X_test, _, keys_test = load_xy("test")

    best_alpha = None
    best_model = None
    best_offsets = None
    best_val_predictions = None
    best_score = float("inf")

    for index, candidate in enumerate(CANDIDATES, start=1):
        model = build_model(X_train, candidate.alpha)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        offsets = residual_offsets(y_train, train_predictions)
        val_predictions = make_quantile_predictions(
            model, X_val, offsets
        )
        score = score_validation(y_val, val_predictions)

        print(
            f"Candidate {index}/{len(CANDIDATES)} {candidate}: "
            f"validation pinball={score:.6f}",
            flush=True,
        )

        if score < best_score:
            best_alpha = candidate
            best_model = model
            best_offsets = offsets
            best_val_predictions = val_predictions
            best_score = score

    if best_model is None:
        raise RuntimeError("No linear candidate completed successfully.")

    print(f"Selected parameters: {best_alpha}")
    print(f"Best validation pinball loss: {best_score:.6f}")

    test_predictions = make_quantile_predictions(
        best_model, X_test, best_offsets
    )

    val_frame = keys_val.copy()
    test_frame = keys_test.copy()

    for quantile in QUANTILE_LEVELS:
        val_frame[quantile] = best_val_predictions[quantile]
        test_frame[quantile] = test_predictions[quantile]

    predictions = pd.concat(
        [val_frame, test_frame], ignore_index=True
    )
    save_predictions(predictions, "linear")


if __name__ == "__main__":
    main()
