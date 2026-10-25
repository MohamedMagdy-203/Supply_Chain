"""Train LightGBM quantile regressors using the repository's fixed time splits.

Hyperparameters are selected on validation data. Test targets are not used for
model selection. Predictions for validation and test are written only through
src.data.save_predictions.

Run from the repository root with:
    python -m src.models.lgbm
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import QUANTILE_LEVELS, SEED
from src.data import load_xy, save_predictions


@dataclass(frozen=True)
class Candidate:
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int


CANDIDATES = (
    Candidate(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=30),
    Candidate(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=30),
    Candidate(num_leaves=15, learning_rate=0.03, n_estimators=500, min_child_samples=50),
)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean quantile (pinball) loss for one quantile level."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def fit_quantile_models(X_train, y_train, X_eval, candidate: Candidate) -> dict[str, np.ndarray]:
    """Fit one LightGBM model per required quantile."""
    outputs = {}
    for name, alpha in QUANTILE_LEVELS.items():
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            num_leaves=candidate.num_leaves,
            learning_rate=candidate.learning_rate,
            n_estimators=candidate.n_estimators,
            min_child_samples=candidate.min_child_samples,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature="auto")
        outputs[name] = np.maximum(model.predict(X_eval), 0.0)
    return outputs


def score_validation(y_val: pd.Series, predictions: dict[str, np.ndarray]) -> float:
    """Average pinball loss across all required quantiles."""
    y = y_val.to_numpy(dtype=float)
    losses = [
        pinball_loss(y, predictions[name], alpha)
        for name, alpha in QUANTILE_LEVELS.items()
    ]
    return float(np.mean(losses))


def main() -> None:
    X_train, y_train, _ = load_xy("train")
    X_val, y_val, keys_val = load_xy("val")
    X_test, _, keys_test = load_xy("test")

    # Tune only on the validation split. Do not inspect test targets here.
    best_candidate = None
    best_val_predictions = None
    best_score = float("inf")
    for i, candidate in enumerate(CANDIDATES, start=1):
        val_predictions = fit_quantile_models(X_train, y_train, X_val, candidate)
        score = score_validation(y_val, val_predictions)
        print(f"Candidate {i}/{len(CANDIDATES)} {candidate}: validation pinball={score:.6f}")
        if score < best_score:
            best_score = score
            best_candidate = candidate
            best_val_predictions = val_predictions

    if best_candidate is None:
        raise RuntimeError("No LightGBM candidate completed successfully.")

    print(f"Selected parameters: {best_candidate}")
    print(f"Best validation pinball loss: {best_score:.6f}")

    # Use the selected candidate's validation predictions and predict test rows.
    # Test labels are not used for tuning or evaluation in this training script.
    val_predictions = best_val_predictions
    test_predictions = fit_quantile_models(X_train, y_train, X_test, best_candidate)

    val_frame = keys_val.copy()
    test_frame = keys_test.copy()
    for quantile in QUANTILE_LEVELS:
        val_frame[quantile] = val_predictions[quantile]
        test_frame[quantile] = test_predictions[quantile]

    predictions = pd.concat([val_frame, test_frame], ignore_index=True)
    save_predictions(predictions, "lgbm")


if __name__ == "__main__":
    main()
