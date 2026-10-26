
"""Compare deep, LightGBM, Ridge-linear, and quantile-regression forecasts on validation, then evaluate test.

IMPORTANT:
- Run this script only after model selection and tuning are complete.
- Validation scores are used for comparison; test scores are final reporting only.
- Do not change models or hyperparameters based on the test results.

Run from the repository root:
    python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import KEY_COLS, PREDICTIONS_DIR, QUANTILE_LEVELS
from src.data import load_features

MODEL_NAMES = ("deep", "lgbm", "linear", "quantreg")


def mean_pinball_loss(y_true: np.ndarray, predictions: pd.DataFrame) -> float:
    """Mean pinball loss across every row and all configured quantiles."""
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    quantiles = list(QUANTILE_LEVELS.values())
    pred = predictions[list(QUANTILE_LEVELS)].to_numpy(dtype=float)

    if pred.shape != (len(y), len(quantiles)):
        raise ValueError(
            f"Expected prediction shape {(len(y), len(quantiles))}, got {pred.shape}."
        )
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets or predictions contain NaN/inf values.")

    errors = y - pred
    levels = np.asarray(quantiles, dtype=float).reshape(1, -1)
    losses = np.maximum(levels * errors, (levels - 1.0) * errors)
    return float(losses.mean())


def load_and_validate_predictions(model_name: str) -> pd.DataFrame:
    """Read a model file and check required columns and unique forecast keys."""
    path = PREDICTIONS_DIR / f"{model_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.models.{model_name}` first."
        )

    predictions = pd.read_parquet(path)
    required = KEY_COLS + list(QUANTILE_LEVELS)
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    predictions = predictions[required].copy()

    if predictions.duplicated(KEY_COLS).any():
        raise ValueError(f"{path} contains duplicate forecast keys.")
    if predictions[list(QUANTILE_LEVELS)].isna().any().any():
        raise ValueError(f"{path} contains missing quantile predictions.")
    if not np.isfinite(predictions[list(QUANTILE_LEVELS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains infinite quantile predictions.")
    return predictions


def score_split(
    model_name: str,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    split_name: str,
) -> float:
    """Score a split only after verifying exact one-to-one key coverage."""
    truth_keys = truth[KEY_COLS]
    split_predictions = predictions.merge(
        truth_keys,
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
    )

    # Compare key sets, not just row counts: this catches one missing row plus
    # one unexpected row, which a row-count-only check would otherwise miss.
    expected_keys = set(
        map(tuple, truth_keys[KEY_COLS].itertuples(index=False, name=None))
    )
    actual_keys = set(
        map(tuple, split_predictions[KEY_COLS].itertuples(index=False, name=None))
    )
    if actual_keys != expected_keys or len(split_predictions) != len(truth):
        missing_count = len(expected_keys - actual_keys)
        extra_count = len(actual_keys - expected_keys)
        raise ValueError(
            f"{model_name}: {split_name} key mismatch; "
            f"missing={missing_count}, unexpected={extra_count}, "
            f"matched={len(split_predictions)}, expected={len(truth)}."
        )

    # Align targets explicitly by forecast key.
    scored = predictions.merge(
        truth[KEY_COLS + ["y_next"]],
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(scored) != len(truth):
        raise ValueError(f"{model_name}: not all {split_name} targets were matched.")

    # Check quantile ordering and nonnegative sales forecasts.
    pred_values = scored[list(QUANTILE_LEVELS)].to_numpy(dtype=float)
    if (np.diff(pred_values, axis=1) < -1e-10).any():
        raise ValueError(
            f"{model_name}: crossing quantiles found in {split_name} predictions."
        )
    if (pred_values < 0).any():
        raise ValueError(
            f"{model_name}: negative predictions found in {split_name} predictions."
        )

    return mean_pinball_loss(scored["y_next"].to_numpy(dtype=float), scored)


def main() -> None:
    # Use exactly the existing validation and test splits.
    validation = load_features("val")[KEY_COLS + ["y_next"]].copy()
    test = load_features("test")[KEY_COLS + ["y_next"]].copy()

    if validation.duplicated(KEY_COLS).any() or test.duplicated(KEY_COLS).any():
        raise ValueError(
            "Feature data contains duplicate (series_id, origin_week) keys."
        )

    predictions_by_model = {
        name: load_and_validate_predictions(name) for name in MODEL_NAMES
    }

    # Require each model to contain exactly the validation + test forecast keys.
    expected_all_keys = set(
        map(
            tuple,
            pd.concat(
                [validation[KEY_COLS], test[KEY_COLS]], ignore_index=True
            ).itertuples(index=False, name=None),
        )
    )
    for model_name, predictions in predictions_by_model.items():
        model_keys = set(
            map(tuple, predictions[KEY_COLS].itertuples(index=False, name=None))
        )
        if model_keys != expected_all_keys:
            raise ValueError(
                f"{model_name}: prediction key set does not exactly match the "
                f"combined validation + test rows; "
                f"missing={len(expected_all_keys - model_keys)}, "
                f"unexpected={len(model_keys - expected_all_keys)}."
            )

    print("Validation mean pinball loss (lower is better)")
    print("=" * 48)

    validation_results = []
    for model_name, predictions in predictions_by_model.items():
        score = score_split(model_name, predictions, validation, "validation")
        validation_results.append((model_name, score))

    validation_results.sort(key=lambda item: item[1])

    for rank, (model_name, score) in enumerate(validation_results, start=1):
        print(f"{rank}. {model_name:<10} {score:.6f}")

    best_name, best_score = validation_results[0]
    print(f"\nValidation winner: {best_name} ({best_score:.6f})")

    # Final test evaluation: report only, do not use it to select or tune models.
    print("\nFINAL TEST mean pinball loss (reporting only; do not retune)")
    print("=" * 58)

    test_results = []
    for model_name, predictions in predictions_by_model.items():
        score = score_split(model_name, predictions, test, "test")
        test_results.append((model_name, score))

    test_results.sort(key=lambda item: item[1])

    for rank, (model_name, score) in enumerate(test_results, start=1):
        print(f"{rank}. {model_name:<10} {score:.6f}")


if __name__ == "__main__":
    main()
