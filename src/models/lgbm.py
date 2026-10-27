from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import QUANTILE_LEVELS, SEED
from src.data import save_predictions
from src.models.common import load_splits, log_to_mlflow, time_folds

MODEL_NAME = "lgbm"
MODELS_DIR = Path("models")


@dataclass(frozen=True)
class Candidate:
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int


CANDIDATES = (
    Candidate(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=30),
    Candidate(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=30),
    Candidate(num_leaves=63, learning_rate=0.03, n_estimators=600, min_child_samples=20),
    Candidate(num_leaves=15, learning_rate=0.03, n_estimators=500, min_child_samples=50),
)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean quantile (pinball) loss for one quantile level."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def fit_quantile_models(X_train, y_train, candidate: Candidate) -> dict[str, LGBMRegressor]:
    """Fit one LightGBM model per quantile and return the fitted models."""
    models = {}
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
        model.fit(X_train, y_train, categorical_feature="auto")   # store/dept/cat are pandas categories
        models[name] = model
    return models


def predict_quantiles(models: dict[str, LGBMRegressor], X) -> dict[str, np.ndarray]:
    """Predict every quantile; negative demand is clipped to 0."""
    return {name: np.maximum(model.predict(X), 0.0) for name, model in models.items()}


def score_validation(y_val: pd.Series, predictions: dict[str, np.ndarray]) -> float:
    """Average pinball loss across all required quantiles."""
    y = y_val.to_numpy(dtype=float)
    losses = [pinball_loss(y, predictions[name], alpha) for name, alpha in QUANTILE_LEVELS.items()]
    return float(np.mean(losses))


def per_quantile_scores(y, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Pinball loss of each quantile, e.g. {"val_pinball_q10": ..., ..., "val_pinball_q99": ...}."""
    y = y.to_numpy(dtype=float)
    return {f"val_pinball_{name}": pinball_loss(y, predictions[name], alpha)
            for name, alpha in QUANTILE_LEVELS.items()}


def cv_score(X_tv, y_tv, folds, candidate: Candidate):
    """Score one candidate on every fold.

    Returns the per-fold scores and the predictions of the last fold
    (= the val rows predicted by a model trained on train only).
    """
    scores, last_predictions = [], None
    for train_rows, valid_rows in folds:
        models = fit_quantile_models(X_tv.iloc[train_rows], y_tv.iloc[train_rows], candidate)
        last_predictions = predict_quantiles(models, X_tv.iloc[valid_rows])
        scores.append(score_validation(y_tv.iloc[valid_rows], last_predictions))
    return scores, last_predictions


def save_models(models: dict[str, LGBMRegressor]) -> None:
    """Save the final boosters so SHAP (src/explain.py) can load them later."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.booster_.save_model(str(MODELS_DIR / f"{MODEL_NAME}_{name}.txt"))


def main() -> None:
    splits = load_splits()
    X_train, y_train, _, weeks_train = splits["train"]
    X_val, y_val, keys_val, weeks_val = splits["val"]
    X_test, _, keys_test, _ = splits["test"]          # test targets are never used here

    # train + val stacked (train first); the folds index into these rows
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    folds = time_folds(weeks_train, weeks_val)

    # 1. Choose the parameters: mean pinball over the 3 folds.
    best = {"score": float("inf")}
    for i, candidate in enumerate(CANDIDATES, start=1):
        fold_scores, val_predictions = cv_score(X_tv, y_tv, folds, candidate)
        score = float(np.mean(fold_scores))
        print(f"Candidate {i}/{len(CANDIDATES)} {candidate}: "
              f"folds={[round(s, 4) for s in fold_scores]} mean={score:.6f}", flush=True)
        if score < best["score"]:
            best = {"score": score, "candidate": candidate,
                    "fold_scores": fold_scores, "val_predictions": val_predictions}

    if "candidate" not in best:
        raise RuntimeError("No LightGBM candidate completed successfully.")
    print(f"Selected parameters: {best['candidate']}")
    print(f"CV pinball: {best['score']:.6f}  (val fold: {best['fold_scores'][-1]:.6f})")

    # 2. Refit the chosen parameters on train + val, then predict test.
    final_models = fit_quantile_models(X_tv, y_tv, best["candidate"])
    test_predictions = predict_quantiles(final_models, X_test)
    save_models(final_models)

    # 3. Write val + test predictions in the shared format.
    val_frame = keys_val.copy()
    test_frame = keys_test.copy()
    for quantile in QUANTILE_LEVELS:
        val_frame[quantile] = best["val_predictions"][quantile]
        test_frame[quantile] = test_predictions[quantile]
    save_predictions(pd.concat([val_frame, test_frame], ignore_index=True), MODEL_NAME)

    # 4. MLflow: one run per fold of the chosen settings + the final run.
    params = asdict(best["candidate"])
    for k, fold_score in enumerate(best["fold_scores"]):
        log_to_mlflow(f"{MODEL_NAME}_fold{k}", "lgbm", params, {"pinball_mean": fold_score}, fold=k)
    log_to_mlflow(MODEL_NAME, "lgbm", params,
                  {"cv_pinball_mean": best["score"], "val_pinball_mean": best["fold_scores"][-1],
                   **per_quantile_scores(y_val, best["val_predictions"])})


if __name__ == "__main__":
    main()