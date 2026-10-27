from __future__ import annotations

import numpy as np
from src.config import VAL_WEEKS
from src.data import load_features, load_xy
from src.features import CAT_FEATURES, FEATURES

#1. model features
DROP_COLUMNS = ["on_sale"]
NUM_FEATURES = [c for c in FEATURES if c not in DROP_COLUMNS]
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES


def select_features(X):
    return X[MODEL_FEATURES]


def load_splits():
    feats = load_features()
    out = {}
    for split in ("train", "val", "test"):
        X, y, keys = load_xy(split, feats)
        weeks = feats.loc[feats["split"] == split, "week_idx"].to_numpy()
        out[split] = (select_features(X), y, keys, weeks)
    return out


#2. CV folds
N_FOLDS = 3
FOLD_WEEKS = VAL_WEEKS                   # 12 weeks per fold, same length as val


def time_folds(weeks_train, weeks_val, n_folds=N_FOLDS, fold_weeks=FOLD_WEEKS):
    """Expanding-window folds over the stacked rows [train, val] (test is never included).

    Fold k validates on a block of `fold_weeks` target weeks and trains on every
    earlier week. The LAST fold is exactly the val split, so the choice is still
    made on val (modeling guide, rule 3), but checked on two earlier windows too.

        fold 0: train = weeks < val_start - 24   valid = the next 12 weeks
        fold 1: train = weeks < val_start - 12   valid = the next 12 weeks
        fold 2: train = all train weeks          valid = the val split

    Returns a list of (train_rows, valid_rows) positions into the stacked rows.
    """
    weeks = np.concatenate([weeks_train, weeks_val])
    val_start = int(np.min(weeks_val))
    folds = []
    for k in range(n_folds):
        start = val_start - fold_weeks * (n_folds - 1 - k)
        end = start + fold_weeks
        train_rows = np.flatnonzero(weeks < start)
        valid_rows = np.flatnonzero((weeks >= start) & (weeks < end))
        if len(train_rows) == 0 or len(valid_rows) == 0:
            raise ValueError(f"Fold {k} is empty: not enough weeks for {n_folds} folds of {fold_weeks} weeks.")
        folds.append((train_rows, valid_rows))
    return folds


#3. MLflow
MLFLOW_EXPERIMENT = "m5-inventory-forecasting"

def log_to_mlflow(run_name, model_family, params, metrics, fold="final"):
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"model_family": model_family, "fold": str(fold)})
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
