"""Loading helpers and the predictions writer used by EVERY model.

Typical use inside a model script
---------------------------------
    from src.data import load_xy, save_predictions

    X_tr, y_tr, _ = load_xy("train")
    X_va, y_va, keys_va = load_xy("val")
    X_te, y_te, keys_te = load_xy("test")
    ...
    preds = keys_te.copy()          # series_id, origin_week
    preds[["q10", ..., "q99"]] = model_output
    save_predictions(preds, "lgbm")
"""
import re

import numpy as np
import pandas as pd

from src.config import KEY_COLS, PREDICTIONS_DIR, PROCESSED_DIR, QUANTILE_COLS
from src.features import CAT_FEATURES, FEATURES

SPLITS = ("train", "val", "test")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


def load_weekly() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "weekly.parquet")


def load_features(split: str | None = None) -> pd.DataFrame:
    """Read features.parquet. Never re-split: use the `split` column."""
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    if split is None:
        return df
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    return df[df["split"] == split].reset_index(drop=True)


def load_series_type() -> pd.DataFrame:
    """ADI / CV2 / demand_type per series (smooth, erratic, intermittent, lumpy)."""
    return pd.read_csv(PROCESSED_DIR / "series_type.csv")


def get_categories(features: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Category levels for store/dept/cat, taken from TRAIN rows only."""
    feats = features if features is not None else load_features()
    train = feats[feats["split"] == "train"]
    return {c: sorted(train[c].unique().tolist()) for c in CAT_FEATURES}


def load_xy(split: str, features: pd.DataFrame | None = None):
    """Return (X, y, keys) for one split.

    X     : FEATURES + CAT_FEATURES (categoricals as pandas 'category' with
            levels fixed from train, so train/val/test encode identically)
    y     : y_next
    keys  : series_id, origin_week (use these as the first two columns of your
            predictions file)
    """
    feats = features if features is not None else load_features()
    cats = get_categories(feats)
    part = feats[feats["split"] == split].reset_index(drop=True)
    X = part[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(part[c], categories=cats[c])
    if X[CAT_FEATURES].isna().any().any():
        raise ValueError("Unseen category level in this split (not present in train).")
    return X, part["y_next"].copy(), part[KEY_COLS].copy()


def save_predictions(preds: pd.DataFrame, model_name: str,
                     features: pd.DataFrame | None = None,
                     fix_crossing: bool = True, clip_zero: bool = True):
    """Validate against the contract and write predictions/{model_name}.parquet.

    Checks: columns, no NaN, no duplicate keys, keys exist in val/test rows of
    features.parquet, the whole TEST split is covered, quantiles are
    non-decreasing (fixed by sorting when fix_crossing=True) and non-negative.
    """
    if not _MODEL_NAME.match(model_name):
        raise ValueError("model_name may contain only letters, digits, _ and -")

    required = KEY_COLS + QUANTILE_COLS
    missing = [c for c in required if c not in preds.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = preds[required].copy()

    if df[QUANTILE_COLS].isna().any().any():
        raise ValueError("Predictions contain NaN.")
    if df.duplicated(KEY_COLS).any():
        raise ValueError("Duplicate (series_id, origin_week) rows.")

    feats = features if features is not None else load_features()
    valid = feats.loc[feats["split"].isin(["val", "test"]), KEY_COLS + ["split"]]

    merged = df.merge(valid, on=KEY_COLS, how="left")
    n_unknown = int(merged["split"].isna().sum())
    if n_unknown:
        raise ValueError(f"{n_unknown} rows have keys that are not val/test rows of features.parquet.")

    test_keys = valid[valid["split"] == "test"][KEY_COLS]
    n_missing = len(test_keys) - len(test_keys.merge(df[KEY_COLS], on=KEY_COLS))
    if n_missing:
        raise ValueError(f"{n_missing} test rows are missing from the predictions.")

    arr = df[QUANTILE_COLS].to_numpy(dtype=float)
    n_crossed = int((np.diff(arr, axis=1) < 0).any(axis=1).sum())
    if n_crossed:
        if not fix_crossing:
            raise ValueError(f"{n_crossed} rows have crossing quantiles.")
        arr = np.sort(arr, axis=1)
    n_negative = int((arr < 0).sum())
    if n_negative and clip_zero:
        arr = np.maximum(arr, 0.0)
    df[QUANTILE_COLS] = arr
    df["origin_week"] = df["origin_week"].astype(int)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{model_name}.parquet"
    df.to_parquet(path, index=False)

    n_val = int((merged["split"] == "val").sum())
    print(f"Saved {path}: {len(df)} rows (val={n_val}, test={len(df) - n_val}), "
          f"crossing fixed in {n_crossed} rows, negative values clipped: {n_negative}.")
    return path