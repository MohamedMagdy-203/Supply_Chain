"""Data-loading helpers for the API.

All heavy I/O is done once per process and cached with lru_cache.

Caches
------
_read_predictions(model)  ->  DataFrame  [series_id, origin_week, q10..q99]
_read_weekly()            ->  DataFrame  [weekly.parquet – full history]
_read_metrics()           ->  DataFrame  [model_metrics.csv]
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Paths (overridable via environment variables for Docker / testing)
# ---------------------------------------------------------------------------
PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", "predictions"))
WEEKLY_PATH = Path(os.getenv("WEEKLY_PATH", "data/processed/weekly.parquet"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "reports/model_metrics.csv"))
INVENTORY_PATH = Path(os.getenv("INVENTORY_PATH", "data/processed/inventory.parquet"))

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "lgbm")
EXCLUDED_MODELS = {"fake"}

# ---------------------------------------------------------------------------
# Quantile contract (shared with src/policy.py)
# ---------------------------------------------------------------------------
QUANTILE_LEVELS: dict[str, float] = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}
QUANTILE_COLS: list[str] = list(QUANTILE_LEVELS)
REQUIRED_COLS: list[str] = ["series_id", "origin_week", *QUANTILE_COLS]

_MODEL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _read_predictions(model: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model}.parquet"
    if not path.exists():
        raise FileNotFoundError(model)
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return df


def get_predictions(model: str | None = None) -> pd.DataFrame:
    model = model or DEFAULT_MODEL
    if not _MODEL_NAME.match(model):
        raise HTTPException(status_code=422, detail="Invalid model name.")
    try:
        return _read_predictions(model)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for model '{model}'. "
                   f"Available: {available_models()}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def get_row(model: str | None, series_id: str, origin_week: int) -> pd.Series:
    df = get_predictions(model)
    match = df[(df["series_id"] == series_id) & (df["origin_week"] == origin_week)]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast for series_id='{series_id}', origin_week={origin_week}.",
        )
    return match.iloc[0]


@lru_cache(maxsize=1)
def _read_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(str(INVENTORY_PATH))
    df = pd.read_parquet(INVENTORY_PATH)
    required = {"series_id", "origin_week", "current_stock", "reserved_stock", "available_stock"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{INVENTORY_PATH.name} is missing columns: {missing}")
    return df


def get_inventory_stock(series_id: str, origin_week: int) -> pd.Series:
    try:
        inventory = _read_inventory()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"inventory.parquet not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    match = inventory[
        (inventory["series_id"] == series_id)
        & (inventory["origin_week"] == origin_week)
    ]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No inventory record for series_id='{series_id}', origin_week={origin_week}.",
        )
    return match.iloc[0]


def available_models() -> list[str]:
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(
        p.stem for p in PREDICTIONS_DIR.glob("*.parquet")
        if p.stem not in EXCLUDED_MODELS
    )


# ---------------------------------------------------------------------------
# Weekly history
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _read_weekly() -> pd.DataFrame:
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError(str(WEEKLY_PATH))
    df = pd.read_parquet(WEEKLY_PATH)
    # Ensure week_start is serialisable (convert to date string later)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def get_history(
    series_id: str,
    up_to_week: int | None = None,
    n_weeks: int | None = None,
) -> pd.DataFrame:
    """Return weekly actuals for *series_id*.

    Parameters
    ----------
    series_id : str
    up_to_week : int | None
        If given, only return rows with week_idx <= up_to_week.
    n_weeks : int | None
        If given, return only the last *n_weeks* rows (after the up_to filter).
    """
    try:
        weekly = _read_weekly()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"weekly.parquet not found: {exc}")

    rows = weekly[weekly["series_id"] == series_id].copy()
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for series_id='{series_id}'.",
        )
    rows = rows.sort_values("week_idx")
    if up_to_week is not None:
        rows = rows[rows["week_idx"] <= up_to_week]
    if n_weeks is not None:
        rows = rows.tail(n_weeks)
    return rows


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _read_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(str(METRICS_PATH))
    return pd.read_csv(METRICS_PATH)


def get_metrics() -> pd.DataFrame:
    try:
        return _read_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"model_metrics.csv not found: {exc}")
