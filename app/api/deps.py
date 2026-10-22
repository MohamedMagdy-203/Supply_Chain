"""Data loading for the API.

Predictions are read once per model and cached in memory.
Every predictions file must follow the shared contract:
    series_id, origin_week, q10, q25, q50, q75, q90, q95, q99
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", "predictions"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "fake")

QUANTILE_LEVELS = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}
QUANTILE_COLS = list(QUANTILE_LEVELS)
REQUIRED_COLS = ["series_id", "origin_week", *QUANTILE_COLS]

_MODEL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


@lru_cache(maxsize=8)
def _read(model: str) -> pd.DataFrame:
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
        return _read(model)
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


def available_models() -> list[str]:
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(p.stem for p in PREDICTIONS_DIR.glob("*.parquet"))