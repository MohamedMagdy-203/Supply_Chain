"""Single source of truth for project constants.

Taken from the Preprocessing notebook (section 1.2). Change values here only,
never inside a model file.
"""
import os
from pathlib import Path

SEED = 42

# Folders (override with environment variables, e.g. inside Docker)
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", "predictions"))

# Scope
STORES = ["CA_1", "CA_2", "CA_3"]
N_ITEMS = 300

# Time split (in weeks, counted from the end of the data)
TEST_WEEKS = 12
VAL_WEEKS = 12

# Features
LAGS = [1, 2, 3, 4, 8, 13, 52]
WINDOWS = [4, 8, 13]
MIN_HISTORY = 13

# Probabilistic outputs (order matters: lowest to highest level)
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

# Keys shared by features.parquet and predictions/{model}.parquet
KEY_COLS = ["series_id", "origin_week"]