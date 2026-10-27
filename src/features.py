"""Leak-safe feature pipeline (moved from Preprocessing.ipynb, section 8-9).

Convention
----------
Each row is built around the TARGET week T. The origin is origin_week = T - 1.
  y_next = sales of week T
  lag_k  = sales of week T - k
Every feature built from sales is shifted by at least 1 week. Price, SNAP and
events of week T itself are allowed (known-future covariates, documented
assumption).
"""
import numpy as np
import pandas as pd

from src.config import LAGS, MIN_HISTORY, WINDOWS

KEYS = ["series_id", "store_id", "item_id", "dept_id", "cat_id",
        "origin_week", "week_idx", "wm_yr_wk", "split", "y_next"]

FEATURES = ([f"lag_{k}" for k in LAGS]
            + [f"roll_mean_{w}" for w in WINDOWS]
            + [f"roll_std_{w}" for w in WINDOWS]
            + ["zero_share_13", "weeks_since_sale",
               "sell_price", "price_change_1w", "on_sale", "snap_days", "n_events",
               "month", "week_of_year", "history_weeks", "has_lag52"])

CAT_FEATURES = ["store_id", "dept_id", "cat_id"]


def weeks_since_sale(s: pd.Series) -> pd.Series:
    """Weeks since the last week with sales, as known at the origin (T-1)."""
    pos = np.arange(len(s))
    last_sale = pd.Series(np.where(s.values > 0, pos, np.nan), index=s.index).ffill()
    return (pos - last_sale).shift(1)


def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Build the model table. Row = target week T, origin = T-1."""
    df = weekly.sort_values(["series_id", "week_idx"]).reset_index(drop=True).copy()
    by_series = df.groupby("series_id")["y"]

    for k in LAGS:
        df[f"lag_{k}"] = by_series.shift(k)

    # shift(1) BEFORE rolling so the window covers T-w ... T-1
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = by_series.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = by_series.transform(lambda s, w=w: s.shift(1).rolling(w).std())

    df["zero_share_13"] = by_series.transform(
        lambda s: (s == 0).astype(float).shift(1).rolling(13).mean())
    df["weeks_since_sale"] = by_series.transform(weeks_since_sale)

    prev_price = df.groupby("series_id")["sell_price"].shift(1)
    df["price_change_1w"] = df["sell_price"] / prev_price - 1

    df["history_weeks"] = df.groupby("series_id").cumcount()
    df["has_lag52"] = df["lag_52"].notna().astype(int)

    df["origin_week"] = df["week_idx"] - 1
    df["y_next"] = df["y"]

    df = df[df["history_weeks"] >= MIN_HISTORY]
    return df[KEYS + FEATURES].reset_index(drop=True)


def check_no_leakage(build_fn, weekly: pd.DataFrame, n_origins: int = 6, seed: int = 0) -> bool:
    """Truncation test.

    For random origins o: cut the data at week o+1, hide the sales of week o+1,
    rebuild features, and compare the rows of origin o with the full build.
    Any difference means the feature used future information.
    """
    full = build_fn(weekly).set_index(["series_id", "origin_week"]).sort_index()
    cols = [c for c in full.select_dtypes("number").columns if c != "y_next"]

    origins_available = full.index.get_level_values("origin_week").unique().to_numpy()
    rng = np.random.default_rng(seed)
    origins = rng.choice(origins_available,
                         size=min(n_origins, len(origins_available)), replace=False)

    bad_cols = set()
    for o in origins:
        cut = weekly[weekly["week_idx"] <= o + 1].copy()
        cut.loc[cut["week_idx"] == o + 1, "y"] = np.nan

        part = build_fn(cut).set_index(["series_id", "origin_week"]).sort_index()
        a = part.xs(o, level="origin_week")[cols]
        b = full.xs(o, level="origin_week")[cols].reindex(a.index)

        for c in cols:
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True):
                bad_cols.add(c)

    if bad_cols:
        print("LEAKAGE FOUND in:", sorted(bad_cols))
        return False
    print(f"No leakage found ({len(origins)} random origins checked).")
    return True