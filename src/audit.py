"""Data audit (Pillar 1). Moved from Preprocessing.ipynb, section 3.

Runs on the RAW data (all series, before sampling) and returns one dictionary
that the pipeline saves as audit_summary.json. No plots here: plotting stays
in the EDA notebook.
"""
import numpy as np
import pandas as pd

from src.config import SEED

ID_COLS_ALL = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def run_audit(sales_raw: pd.DataFrame, cal: pd.DataFrame, prices_raw: pd.DataFrame,
              day_cols: list[str], seed: int = SEED, n_cells: int = 2_000_000) -> dict:
    audit: dict = {}

    # 3.1 schema and cardinality
    tables = {}
    for name, df in [("sales", sales_raw), ("calendar", cal), ("sell_prices", prices_raw)]:
        mb = df.memory_usage(deep=True).sum() / 1e6
        tables[name] = {"rows": int(len(df)), "cols": int(df.shape[1]), "memory_mb": round(float(mb), 1)}
        print(f"  {name:12s} rows={len(df):>10,}  cols={df.shape[1]:>5}  memory={mb:6.0f} MB")
    cardinality = {c: int(sales_raw[c].nunique()) for c in ID_COLS_ALL}
    cardinality["event_name_1"] = int(cal["event_name_1"].nunique())
    audit["tables"] = tables
    audit["n_series_raw"] = int(len(sales_raw))
    audit["n_days_sales"] = len(day_cols)
    audit["cardinality"] = cardinality
    print("  cardinality:", cardinality)

    # 3.2 null map
    audit["null_pct_calendar"] = cal.isna().mean().mul(100).round(2).to_dict()
    audit["null_pct_prices"] = prices_raw.isna().mean().mul(100).round(2).to_dict()
    audit["null_cells_sales"] = int(sales_raw.isna().sum().sum())

    # 3.3 duplicates and time coverage
    dup_sales = int(sales_raw.duplicated(subset=["item_id", "store_id"]).sum())
    dup_prices = int(prices_raw.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).sum())
    dates = pd.to_datetime(cal["date"])
    last_sales_date = dates[cal["d"] == day_cols[-1]].iloc[0]
    audit["duplicates"] = {"sales_series": dup_sales, "price_rows": dup_prices}
    audit["calendar_has_no_gaps"] = bool((dates.diff().dropna().dt.days == 1).all())
    audit["sales_days_in_calendar"] = bool(set(day_cols) <= set(cal["d"]))
    audit["sales_first_date"] = str(dates.iloc[0].date())
    audit["sales_last_date"] = str(last_sales_date.date())
    print(f"  duplicated series={dup_sales}, duplicated price rows={dup_prices}, "
          f"calendar has gaps: {not audit['calendar_has_no_gaps']}")
    print(f"  sales range {audit['sales_first_date']} -> {audit['sales_last_date']} "
          f"({len(day_cols)} days = {len(day_cols) // 7} full weeks + {len(day_cols) % 7} extra day(s))")

    # 3.4 target distribution (estimated on a random sample of cells)
    vals = sales_raw[day_cols].to_numpy()
    rng = np.random.default_rng(seed)
    rows_i = rng.integers(0, vals.shape[0], n_cells)
    cols_i = rng.integers(0, vals.shape[1], n_cells)
    sample_vals = vals[rows_i, cols_i]
    pcts = np.percentile(sample_vals, [50, 90, 99, 99.9, 100])
    audit["daily_zero_share"] = float((sample_vals == 0).mean())
    audit["daily_percentiles_50_90_99_99.9_max"] = [float(p) for p in pcts]
    audit["daily_mean"] = float(sample_vals.mean())
    audit["daily_std"] = float(sample_vals.std())
    print(f"  share of zero daily cells: {audit['daily_zero_share']:.1%}")

    # 3.5 sanity checks
    audit["negative_sales"] = bool((vals < 0).any())
    audit["min_price"] = float(prices_raw["sell_price"].min())
    audit["non_positive_prices"] = int((prices_raw["sell_price"] <= 0).sum())
    if audit["negative_sales"] or audit["non_positive_prices"] or dup_sales or dup_prices:
        print("  WARNING: the audit found data errors, check audit_summary.json")
    return audit