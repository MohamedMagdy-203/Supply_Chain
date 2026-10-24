"""Raw M5 -> weekly table. Moved from Preprocessing.ipynb, sections 2, 5, 6 and 7.2.

Order of use (see scripts/build_dataset.py):
    load_raw -> sample_items -> build_weekly_calendar -> aggregate_weekly
    -> attach_prices_and_clean -> add_splits -> compute_series_type
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEED

RAW_FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id"]

WEEKLY_COLS = ["series_id", "store_id", "item_id", "dept_id", "cat_id", "wm_yr_wk", "week_idx",
               "week_start", "month", "week_of_year", "y", "sell_price", "on_sale",
               "snap_days", "n_events"]


# ---------------------------------------------------------------- 2. load
def load_raw(raw_dir: Path):
    """Read the three M5 files. Returns (sales_raw, cal, prices_raw, day_cols)."""
    raw_dir = Path(raw_dir)
    for name in RAW_FILES:
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"Missing raw file: {raw_dir / name} "
                                    f"(run scripts/download_data.py first)")

    header = pd.read_csv(raw_dir / "sales_train_evaluation.csv", nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]

    sales_raw = pd.read_csv(raw_dir / "sales_train_evaluation.csv",
                            dtype={c: "int16" for c in day_cols})
    cal = pd.read_csv(raw_dir / "calendar.csv")
    prices_raw = pd.read_csv(raw_dir / "sell_prices.csv",
                             dtype={"store_id": "category", "item_id": "category",
                                    "wm_yr_wk": "int32", "sell_price": "float32"})
    print(f"  sales {sales_raw.shape} | calendar {cal.shape} | prices {prices_raw.shape}")
    return sales_raw, cal, prices_raw, day_cols


# ---------------------------------------------------------------- 5. sample
def sample_items(sales_raw: pd.DataFrame, day_cols: list[str], stores: list[str],
                 n_items: int, test_weeks: int, val_weeks: int, seed: int = SEED):
    """Stratified sample of items (department x volume tier), using TRAIN sales only.

    Returns (chosen, chosen_items, sales_s):
      chosen        DataFrame item_id, dept_id, tier, total (saved as sample_items.csv)
      chosen_items  sorted list of item ids
      sales_s       sales rows of the chosen items in the chosen stores
    """
    n_full_weeks = len(day_cols) // 7
    n_train_weeks = n_full_weeks - test_weeks - val_weeks
    train_days = day_cols[: n_train_weeks * 7]
    print(f"  full weeks {n_full_weeks} | train weeks {n_train_weeks} | train days {len(train_days)}")

    sales_ca = sales_raw[sales_raw["store_id"].isin(stores)].copy()
    sales_ca["train_total"] = sales_ca[train_days].sum(axis=1)

    items = (sales_ca.groupby(["item_id", "dept_id"])
                     .agg(n_stores=("store_id", "nunique"), total=("train_total", "sum"))
                     .reset_index())
    items = items[(items["n_stores"] == len(stores)) & (items["total"] > 0)].copy()

    rank_in_dept = items.groupby("dept_id")["total"].rank(method="first", pct=True)
    items["tier"] = pd.cut(rank_in_dept, bins=[0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"])
    items["group"] = items["dept_id"] + "_" + items["tier"].astype(str)

    frac = min(1.0, n_items / len(items))
    chosen = items.groupby("group", group_keys=False).sample(frac=frac, random_state=seed)
    chosen_items = sorted(chosen["item_id"])
    print(f"  eligible items {len(items)} | sampled items {len(chosen_items)} "
          f"| series {len(chosen_items) * len(stores)}")

    sales_s = sales_ca[sales_ca["item_id"].isin(chosen_items)].drop(columns="train_total")
    chosen = chosen[["item_id", "dept_id", "tier", "total"]]
    return chosen, chosen_items, sales_s


def filter_prices(prices_raw: pd.DataFrame, stores: list[str], chosen_items: list[str]) -> pd.DataFrame:
    prices_s = prices_raw[prices_raw["store_id"].isin(stores)
                          & prices_raw["item_id"].isin(chosen_items)].copy()
    prices_s["store_id"] = prices_s["store_id"].astype(str)
    prices_s["item_id"] = prices_s["item_id"].astype(str)
    prices_s["sell_price"] = prices_s["sell_price"].astype("float64")
    return prices_s


# ---------------------------------------------------------------- 6. weekly
def build_weekly_calendar(cal: pd.DataFrame, day_cols: list[str], snap_col: str):
    """Weekly calendar. Walmart weeks run Saturday-Friday; incomplete weeks are dropped.

    Returns (cal_sales, cal_week). `week_idx` is our own continuous counter:
    never use wm_yr_wk for time arithmetic (it jumps at the year boundary).
    """
    cal_sales = cal[cal["d"].isin(day_cols)].copy()
    cal_sales["date"] = pd.to_datetime(cal_sales["date"])
    cal_sales["n_events"] = (cal_sales["event_name_1"].notna().astype(int)
                             + cal_sales["event_name_2"].notna().astype(int))

    cal_week = (cal_sales.groupby("wm_yr_wk", as_index=False)
                         .agg(n_days=("d", "size"),
                              week_start=("date", "min"),
                              snap_days=(snap_col, "sum"),
                              n_events=("n_events", "sum")))

    partial = cal_week[cal_week["n_days"] != 7]
    print(f"  partial weeks dropped: {len(partial)} "
          f"({partial['n_days'].tolist() if len(partial) else 'none'} days)")

    cal_week = cal_week[cal_week["n_days"] == 7].sort_values("week_start").reset_index(drop=True)
    cal_week["week_idx"] = np.arange(len(cal_week))
    cal_week["month"] = cal_week["week_start"].dt.month
    cal_week["week_of_year"] = cal_week["week_start"].dt.isocalendar().week.astype(int)
    cal_week = cal_week.drop(columns="n_days")
    print(f"  full weeks: {len(cal_week)}")
    return cal_sales, cal_week


def aggregate_weekly(sales_s: pd.DataFrame, cal_sales: pd.DataFrame,
                     cal_week: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """Wide daily sales -> long -> weekly sum (the target y), joined with the weekly calendar."""
    long = sales_s.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype("int32")
    long["wm_yr_wk"] = long["d"].map(cal_sales.set_index("d")["wm_yr_wk"])

    weekly = (long.groupby(ID_COLS + ["wm_yr_wk"], as_index=False)["sales"].sum()
                  .rename(columns={"sales": "y"}))
    weekly = weekly.merge(cal_week, on="wm_yr_wk", how="inner")   # drops the partial week
    print(f"  long rows {len(long):,} | weekly rows {len(weekly):,}")
    return weekly


def attach_prices_and_clean(weekly: pd.DataFrame, prices_s: pd.DataFrame) -> pd.DataFrame:
    """Join prices and remove pre-launch weeks (no price = the product did not exist yet)."""
    weekly = weekly.merge(prices_s, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    weekly["series_id"] = weekly["store_id"] + "__" + weekly["item_id"]

    print(f"  rows before cleaning {len(weekly):,} | missing price {weekly['sell_price'].isna().mean():.1%}")

    first_week = (weekly.loc[weekly["sell_price"].notna()]
                        .groupby("series_id")["week_idx"].min().rename("first_week"))
    n_no_price = weekly["series_id"].nunique() - len(first_week)
    weekly = weekly.merge(first_week, on="series_id", how="inner")

    prelaunch = weekly[weekly["week_idx"] < weekly["first_week"]]
    print(f"  series without any price (removed): {n_no_price} | pre-launch rows: {len(prelaunch):,} "
          f"| units sold in them: {int(prelaunch['y'].sum())}")

    weekly = weekly[weekly["week_idx"] >= weekly["first_week"]].drop(columns="first_week")
    weekly["on_sale"] = weekly["sell_price"].notna().astype(int)

    weekly = weekly[WEEKLY_COLS].sort_values(["series_id", "week_idx"]).reset_index(drop=True)

    g = weekly.groupby("series_id")["week_idx"]
    assert ((g.max() - g.min() + 1) == g.size()).all(), "Some series have missing weeks"
    assert not weekly.duplicated(["series_id", "week_idx"]).any(), "Duplicated series-weeks"
    assert (weekly["y"] >= 0).all(), "Negative sales"
    print(f"  rows after cleaning {len(weekly):,} | series {weekly['series_id'].nunique()} "
          f"| weeks per series min/median/max: {g.size().min()}/{int(g.size().median())}/{g.size().max()}")
    return weekly


def add_splits(weekly: pd.DataFrame, test_weeks: int, val_weeks: int):
    """Time-based split by target week, counted from the end. Never random."""
    last_week = int(weekly["week_idx"].max())
    test_start = last_week - test_weeks + 1
    val_start = test_start - val_weeks

    weekly = weekly.copy()
    weekly["split"] = np.select([weekly["week_idx"] >= test_start, weekly["week_idx"] >= val_start],
                                ["test", "val"], default="train")
    splits = {"train": [0, val_start - 1], "val": [val_start, test_start - 1], "test": [test_start, last_week]}
    for part, (a, b) in splits.items():
        s = weekly.loc[weekly["week_idx"] == a, "week_start"].iloc[0].date()
        e = weekly.loc[weekly["week_idx"] == b, "week_start"].iloc[0].date()
        print(f"  {part:5s} weeks {a:>3d}-{b:<3d} from {s} to {e}")
    return weekly, splits


# ---------------------------------------------------------------- 7.2 demand type
def compute_series_type(weekly: pd.DataFrame) -> pd.DataFrame:
    """ADI / CV2 classification (smooth, erratic, intermittent, lumpy), TRAIN rows only."""
    train = weekly[weekly["split"] == "train"]
    n_weeks = train.groupby("series_id")["y"].size().rename("n_weeks")
    nonzero = (train[train["y"] > 0].groupby("series_id")["y"]
                    .agg(n_nonzero="size", mean_nz="mean", std_nz=lambda s: s.std(ddof=0)))

    st = pd.concat([n_weeks, nonzero], axis=1)
    st["n_nonzero"] = st["n_nonzero"].fillna(0)
    st["adi"] = st["n_weeks"] / st["n_nonzero"].replace(0, np.nan)
    st["cv2"] = (st["std_nz"] / st["mean_nz"]) ** 2

    st["demand_type"] = np.select(
        [st["adi"].isna(),
         (st["adi"] < 1.32) & (st["cv2"] < 0.49),
         (st["adi"] < 1.32) & (st["cv2"] >= 0.49),
         (st["adi"] >= 1.32) & (st["cv2"] < 0.49)],
        ["no_sales", "smooth", "erratic", "intermittent"], default="lumpy")
    print("  demand types:", st["demand_type"].value_counts().to_dict())
    return st[["adi", "cv2", "demand_type"]].reset_index()