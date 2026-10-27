"""Tiny fake M5 dataset in the real file format (for testing the pipeline offline)."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_raw_m5(raw_dir: Path, n_weeks: int = 90, n_items: int = 12, seed: int = 0) -> dict:
    """Write calendar.csv, sales_train_evaluation.csv, sell_prices.csv.

    Built to hit every special case of the real data:
      - sales end with a PARTIAL week (2 extra days)
      - calendar is longer than sales (hidden test period)
      - wm_yr_wk jumps at the year boundary (11152 -> 11201)
      - one item launches late (week 20): zeros and no price before
      - one item is missing in CA_3 (must be excluded from the sample)
      - one non-CA store (TX_1, must be excluded)
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_days = n_weeks * 7 + 2
    cal_days = n_days + 28
    dates = pd.date_range("2011-01-29", periods=cal_days)          # d_1 is a Saturday
    week_no = np.arange(cal_days) // 7
    wm_yr_wk = 11100 + 100 * (week_no // 52) + (week_no % 52) + 1

    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": wm_yr_wk,
        "weekday": dates.day_name(), "wday": dates.dayofweek + 1,
        "month": dates.month, "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(cal_days)],
        "event_name_1": [("Event" if i % 30 == 0 else None) for i in range(cal_days)],
        "event_type_1": [("Cultural" if i % 30 == 0 else None) for i in range(cal_days)],
        "event_name_2": [("Event2" if i % 90 == 0 else None) for i in range(cal_days)],
        "event_type_2": [("Sporting" if i % 90 == 0 else None) for i in range(cal_days)],
        "snap_CA": rng.integers(0, 2, cal_days), "snap_TX": rng.integers(0, 2, cal_days),
        "snap_WI": rng.integers(0, 2, cal_days),
    })
    cal.to_csv(raw_dir / "calendar.csv", index=False)

    stores = ["CA_1", "CA_2", "CA_3", "TX_1"]
    day_cols = [f"d_{i + 1}" for i in range(n_days)]
    sales_rows, price_rows = [], []
    launch_week_late = 20

    for i in range(n_items):
        dept = "FOODS_1" if i % 2 == 0 else "HOBBIES_1"
        cat = dept.split("_")[0]
        item = f"{dept}_{i:03d}"
        for store in stores:
            if i == n_items - 1 and store == "CA_3":
                continue                                              # item missing in CA_3
            launch = launch_week_late if i == 0 else 0
            base = rng.uniform(0.2, 6)
            daily = rng.poisson(base, n_days).astype(np.int16)
            daily[: launch * 7] = 0
            sales_rows.append({"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": dept,
                               "cat_id": cat, "store_id": store, "state_id": store.split("_")[0],
                               **dict(zip(day_cols, daily))})
            for w in sorted(set(wm_yr_wk[launch * 7:])):
                price_rows.append({"store_id": store, "item_id": item, "wm_yr_wk": int(w),
                                   "sell_price": round(float(rng.uniform(1, 9)), 2)})

    pd.DataFrame(sales_rows).to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(price_rows).to_csv(raw_dir / "sell_prices.csv", index=False)
    return {"n_weeks": n_weeks, "late_item": f"FOODS_1_{0:03d}", "late_week": launch_week_late,
            "two_store_item": f"{'FOODS_1' if (n_items - 1) % 2 == 0 else 'HOBBIES_1'}_{n_items - 1:03d}"}