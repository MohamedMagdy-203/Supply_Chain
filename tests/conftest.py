import numpy as np
import pandas as pd
import pytest


def make_weekly(n_series=6, n_weeks=90, seed=0):
    """Small synthetic weekly table with the same columns as weekly.parquet."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2011-01-29")
    for s in range(n_series):
        first = 0 if s < n_series - 1 else 30          # last series launches late
        base = rng.uniform(1, 20)
        for w in range(first, n_weeks):
            rows.append(dict(
                series_id=f"CA_{s % 3 + 1}__ITEM_{s}",
                store_id=f"CA_{s % 3 + 1}", item_id=f"ITEM_{s}",
                dept_id=f"DEPT_{s % 2}", cat_id="FOODS",
                wm_yr_wk=11101 + w, week_idx=w,
                week_start=start + pd.Timedelta(weeks=w),
                month=(w // 4) % 12 + 1, week_of_year=w % 52 + 1,
                y=int(rng.poisson(base)),
                sell_price=float(rng.uniform(1, 5)), on_sale=1,
                snap_days=int(rng.integers(0, 4)), n_events=int(rng.integers(0, 2)),
            ))
    df = pd.DataFrame(rows)
    last = df["week_idx"].max()
    test_start, val_start = last - 11, last - 23
    df["split"] = np.select([df["week_idx"] >= test_start, df["week_idx"] >= val_start],
                            ["test", "val"], default="train")
    return df


@pytest.fixture(scope="session")
def weekly():
    return make_weekly()