"""Build every dataset the team needs from the raw M5 files.

    python -m scripts.build_dataset                       # defaults from src/config.py
    python -m scripts.build_dataset --n-items 50 --skip-audit

Steps
  1. load raw data
  2. audit of the raw data              -> audit_summary.json
  3. stratified item sample             -> sample_items.csv
  4. weekly aggregation + cleaning + time splits   -> weekly.parquet
  5. demand type of each series         -> series_type.csv
  6. leak-safe features + leakage test  -> features.parquet
  7. save outputs, data dictionary, reload and verify the contract
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import audit as audit_mod
from src import prepare
from src.config import (LAGS, MIN_HISTORY, N_ITEMS, PROCESSED_DIR, RAW_DIR, SEED,
                        STORES, TEST_WEEKS, VAL_WEEKS, WINDOWS)
from src.dictionary import DATA_DICTIONARY
from src.features import build_features, check_no_leakage


def _step(n: int, title: str) -> None:
    print(f"\n[{n}/7] {title}")


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, docs_dir: Path = Path("docs"),
        n_items: int = N_ITEMS, stores: list[str] = STORES, seed: int = SEED,
        run_audit: bool = True, audit_cells: int = 2_000_000) -> dict:
    raw_dir, out_dir, docs_dir = Path(raw_dir), Path(out_dir), Path(docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    _step(1, "Load raw data")
    sales_raw, cal, prices_raw, day_cols = prepare.load_raw(raw_dir)

    _step(2, "Data audit")
    if run_audit:
        audit = audit_mod.run_audit(sales_raw, cal, prices_raw, day_cols, seed=seed, n_cells=audit_cells)
        (out_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2))
    else:
        print("  skipped (--skip-audit)")

    _step(3, "Sample items")
    chosen, chosen_items, sales_s = prepare.sample_items(
        sales_raw, day_cols, stores, n_items, TEST_WEEKS, VAL_WEEKS, seed)
    chosen.to_csv(out_dir / "sample_items.csv", index=False)
    prices_s = prepare.filter_prices(prices_raw, stores, chosen_items)
    del sales_raw, prices_raw

    _step(4, "Weekly aggregation, cleaning and splits")
    snap_col = f"snap_{stores[0].split('_')[0]}"
    cal_sales, cal_week = prepare.build_weekly_calendar(cal, day_cols, snap_col)
    weekly = prepare.aggregate_weekly(sales_s, cal_sales, cal_week, day_cols)
    weekly = prepare.attach_prices_and_clean(weekly, prices_s)
    weekly, splits = prepare.add_splits(weekly, TEST_WEEKS, VAL_WEEKS)

    _step(5, "Demand type per series")
    series_type = prepare.compute_series_type(weekly)
    series_type.to_csv(out_dir / "series_type.csv", index=False)

    _step(6, "Features and leakage test")
    features = build_features(weekly)
    print(f"  feature table {features.shape} | rows per split: {features['split'].value_counts().to_dict()}")
    if not check_no_leakage(build_features, weekly):
        raise RuntimeError("Leakage detected in build_features. Nothing was saved.")

    _step(7, "Save and verify")
    weekly.to_parquet(out_dir / "weekly.parquet", index=False)
    features.to_parquet(out_dir / "features.parquet", index=False)
    (out_dir / "splits.json").write_text(json.dumps(
        {"splits_week_idx": splits, "seed": seed, "stores": stores, "lags": LAGS,
         "windows": WINDOWS, "min_history": MIN_HISTORY}, indent=2))
    (docs_dir / "data_dictionary.md").write_text(DATA_DICTIONARY, encoding="utf-8")

    _verify(out_dir)
    print("\nDone. Files written to", out_dir)
    return {"weekly": weekly.shape, "features": features.shape, "splits": splits}


def _verify(out_dir: Path) -> None:
    """Reload from disk and check the contract that teammates will actually load."""
    w = pd.read_parquet(out_dir / "weekly.parquet")
    f = pd.read_parquet(out_dir / "features.parquet")

    assert not w.duplicated(["series_id", "week_idx"]).any()
    assert not f.duplicated(["series_id", "origin_week"]).any()
    assert (f["origin_week"] == f["week_idx"] - 1).all()
    assert f["y_next"].notna().all()
    assert set(f["split"]) == {"train", "val", "test"}
    assert f.loc[f["split"] == "train", "week_idx"].max() < f.loc[f["split"] == "val", "week_idx"].min()
    assert f.loc[f["split"] == "val", "week_idx"].max() < f.loc[f["split"] == "test", "week_idx"].min()

    print(f"  weekly   {w.shape} | series {w['series_id'].nunique()}")
    print(f"  features {f.shape} | series {f['series_id'].nunique()}")
    print(f"  rows per split: {f['split'].value_counts().to_dict()}")
    print("  all contract checks passed")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--docs-dir", type=Path, default=Path("docs"))
    p.add_argument("--n-items", type=int, default=N_ITEMS)
    p.add_argument("--skip-audit", action="store_true", help="skip step 2 (faster re-runs)")
    a = p.parse_args(argv)
    run(a.raw_dir, a.out_dir, a.docs_dir, a.n_items, run_audit=not a.skip_audit)


if __name__ == "__main__":
    main()