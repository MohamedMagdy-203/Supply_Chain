# Shared Data Contracts

## 1. data/processed/weekly.parquet
series_id, store_id, item_id, dept_id, cat_id, week_id, y, price, snap_days, n_events

## 2. features.parquet
Key: (series_id, origin_week)
Columns: features..., y_next

## 3. predictions/{model}.parquet
series_id, origin_week, q10, q25, q50, q75, q90, q95, q99
