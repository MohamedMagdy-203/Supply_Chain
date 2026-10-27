"""Build a deterministic warehouse inventory snapshot for forecast replay weeks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "predictions" / "lgbm.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "inventory.parquet"


def build_inventory() -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS_PATH, columns=["series_id", "origin_week", "q50"])
    rows = []
    for row in predictions.itertuples(index=False):
        seed = int(hashlib.sha256(
            f"{row.series_id}:{row.origin_week}".encode("utf-8")
        ).hexdigest()[:8], 16)
        coverage_factor = 0.35 + (seed % 120) / 100
        reserved_ratio = 0.05 + ((seed // 120) % 16) / 100
        current_stock = max(0, round(float(row.q50) * coverage_factor))
        reserved_stock = min(current_stock, round(current_stock * reserved_ratio))
        rows.append({
            "series_id": row.series_id,
            "origin_week": int(row.origin_week),
            "current_stock": current_stock,
            "reserved_stock": reserved_stock,
            "available_stock": current_stock - reserved_stock,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    inventory = build_inventory()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(inventory):,} inventory records to {OUTPUT_PATH}")
