"""Read-only check of every predictions/*.parquet (nothing is written or changed).

    python check_predictions.py
Run it from the project root. Do not commit this file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

Q = {"q10": .10, "q25": .25, "q50": .50, "q75": .75, "q90": .90, "q95": .95, "q99": .99}
KEY = ["series_id", "origin_week"]

p = argparse.ArgumentParser()
p.add_argument("--processed-dir", default="data/processed")
p.add_argument("--predictions-dir", default="predictions")
a = p.parse_args()

feats = pd.read_parquet(Path(a.processed_dir) / "features.parquet")
valid = feats.loc[feats["split"].isin(["val", "test"]), KEY + ["split", "y_next"]]
test = valid[valid["split"] == "test"]
print(f"Expected test rows: {len(test)}\n")

files = sorted(Path(a.predictions_dir).glob("*.parquet"))
if not files:
    print("No prediction files found.")
for path in files:
    df = pd.read_parquet(path)
    issues = []
    missing = [c for c in KEY + list(Q) if c not in df.columns]
    if missing:
        print(f"== {path.name}: MISSING COLUMNS {missing}\n")
        continue
    extra = [c for c in df.columns if c not in KEY + list(Q)]
    if extra:
        issues.append(f"extra columns {extra}")
    if df[list(Q)].isna().any().any():
        issues.append(f"NaN in {int(df[list(Q)].isna().any(axis=1).sum())} rows")
    if df.duplicated(KEY).any():
        issues.append(f"{int(df.duplicated(KEY).sum())} duplicate keys")

    m = df.merge(valid, on=KEY, how="left")
    n_unknown = int(m["split"].isna().sum())
    if n_unknown:
        issues.append(f"{n_unknown} rows with keys not in val/test")
    n_missing_test = len(test) - len(test.merge(df[KEY], on=KEY))
    if n_missing_test:
        issues.append(f"{n_missing_test} test rows missing")

    arr = df[list(Q)].to_numpy(float)
    crossed = int((np.diff(arr, axis=1) < 0).any(axis=1).sum())
    neg = int((arr < 0).sum())
    if crossed:
        issues.append(f"{crossed} rows with crossing quantiles")
    if neg:
        issues.append(f"{neg} negative values")

    t = m[m["split"] == "test"].dropna(subset=list(Q) + ["y_next"])
    y = t["y_next"].to_numpy(float)
    pin = np.mean([np.mean(np.maximum(tau * (y - t[c]), (tau - 1) * (y - t[c]))) for c, tau in Q.items()])
    cov80 = float(((y >= t["q10"]) & (y <= t["q90"])).mean())
    cov50 = float(((y >= t["q25"]) & (y <= t["q75"])).mean())
    wape = float(np.abs(y - t["q50"]).sum() / y.sum())

    n_val = int((m["split"] == "val").sum())
    print(f"== {path.name}: rows={len(df)} (val={n_val}, test={len(t)})")
    print(f"   mean pinball (test) = {pin:.4f} | coverage 80% = {cov80:.1%} (target 80%) | "
          f"coverage 50% = {cov50:.1%} (target 50%) | WAPE(q50) = {wape:.3f}")
    print("   OK" if not issues else "   PROBLEMS: " + "; ".join(issues))
    print()