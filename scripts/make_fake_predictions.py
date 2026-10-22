import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
for s in range(10):
    for w in range(1, 13):
        base = rng.uniform(5, 50)
        q = np.sort(base * np.array([0.6, 0.75, 1.0, 1.25, 1.5, 1.65, 1.9]))
        rows.append([f"CA_1_ITEM_{s}", w, *q])

cols = ["series_id", "origin_week", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
pd.DataFrame(rows, columns=cols).to_parquet("predictions/fake.parquet", index=False)
print("saved predictions/fake.parquet")
