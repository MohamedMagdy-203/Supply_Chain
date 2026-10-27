from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import shap  

from src.config import KEY_COLS, QUANTILE_LEVELS  
from src.data import load_xy 
from src.models.common import select_features
from src.models.lgbm import MODEL_NAME, MODELS_DIR  

SHAP_DIR = Path("reports/shap")
SAMPLE_SIZE = 4000                         # rows shown in the beeswarm plots
TOP_K = 10                                 # features listed in the local explanation


def load_models() -> dict[str, lgb.Booster]:
    models = {}
    for name in QUANTILE_LEVELS:
        path = MODELS_DIR / f"{MODEL_NAME}_{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 'python -m src.models.lgbm' first.")
        models[name] = lgb.Booster(model_file=str(path))
    return models


def load_rows():
    """Val + test rows, with the same columns the models were trained on."""
    X_val, y_val, keys_val = load_xy("val")
    X_test, y_test, keys_test = load_xy("test")
    X = select_features(pd.concat([X_val, X_test], ignore_index=True))
    y = pd.concat([y_val, y_test], ignore_index=True)
    keys = pd.concat([keys_val, keys_test], ignore_index=True)
    return X, y, keys


def shap_table(booster: lgb.Booster, X: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """One row per (series_id, origin_week): base value, prediction and one SHAP column per feature."""
    explainer = shap.TreeExplainer(booster)
    values = np.asarray(explainer.shap_values(X))
    table = keys.reset_index(drop=True).copy()
    table["base_value"] = float(np.ravel(explainer.expected_value)[0])
    table["prediction"] = booster.predict(X)
    table[list(X.columns)] = values.astype(np.float32)

    # check: base value + SHAP values must give back the prediction
    rebuilt = table["base_value"] + table[list(X.columns)].sum(axis=1)
    error = float(np.max(np.abs(rebuilt - table["prediction"])))
    print(f"  additivity check: max error {error:.5f}")
    return table


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Categories as integer codes (only for plot colours)."""
    X = X.copy()
    for column in X.select_dtypes(include=["category"]).columns:
        X[column] = X[column].cat.codes
    return X


def plot_global(tables: dict[str, pd.DataFrame], X: pd.DataFrame) -> None:
    features = list(X.columns)
    rng = np.random.default_rng(42)
    rows = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)

    for name in ["q50", "q90"]:
        plt.figure()
        shap.summary_plot(tables[name].iloc[rows][features].to_numpy(), numeric_view(X.iloc[rows]),
                          show=False, max_display=20)
        plt.title(f"SHAP summary, LightGBM {name}")
        plt.savefig(SHAP_DIR / f"shap_summary_{name}.png", dpi=130, bbox_inches="tight")
        plt.close("all")

    importance = pd.DataFrame({name: t[features].abs().mean() for name, t in tables.items()})
    importance = importance.sort_values("q50", ascending=False)
    importance.index.name = "feature"
    importance.to_csv(SHAP_DIR / "shap_importance.csv")

    ax = importance.head(15)[["q10", "q50", "q90"]].iloc[::-1].plot.barh(figsize=(8, 6), width=0.8)
    ax.set_xlabel("mean |SHAP| (units sold)")
    ax.set_title("Feature importance by quantile (LightGBM)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_importance.png", dpi=130)
    plt.close("all")


def explain_one(tables, X, y, keys, series_id, origin_week) -> None:
    """Waterfall plot (q50) + JSON (all quantiles) for one store-item-week."""
    match = np.flatnonzero((keys["series_id"] == series_id).to_numpy()
                           & (keys["origin_week"] == origin_week).to_numpy())
    if len(match) == 0:
        raise KeyError(f"No val/test row for {series_id} at origin_week {origin_week}")
    i = int(match[0])
    features = list(X.columns)
    row = X.iloc[i]

    result = {"series_id": series_id, "origin_week": int(origin_week),
              "target_week": int(origin_week) + 1, "actual": float(y.iloc[i]), "quantiles": {}}
    for name, table in tables.items():
        contrib = table.iloc[i][features].astype(float)
        top = contrib.abs().sort_values(ascending=False).index[:TOP_K]
        result["quantiles"][name] = {
            "base_value": round(float(table.iloc[i]["base_value"]), 3),
            "prediction": round(float(table.iloc[i]["prediction"]), 3),
            "top_features": [{"feature": f, "value": str(row[f]), "shap": round(float(contrib[f]), 3)}
                             for f in top],
        }

    stem = f"local_{series_id}_{origin_week}"
    (SHAP_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2))

    q50 = tables["q50"].iloc[i]
    explanation = shap.Explanation(
        values=q50[features].to_numpy(dtype=float),
        base_values=float(q50["base_value"]),
        data=numeric_view(X.iloc[[i]]).iloc[0].to_numpy(dtype=float),
        feature_names=features)
    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    p = result["quantiles"]
    plt.title(f"{series_id}, target week {origin_week + 1}\n"
              f"q10={p['q10']['prediction']:.1f}  q50={p['q50']['prediction']:.1f}  "
              f"q90={p['q90']['prediction']:.1f}  actual={result['actual']:.0f}", fontsize=10)
    plt.savefig(SHAP_DIR / f"{stem}.png", dpi=130, bbox_inches="tight")
    plt.close("all")


def default_example(y: pd.Series, keys: pd.DataFrame) -> tuple[str, int]:
    """The highest-selling series in the data shown, at its last origin week."""
    series_id = y.groupby(keys["series_id"]).sum().idxmax()
    return series_id, int(keys.loc[keys["series_id"] == series_id, "origin_week"].max())


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()
    X, y, keys = load_rows()

    tables = {}
    for name, booster in models.items():
        print(f"SHAP for {name} ...", flush=True)
        tables[name] = shap_table(booster, X, keys)
        tables[name].to_parquet(SHAP_DIR / f"shap_values_{name}.parquet", index=False)

    plot_global(tables, X)

    if len(sys.argv) == 3:
        series_id, origin_week = sys.argv[1], int(sys.argv[2])
    else:
        series_id, origin_week = default_example(y, keys)
    explain_one(tables, X, y, keys, series_id, origin_week)
    print(f"SHAP outputs written to {SHAP_DIR} (local example: {series_id} @ {origin_week})")


if __name__ == "__main__":
    main()
