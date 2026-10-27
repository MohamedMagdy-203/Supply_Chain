import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PREDICTIONS_DIR, QUANTILE_LEVELS
from src.data import load_features, load_series_type

MODELS = ["linear", "quantreg", "lgbm"]
OUTPUT = Path("reports/member3_models.md")
SHAP_IMPORTANCE = Path("reports/shap/shap_importance.csv")

CU = 0.30                                  # cost of stock-out, share of unit price
CO = 0.10                                  # cost of overstock, share of unit price
TAU = CU / (CU + CO)                       # critical ratio = 0.75 -> order up to q75
ORDER_QUANTILE = "q75"


#metrics
def pinball_loss(y_true, y_pred, alpha):
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def pinball_per_quantile(df: pd.DataFrame) -> dict:
    """Pinball loss of every quantile, plus their mean."""
    y = df["y_next"].to_numpy(dtype=float)
    out = {name: pinball_loss(y, df[name].to_numpy(dtype=float), q) for name, q in QUANTILE_LEVELS.items()}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def service_levels(df: pd.DataFrame) -> dict:
    """Share of actual demand at or below each quantile (q90 should be about 0.90)."""
    y = df["y_next"].to_numpy(dtype=float)
    return {name: float(np.mean(y <= df[name].to_numpy(dtype=float))) for name in QUANTILE_LEVELS}


def intervals(df: pd.DataFrame) -> dict:
    """Coverage and width of the central intervals, and the error of the median."""
    y = df["y_next"].to_numpy(dtype=float)
    return {
        "coverage q25-q75": float(np.mean((y >= df["q25"]) & (y <= df["q75"]))),   # target 0.50
        "coverage q10-q90": float(np.mean((y >= df["q10"]) & (y <= df["q90"]))),   # target 0.80
        "width q25-q75": float(np.mean(df["q75"] - df["q25"])),
        "width q10-q90": float(np.mean(df["q90"] - df["q10"])),
        "MAE of q50": float(np.mean(np.abs(y - df["q50"]))),
    }


def ordering_at_tau(df: pd.DataFrame) -> dict:
    """Order up to the q75 forecast for one week and count what happens.

    cost = Cu * price * units short + Co * price * units left over
    """
    y = df["y_next"].to_numpy(dtype=float)
    stock = df[ORDER_QUANTILE].to_numpy(dtype=float)
    price = df["sell_price"].fillna(df["sell_price"].median()).to_numpy(dtype=float)
    short = np.maximum(y - stock, 0.0)
    left = np.maximum(stock - y, 0.0)
    cost = CU * price * short + CO * price * left
    return {
        "stock-out rate": float(np.mean(short > 0)),             # target about 1 - tau = 0.25
        "units short / week": float(short.mean()),
        "units left over / week": float(left.mean()),
        "cost $ / series-week": float(cost.mean()),
        "total cost $": float(cost.sum()),
    }


#tables
def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    header = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    line = "|" + "---|" * (len(df.columns) + 1)
    def cell(v):
        return "" if pd.isna(v) else f"{v:,.{digits}f}"
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in df.iterrows()]
    return "\n".join([header, line, *rows])


def table(rows: dict, index_name: str, target: dict | None = None, digits: int = 3) -> str:
    df = pd.DataFrame(rows).T
    if target is not None:
        df = pd.concat([pd.DataFrame([target], index=["target"]), df])
    df.index.name = index_name
    return markdown_table(df, digits)


#report
def main():
    feats = load_features()
    truth = feats[feats["split"].isin(["val", "test"])][
        ["series_id", "store_id", "origin_week", "week_idx", "split", "y_next", "sell_price"]]
    truth = truth.merge(load_series_type()[["series_id", "demand_type"]], on="series_id", how="left")

    results = {split: {"pinball": {}, "service": {}, "intervals": {}, "ordering": {}} for split in ["val", "test"]}
    by_type = {}
    for model in MODELS:
        path = PREDICTIONS_DIR / f"{model}.parquet"
        if not path.exists():
            print(f"skipped {model}: {path} not found")
            continue
        df = truth.merge(pd.read_parquet(path), on=["series_id", "origin_week"], how="inner")
        for split, r in results.items():
            part = df[df["split"] == split]
            r["pinball"][model] = pinball_per_quantile(part)
            r["service"][model] = service_levels(part)
            r["intervals"][model] = intervals(part)
            r["ordering"][model] = ordering_at_tau(part)
        test = df[df["split"] == "test"]
        by_type[model] = {t: pinball_per_quantile(g)["mean"] for t, g in test.groupby("demand_type")}

    if not by_type:
        print("No prediction files found. Run python -m src.models.linear and python -m src.models.lgbm first.")
        return

    weeks = {s: truth.loc[truth["split"] == s, "week_idx"] for s in ["val", "test"]}
    lines = [
        "# Member 3: Linear and LightGBM quantile models", "",
        "## Scope", "",
        f"- Stores: {feats['store_id'].nunique()} ({', '.join(sorted(feats['store_id'].unique()))}), "
        f"items: {feats['series_id'].str.split('__').str[1].nunique()}, series: {feats['series_id'].nunique()}",
        "- Granularity: weekly; horizon: 1 week ahead",
        f"- Validation: weeks {weeks['val'].min()}-{weeks['val'].max()} ({len(weeks['val'])} rows); "
        f"test: weeks {weeks['test'].min()}-{weeks['test'].max()} ({len(weeks['test'])} rows, held out)",
        f"- Quantiles: {', '.join(f'{q:.2f}' for q in QUANTILE_LEVELS.values())}",
        f"- Costs: Cu = {CU:.0%} and Co = {CO:.0%} of unit price -> critical ratio tau = {TAU:.2f} "
        f"-> order up to {ORDER_QUANTILE}", "",
        "How to read the tables:",
        "- **Pinball loss**: lower is better (units sold per series-week).",
        "- **Service level**: share of weeks where demand was at or below the forecast; should match "
        "the quantile level (target row).",
        f"- **Ordering at tau**: stock up to {ORDER_QUANTILE} for one week. Stock-out rate should be near "
        f"{1 - TAU:.2f}; lower cost is better. This is a quick check under the README cost assumptions; "
        "the full policy comparison is Member 1's.", "",
    ]

    #val and test tables
    for split, r in results.items():
        lines += [
            f"## {split.capitalize()} split" + (" (model selection)" if split == "val" else " (one-time check)"), "",
            "### Pinball loss per quantile", "", table(r["pinball"], "model"), "",
            "### Service level per quantile", "", table(r["service"], "model", target=dict(QUANTILE_LEVELS)), "",
            "### Intervals and median error", "",
            table(r["intervals"], "model", target={"coverage q25-q75": 0.50, "coverage q10-q90": 0.80,
                                                   "width q25-q75": np.nan, "width q10-q90": np.nan,
                                                   "MAE of q50": np.nan}), "",
            f"### Ordering at tau = {TAU:.2f} ({ORDER_QUANTILE})", "", table(r["ordering"], "model", digits=2), "",
        ]

    df = pd.DataFrame(by_type)
    df.index.name = "demand_type"
    lines += ["## Test mean pinball by demand type", "", markdown_table(df), ""]

    #SHAP (from src/explain.py)
    lines += ["## SHAP: most important features (LightGBM)", ""]
    if SHAP_IMPORTANCE.exists():
        imp = pd.read_csv(SHAP_IMPORTANCE, index_col="feature")
        cols = [c for c in ["q10", "q50", ORDER_QUANTILE, "q90", "q99"] if c in imp.columns]
        top = imp[cols].head(10)
        top.index.name = "feature (mean |SHAP|, units)"
        lines += [markdown_table(top), "",
                  "Plots: `reports/shap/shap_summary_q50.png`, `shap_summary_q90.png`, `shap_importance.png`, "
                  "and the one-item explanation `local_*.png`.", ""]
    else:
        lines += ["Not available yet: run `python -m src.explain` after LightGBM, then run this report again.", ""]

    #final model selection (on val only)
    val_mean = {m: v["mean"] for m, v in results["val"]["pinball"].items()}
    val_q75 = {m: v[ORDER_QUANTILE] for m, v in results["val"]["pinball"].items()}
    best = min(val_mean, key=val_mean.get)
    test_mean = results["test"]["pinball"][best]["mean"]
    test_best = min(results["test"]["pinball"], key=lambda m: results["test"]["pinball"][m]["mean"])
    lines += [
        "## Model selection", "",
        f"- Selected on **val mean pinball**: **{best}** ({val_mean[best]:.3f}).",
        f"- Best on val pinball at the ordering quantile {ORDER_QUANTILE}: "
        f"**{min(val_q75, key=val_q75.get)}** ({min(val_q75.values()):.3f}).",
        f"- Test check: {best} scores {test_mean:.3f}; the best test score is {test_best} "
        + ("(the same model, so the choice holds)." if test_best == best else
           "(a different model: report it, but do not change the choice based on test)."),
        "- Runs and per-fold scores are in MLflow: experiment `m5-inventory-forecasting` "
        "(`mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`).", "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT}  (selected model: {best})")


if __name__ == "__main__":
    main()