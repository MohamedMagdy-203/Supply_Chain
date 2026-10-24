"""Exploratory data analysis (Pillar 3). Moved from Preprocessing.ipynb, sections 7 and 10.

Rules
  * Everything here uses TRAIN rows only. Looking at val/test and then making
    decisions is a quiet form of leakage.
  * Each function computes one analysis, saves its figure (if given a path) and
    returns the numbers. scripts/run_eda.py calls them all and writes the report.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # no window: figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_regression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from src.config import SEED


def _save(fig, path: Path | None) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 7.1
def target_stats(train: pd.DataFrame, fig_path: Path | None = None):
    """Mean, std, skewness, kurtosis and share of zero weeks, per department."""
    by_dept = (train.groupby("dept_id")["y"]
                    .agg(mean="mean", std="std", median="median", max="max",
                         skew=lambda s: skew(s), kurtosis=lambda s: kurtosis(s),
                         pct_zero=lambda s: (s == 0).mean())
                    .round(3))
    overall = {"zero_share": float((train["y"] == 0).mean()),
               "skew": float(skew(train["y"])), "kurtosis": float(kurtosis(train["y"])),
               "mean": float(train["y"].mean()), "std": float(train["y"].std())}

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].hist(np.clip(train["y"], 0, 60), bins=60)
    ax[0].set_title("Weekly units sold per series (clipped at 60)")
    ax[1].hist(np.log1p(train["y"]), bins=50)
    ax[1].set_title("log(1 + weekly units sold)")
    _save(fig, fig_path)
    return by_dept, overall


# ------------------------------------------------------------------ 7.2
def demand_types(series_type: pd.DataFrame, fig_path: Path | None = None):
    """Counts of smooth / erratic / intermittent / lumpy series (computed by the pipeline)."""
    counts = series_type["demand_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(series_type["adi"], series_type["cv2"], s=8, alpha=0.5)
    ax.axvline(1.32, color="red", ls="--")
    ax.axhline(0.49, color="red", ls="--")
    ax.set(xlabel="ADI", ylabel="CV²", title="Demand types (train period)")
    _save(fig, fig_path)
    return counts.to_frame("n_series")


# ------------------------------------------------------------------ 7.3
def demand_over_time(train: pd.DataFrame, fig_stores: Path | None = None,
                     fig_assortment: Path | None = None):
    """Weekly totals per store and per category, and the number of series on sale."""
    total = (train.groupby("week_idx")
                  .agg(y=("y", "sum"), week_start=("week_start", "first"),
                       n_series=("series_id", "nunique")))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    train.pivot_table(index="week_start", columns="store_id", values="y", aggfunc="sum").plot(
        ax=ax[0], title="Weekly units by store")
    train.pivot_table(index="week_start", columns="cat_id", values="y", aggfunc="sum").plot(
        ax=ax[1], title="Weekly units by category")
    _save(fig, fig_stores)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    total["n_series"].plot(ax=ax, title="Number of series on sale per week (assortment growth)")
    _save(fig, fig_assortment)
    return total


# ------------------------------------------------------------------ 7.4 - 7.6
def seasonality(total: pd.DataFrame, fig_yearly: Path | None = None,
                fig_acf: Path | None = None, fig_stl: Path | None = None) -> dict:
    """Yearly pattern, autocorrelation and STL decomposition of the total weekly demand."""
    out: dict = {"acf_lag52": None, "acf_lag1": None, "stl_seasonal_share": None}

    tot = total.copy()
    tot["year"] = tot["week_start"].dt.year
    tot["woy"] = tot["week_start"].dt.isocalendar().week.astype(int)
    fig, ax = plt.subplots(figsize=(10, 4))
    for year, part in tot.groupby("year"):
        ax.plot(part["woy"], part["y"], label=str(year))
    ax.set(xlabel="Week of the year", ylabel="Total units",
           title="Yearly seasonality (total train demand, one line per year)")
    ax.legend()
    _save(fig, fig_yearly)

    y = total["y"].reset_index(drop=True)
    diff = y.diff().dropna()
    nlags = int(min(60, len(diff) // 2 - 1))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(y, lags=min(60, len(y) // 2 - 1), ax=ax[0], title="ACF: total weekly units")
    plot_acf(diff, lags=nlags, ax=ax[1], title="ACF: weekly change of total units")
    _save(fig, fig_acf)
    values = acf(diff, nlags=nlags)
    out["acf_lag1"] = float(values[1])
    if nlags >= 52:
        out["acf_lag52"] = float(values[52])
    out["acf_confidence_band"] = float(1.96 / np.sqrt(len(diff)))

    if len(y) >= 2 * 52:                                   # STL needs two full cycles
        res = STL(y, period=52, robust=True).fit()
        fig = res.plot()
        fig.set_size_inches(10, 7)
        _save(fig, fig_stl)
        out["stl_seasonal_share"] = float(res.seasonal.var() / y.var())
        out["stl_residual_share"] = float(res.resid.var() / y.var())
    return out


# ------------------------------------------------------------------ 7.7
def price_effect(train: pd.DataFrame, fig_path: Path | None = None):
    """Change in log sales against the price change of the same week (descriptive only)."""
    d = train.sort_values(["series_id", "week_idx"]).copy()
    d["price_change"] = d["sell_price"] / d.groupby("series_id")["sell_price"].shift(1) - 1
    d["log_sales_change"] = np.log1p(d["y"]) - np.log1p(d.groupby("series_id")["y"].shift(1))
    d["price_bin"] = pd.cut(d["price_change"], bins=[-1, -0.2, -0.05, -0.001, 0.001, 0.05, 0.2, 10])

    table = d.groupby("price_bin", observed=True)["log_sales_change"].agg(["mean", "count"]).round(4)
    summary = {
        "mean_change_when_price_cut": float(d.loc[d["price_change"] < -0.05, "log_sales_change"].mean()),
        "mean_change_when_price_rise": float(d.loc[d["price_change"] > 0.05, "log_sales_change"].mean()),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    table["mean"].plot.bar(ax=ax, title="Average change in log sales by price change bucket")
    ax.set_ylabel("mean change in log(1+sales)")
    _save(fig, fig_path)
    return table, summary


# ------------------------------------------------------------------ 7.8
def snap_and_events(train: pd.DataFrame, fig_path: Path | None = None):
    """Average weekly units by number of SNAP days and by presence of an event."""
    snap = train.groupby(["snap_days", "cat_id"])["y"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    snap.plot(ax=ax, marker="o", title="Average weekly units by number of SNAP days in the week")
    ax.set_ylabel("mean units")
    _save(fig, fig_path)

    events = (train.assign(has_event=train["n_events"] > 0)
                   .groupby(["cat_id", "has_event"])["y"].mean().unstack().round(2))
    events.columns = [("event week" if c else "no event week") for c in events.columns]

    summary = {
        "snap_spearman_with_log_sales": float(train["snap_days"].corr(np.log1p(train["y"]), method="spearman")),
        "event_ratio": float(train.loc[train["n_events"] > 0, "y"].mean()
                             / train.loc[train["n_events"] == 0, "y"].mean()),
    }
    return snap.round(2), events, summary


# ------------------------------------------------------------------ 7.9
def outliers(train: pd.DataFrame):
    """Flag weeks above Q3 + 3*IQR of each series' NON-zero weeks. Flagged, not removed."""
    nz = train[train["y"] > 0]
    q = nz.groupby("series_id")["y"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    q["upper_fence"] = q["q3"] + 3 * (q["q3"] - q["q1"])

    t = train.merge(q["upper_fence"], on="series_id", how="left")
    t["is_outlier"] = t["y"] > t["upper_fence"]
    by_cat = t.groupby("cat_id")["is_outlier"].mean().round(4).to_frame("outlier_share")
    top = (t.loc[t["is_outlier"]].nlargest(5, "y")[["series_id", "week_start", "y", "upper_fence"]]
             .reset_index(drop=True))
    return {"outlier_share": float(t["is_outlier"].mean())}, by_cat, top


# ------------------------------------------------------------------ 10.x feature-level
def numeric_features(train_feat: pd.DataFrame, features: list[str]) -> list[str]:
    """Features with more than one value (constant columns carry no information)."""
    return [c for c in features if train_feat[c].nunique() > 1]


def correlations(train_feat: pd.DataFrame, num_features: list[str], fig_path: Path | None = None):
    """Pearson and Spearman correlation of each feature with the target y_next."""
    cols = num_features + ["y_next"]
    corr = pd.DataFrame({
        "pearson": train_feat[cols].corr(method="pearson")["y_next"],
        "spearman": train_feat[cols].corr(method="spearman")["y_next"],
    }).drop(index="y_next").sort_values("spearman")

    fig, ax = plt.subplots(figsize=(8, 8))
    corr.plot.barh(ax=ax, title="Correlation of each feature with the target (train)")
    _save(fig, fig_path)
    return corr.sort_values("spearman", ascending=False).round(3)


def vif_table(train_feat: pd.DataFrame, num_features: list[str], seed: int = SEED) -> pd.Series:
    """Variance inflation factor on a random sample of complete rows.

    lag_52 and the 0/1 flags are skipped. Huge values (1e15, inf) are expected for
    roll_mean_4 (it is exactly the mean of lag_1..lag_4) and are not a bug.
    """
    cols = [c for c in num_features if c not in ["lag_52", "has_lag52", "on_sale"]]
    sample = (train_feat[cols].replace([np.inf, -np.inf], np.nan).dropna()
              .sample(n=min(5000, len(train_feat)), random_state=seed))
    cols = [c for c in cols if sample[c].nunique() > 1]
    X = add_constant(sample[cols], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.Series(values, index=cols).sort_values(ascending=False)


def mutual_information(train_feat: pd.DataFrame, num_features: list[str],
                       fig_path: Path | None = None, seed: int = SEED) -> pd.Series:
    """Mutual information with the target (captures non-linear relationships)."""
    s = train_feat.sample(n=min(5000, len(train_feat)), random_state=seed)
    X = s[num_features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    mi = pd.Series(mutual_info_regression(X, s["y_next"], random_state=seed),
                   index=num_features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    mi.plot.barh(ax=ax, title="Mutual information with the target (train sample)")
    _save(fig, fig_path)
    return mi.sort_values(ascending=False)