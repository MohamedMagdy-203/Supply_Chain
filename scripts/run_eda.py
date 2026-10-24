"""Run the whole EDA and write figures, tables and a report.

    python -m scripts.run_eda
    python -m scripts.run_eda --reports-dir reports

Reads  data/processed/{weekly,features}.parquet and series_type.csv (train rows only)
Writes reports/figures/*.png        one figure per analysis
       reports/eda/*.csv            the tables behind the figures
       reports/eda_summary.md       numbers + a SUGGESTED decision for each finding

The suggested decisions are rule-based. Member 2 must read them, edit
reports/eda_summary.md if a rule does not fit, and own the final wording.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import eda
from src.config import PROCESSED_DIR
from src.features import FEATURES


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return format(float(v), ".4g")
    return str(v)


def _md_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    d = df.head(max_rows).reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def run_eda(processed_dir: Path = PROCESSED_DIR, reports_dir: Path = Path("reports")) -> Path:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    fig_dir, tab_dir = reports_dir / "figures", reports_dir / "eda"
    tab_dir.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_parquet(processed_dir / "weekly.parquet")
    features = pd.read_parquet(processed_dir / "features.parquet")
    series_type = pd.read_csv(processed_dir / "series_type.csv")
    train = weekly[weekly["split"] == "train"].copy()
    train_feat = features[features["split"] == "train"].copy()
    print(f"EDA on train rows only: weekly {len(train):,} | features {len(train_feat):,}")

    findings = []

    def add(title, figure, table, evidence, decision):
        findings.append({"title": title, "figure": figure, "table": table,
                         "evidence": evidence, "decision": decision})

    # 7.1 target distribution
    print("  7.1 target distribution")
    by_dept, ov = eda.target_stats(train, fig_dir / "01_target_distribution.png")
    by_dept.to_csv(tab_dir / "01_target_stats_by_dept.csv")
    add("Target distribution", "01_target_distribution.png", by_dept,
        f"{ov['zero_share']:.1%} of weeks are zero, skewness {ov['skew']:.2f}, kurtosis {ov['kurtosis']:.1f}.",
        "Use quantile (pinball) loss. Report WAPE, and compute MAPE only on weeks with y > 0. "
        "Keep the long right tail (the policy needs q95 and q99)."
        if ov["skew"] > 1 or ov["zero_share"] > 0.1 else
        "Distribution is close to symmetric with few zeros: MAPE is usable, keep WAPE as the main metric.")

    # 7.2 demand types
    print("  7.2 demand types")
    counts = eda.demand_types(series_type, fig_dir / "02_demand_types.png")
    counts.to_csv(tab_dir / "02_demand_types.csv")
    hard = int(counts.loc[counts.index.isin(["intermittent", "lumpy"]), "n_series"].sum())
    add("Demand types (ADI / CV2)", "02_demand_types.png", counts,
        f"{hard} of {len(series_type)} series ({hard / len(series_type):.0%}) are intermittent or lumpy.",
        "The evaluation harness reports intermittent and lumpy series separately (use series_type.csv).")

    # 7.3 over time
    print("  7.3 demand over time")
    total = eda.demand_over_time(train, fig_dir / "03_demand_by_store_category.png",
                                 fig_dir / "03_assortment_growth.png")
    total.to_csv(tab_dir / "03_total_weekly_demand.csv")
    add("Demand over time", "03_demand_by_store_category.png", total[["y", "n_series"]].describe().T,
        f"Series on sale go from {int(total['n_series'].iloc[0])} to {int(total['n_series'].iloc[-1])}"
        + (": part of the rise in totals is new items, not growth." if total["n_series"].iloc[-1] > total["n_series"].iloc[0]
           else "."),
        "store_id, dept_id and cat_id are features (levels differ). Read totals with the assortment growth in mind.")

    # 7.4 - 7.6 seasonality
    print("  7.4-7.6 seasonality, ACF, STL")
    se = eda.seasonality(total, fig_dir / "04_yearly_seasonality.png",
                         fig_dir / "05_acf.png", fig_dir / "06_stl.png")
    band = se["acf_confidence_band"]
    lag52 = se["acf_lag52"]
    seasonal_ok = lag52 is not None and abs(lag52) > band
    evidence = f"ACF of the weekly change: lag 1 = {se['acf_lag1']:.2f}"
    evidence += (f", lag 52 = {lag52:.2f} (band +/-{band:.2f})." if lag52 is not None
                 else " (not enough history to read lag 52).")
    if se["stl_seasonal_share"] is not None:
        evidence += f" STL: seasonal part explains {se['stl_seasonal_share']:.0%} of the variance."
    add("Seasonality, autocorrelation and STL", "04_yearly_seasonality.png",
        pd.DataFrame({k: [v] for k, v in se.items() if v is not None}).T.rename(columns={0: "value"}),
        evidence,
        "Keep lag_52 and week_of_year (yearly pattern outside the confidence band)." if seasonal_ok else
        "Yearly pattern is weak in the totals. Keep lag_52 and week_of_year anyway (cheap) and let the models decide.")

    # 7.7 price
    print("  7.7 price effect")
    pe_table, pe = eda.price_effect(train, fig_dir / "07_price_effect.png")
    pe_table.to_csv(tab_dir / "07_price_effect.csv")
    cut, rise = pe["mean_change_when_price_cut"], pe["mean_change_when_price_rise"]
    add("Price effect", "07_price_effect.png", pe_table,
        f"Mean change in log sales: {cut:+.3f} after a price cut (> 5%), {rise:+.3f} after a price rise.",
        "Keep sell_price and price_change_1w as known-future features (sales move against price)."
        if cut > rise else "Price effect is not visible in this view. Keep the price features, check SHAP later.")

    # 7.8 SNAP and events
    print("  7.8 SNAP and events")
    snap, events, sn = eda.snap_and_events(train, fig_dir / "08_snap_effect.png")
    snap.to_csv(tab_dir / "08_snap_effect.csv")
    events.to_csv(tab_dir / "08_event_effect.csv")
    add("SNAP days and events", "08_snap_effect.png", events,
        f"Spearman(snap_days, log sales) = {sn['snap_spearman_with_log_sales']:+.3f}; "
        f"event weeks sell {sn['event_ratio']:.2f}x the non-event average.",
        "Keep snap_days and n_events as known-future covariates (assumption: the calendar is known in advance).")

    # 7.9 outliers
    print("  7.9 outliers")
    out_sum, out_cat, out_top = eda.outliers(train)
    out_cat.to_csv(tab_dir / "09_outliers_by_category.csv")
    out_top.to_csv(tab_dir / "09_top_outliers.csv", index=False)
    add("Outliers", None, out_cat,
        f"{out_sum['outlier_share']:.2%} of weeks are above Q3 + 3 IQR of their series.",
        "Flag but KEEP outliers: spikes are real demand and the inventory policy depends on q95 and q99.")

    # 10.x feature level
    print("  10.1 correlations")
    num = eda.numeric_features(train_feat, FEATURES)
    corr = eda.correlations(train_feat, num, fig_dir / "10_correlations.png")
    corr.to_csv(tab_dir / "10_correlations.csv")
    top3 = ", ".join(corr.index[:3])
    add("Feature correlations with the target", "10_correlations.png", corr,
        f"Strongest by Spearman: {top3}.",
        "Lags and rolling means are the core features. Weak features stay for tree models and are checked with SHAP.")

    print("  10.2 VIF")
    vif = eda.vif_table(train_feat, num)
    vif.to_frame("vif").to_csv(tab_dir / "10_vif.csv")
    high = [c for c, v in vif.items() if v > 10]
    add("Collinearity (VIF)", None, vif.to_frame("vif"),
        f"{len(high)} of {len(vif)} features have VIF > 10: {', '.join(high[:8])}{' ...' if len(high) > 8 else ''}.",
        "LightGBM is unaffected. For the linear model use Ridge or regularized quantile regression, and do not feed "
        "roll_mean_4 together with lag_1..lag_4 (perfect linear combination).")

    print("  10.3 mutual information")
    mi = eda.mutual_information(train_feat, num, fig_dir / "10_mutual_information.png")
    mi.to_frame("mutual_information").to_csv(tab_dir / "10_mutual_information.csv")
    add("Mutual information", "10_mutual_information.png", mi.head(10).to_frame("mutual_information"),
        f"Most informative: {', '.join(mi.index[:5])}.",
        "Confirms which features carry non-linear signal. Features near zero are candidates to drop for the deep model.")

    path = _write_report(reports_dir, findings)
    print("Done. Report:", path)
    return path


def _write_report(reports_dir: Path, findings: list[dict]) -> Path:
    lines = ["# EDA summary (train period only)", "",
             "Generated by `python -m scripts.run_eda`. Decisions marked *suggested* are rule-based:",
             "the owner of the EDA reviews each one and edits this file if a rule does not fit.", ""]
    for i, f in enumerate(findings, 1):
        lines += [f"## {i}. {f['title']}", ""]
        if f["figure"]:
            lines += [f"![{f['title']}](figures/{f['figure']})", ""]
        lines += [_md_table(f["table"]), "", f"**Evidence:** {f['evidence']}", "",
                  f"**Suggested decision:** {f['decision']}", ""]
    path = reports_dir / "eda_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    a = p.parse_args(argv)
    run_eda(a.processed_dir, a.reports_dir)


if __name__ == "__main__":
    main()