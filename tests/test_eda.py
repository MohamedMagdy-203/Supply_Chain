import pandas as pd
import pytest

from scripts.run_eda import run_eda
from src.features import build_features
from tests.conftest import make_weekly


@pytest.fixture(scope="module")
def eda_out(tmp_path_factory):
    root = tmp_path_factory.mktemp("eda")
    proc, rep = root / "processed", root / "reports"
    proc.mkdir()
    weekly = make_weekly(n_series=8, n_weeks=200)
    build_features(weekly).to_parquet(proc / "features.parquet", index=False)
    weekly.to_parquet(proc / "weekly.parquet", index=False)
    pd.DataFrame({"series_id": weekly["series_id"].unique(), "adi": 1.0, "cv2": 0.2,
                  "demand_type": "smooth"}).to_csv(proc / "series_type.csv", index=False)
    run_eda(proc, rep)
    return rep


def test_report_is_written(eda_out):
    text = (eda_out / "eda_summary.md").read_text(encoding="utf-8")
    assert "Suggested decision" in text
    assert "Target distribution" in text and "Mutual information" in text


def test_figures_and_tables_are_written(eda_out):
    assert len(list((eda_out / "figures").glob("*.png"))) >= 10
    assert len(list((eda_out / "eda").glob("*.csv"))) >= 8


def test_eda_uses_train_rows_only(eda_out):
    # the totals table must not contain any validation or test week
    total = pd.read_csv(eda_out / "eda" / "03_total_weekly_demand.csv")
    weekly = make_weekly(n_series=8, n_weeks=200)
    first_val_week = weekly.loc[weekly["split"] == "val", "week_idx"].min()
    assert total["week_idx"].max() < first_val_week