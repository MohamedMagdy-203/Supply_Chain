import json

import pandas as pd
import pytest

from scripts.build_dataset import run
from tests.synthetic_m5 import make_raw_m5


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5")
    info = make_raw_m5(root / "raw")
    out, docs = root / "processed", root / "docs"
    run(root / "raw", out, docs, n_items=100, audit_cells=5000)
    return info, out, docs


def test_all_outputs_exist(built):
    _, out, docs = built
    for name in ["weekly.parquet", "features.parquet", "splits.json", "sample_items.csv",
                 "series_type.csv", "audit_summary.json"]:
        assert (out / name).exists(), name
    assert (docs / "data_dictionary.md").exists()


def test_weekly_contract_columns(built):
    _, out, _ = built
    w = pd.read_parquet(out / "weekly.parquet")
    for c in ["series_id", "store_id", "item_id", "dept_id", "cat_id", "week_idx", "y",
              "sell_price", "on_sale", "snap_days", "n_events", "split"]:
        assert c in w.columns
    assert not w.duplicated(["series_id", "week_idx"]).any()


def test_partial_last_week_is_removed(built):
    info, out, _ = built
    w = pd.read_parquet(out / "weekly.parquet")
    assert w["week_idx"].max() == info["n_weeks"] - 1


def test_prelaunch_weeks_are_removed(built):
    info, out, _ = built
    w = pd.read_parquet(out / "weekly.parquet")
    late = w[w["item_id"] == info["late_item"]]
    assert late["week_idx"].min() == info["late_week"]


def test_sample_excludes_non_ca_and_incomplete_items(built):
    info, out, _ = built
    w = pd.read_parquet(out / "weekly.parquet")
    assert set(w["store_id"]) == {"CA_1", "CA_2", "CA_3"}
    assert info["two_store_item"] not in set(w["item_id"])


def test_splits_are_time_ordered_and_sized(built):
    _, out, _ = built
    s = json.loads((out / "splits.json").read_text())["splits_week_idx"]
    assert s["train"][1] + 1 == s["val"][0]
    assert s["val"][1] + 1 == s["test"][0]
    assert s["val"][1] - s["val"][0] + 1 == 12
    assert s["test"][1] - s["test"][0] + 1 == 12


def test_features_contract(built):
    _, out, _ = built
    f = pd.read_parquet(out / "features.parquet")
    assert (f["origin_week"] == f["week_idx"] - 1).all()
    assert not f.duplicated(["series_id", "origin_week"]).any()
    assert set(f["split"]) == {"train", "val", "test"}


def test_pipeline_is_deterministic(built, tmp_path):
    info, out, _ = built
    root = tmp_path
    make_raw_m5(root / "raw")
    run(root / "raw", root / "p2", root / "d2", n_items=100, audit_cells=5000)
    a = pd.read_parquet(out / "features.parquet")
    b = pd.read_parquet(root / "p2" / "features.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_missing_raw_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        run(tmp_path, tmp_path / "o", tmp_path / "d")