import numpy as np
import pytest

from src import data
from src.config import QUANTILE_COLS
from src.features import build_features
from src.models import baseline
from tests.conftest import make_weekly


@pytest.fixture(scope="module")
def feats():
    return build_features(make_weekly(n_series=8, n_weeks=140))


@pytest.fixture()
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PREDICTIONS_DIR", tmp_path)
    return tmp_path


def test_run_writes_valid_files(feats, out_dir):
    baseline.run(feats)
    for name in baseline.BASELINES:
        assert (out_dir / f"{name}.parquet").exists()


def test_naive_uses_last_year_then_last_week(feats):
    X, _, _ = data.load_xy("test", feats)
    f = baseline.point_forecast(X, "naive")
    has52 = X["lag_52"].notna().to_numpy()
    assert has52.any() and (~has52).any() or has52.all()
    assert np.allclose(f[has52], X["lag_52"].to_numpy()[has52])
    if (~has52).any():
        assert np.allclose(f[~has52], X["lag_1"].to_numpy()[~has52])


def test_ma4_is_roll_mean_4(feats):
    X, _, _ = data.load_xy("test", feats)
    assert np.allclose(baseline.point_forecast(X, "ma4"), X["roll_mean_4"].to_numpy())


def test_quantiles_are_ordered_and_non_negative(feats, out_dir):
    baseline.run(feats)
    import pandas as pd
    p = pd.read_parquet(out_dir / "naive.parquet")
    arr = p[QUANTILE_COLS].to_numpy()
    assert (np.diff(arr, axis=1) >= 0).all() and (arr >= 0).all()


def test_val_and_test_targets_do_not_change_predictions(feats, out_dir):
    """Quantiles must come from TRAIN errors only: scramble y_next in val/test."""
    import pandas as pd
    baseline.run(feats)
    before = pd.read_parquet(out_dir / "ma4.parquet")

    changed = feats.copy()
    mask = changed["split"] != "train"
    changed.loc[mask, "y_next"] = changed.loc[mask, "y_next"] * 5 + 7
    baseline.run(changed)
    after = pd.read_parquet(out_dir / "ma4.parquet")
    pd.testing.assert_frame_equal(before, after)


def test_short_series_fall_back_to_department(feats):
    X, y, k = data.load_xy("train", feats)
    tables = baseline.fit_error_quantiles(k, X, y, baseline.point_forecast(X, "ma4"), min_obs=10_000)
    assert len(tables[0]) == 0                       # no series qualifies
    Xt, _, kt = data.load_xy("test", feats)
    out = baseline.predict_quantiles(kt, Xt, baseline.point_forecast(Xt, "ma4"), tables)
    assert not out[QUANTILE_COLS].isna().any().any()