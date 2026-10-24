import numpy as np
import pytest

from src import data
from src.config import QUANTILE_COLS
from src.features import CAT_FEATURES, FEATURES, build_features


@pytest.fixture()
def feats(weekly, tmp_path, monkeypatch):
    f = build_features(weekly)
    monkeypatch.setattr(data, "PREDICTIONS_DIR", tmp_path)
    return f


def _preds(feats, split="test"):
    _, _, keys = data.load_xy(split, feats)
    out = keys.copy()
    base = feats.loc[feats.split == split, "lag_1"].fillna(3).to_numpy()
    for i, c in enumerate(QUANTILE_COLS):
        out[c] = base * (0.5 + 0.2 * i)
    return out


def test_load_xy_shapes_and_categoricals(feats):
    X, y, keys = data.load_xy("train", feats)
    assert list(X.columns) == FEATURES + CAT_FEATURES
    assert len(X) == len(y) == len(keys)
    assert str(X["store_id"].dtype) == "category"
    Xt, _, _ = data.load_xy("test", feats)
    assert list(Xt["store_id"].cat.categories) == list(X["store_id"].cat.categories)


def test_save_predictions_ok(feats):
    path = data.save_predictions(_preds(feats), "unit", features=feats)
    assert path.exists()


def test_save_predictions_fixes_crossing(feats):
    p = _preds(feats)
    p.loc[p.index[0], ["q10", "q99"]] = [50.0, 1.0]
    data.save_predictions(p, "cross", features=feats)
    with pytest.raises(ValueError, match="crossing"):
        data.save_predictions(p, "cross2", features=feats, fix_crossing=False)


def test_save_predictions_rejects_missing_test_rows(feats):
    with pytest.raises(ValueError, match="missing"):
        data.save_predictions(_preds(feats).iloc[:-3], "short", features=feats)


def test_save_predictions_rejects_unknown_keys(feats):
    p = _preds(feats)
    p.loc[p.index[0], "series_id"] = "NOPE"
    with pytest.raises(ValueError, match="not val/test"):
        data.save_predictions(p, "bad", features=feats)


def test_save_predictions_rejects_nan_and_bad_name(feats):
    p = _preds(feats)
    p.loc[p.index[0], "q50"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        data.save_predictions(p, "nan", features=feats)
    with pytest.raises(ValueError, match="model_name"):
        data.save_predictions(_preds(feats), "../x", features=feats)