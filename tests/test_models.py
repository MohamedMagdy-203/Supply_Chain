import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import QUANTILE_LEVELS
from src.features import build_features
from src.models import common, lgbm, linear

SMALL = lgbm.Candidate(num_leaves=7, learning_rate=0.1, n_estimators=30, min_child_samples=5)


@pytest.fixture()
def splits(weekly, tmp_path, monkeypatch):
    """Synthetic train / val / test, and predictions written to a temporary folder."""
    monkeypatch.setattr(data, "PREDICTIONS_DIR", tmp_path)
    feats = build_features(weekly)
    out = {}
    for name in ["train", "val", "test"]:
        X, y, keys = data.load_xy(name, feats)
        weeks = feats.loc[feats["split"] == name, "week_idx"].to_numpy()
        out[name] = (common.select_features(X), y, keys, weeks)
    return feats, out


def check(predictions, n_rows):
    assert list(predictions) == list(QUANTILE_LEVELS)
    for values in predictions.values():
        assert len(values) == n_rows
        assert np.isfinite(values).all() and (values >= 0).all()


def test_pinball_loss():
    y = np.array([10.0, 10.0])
    # q90: too low by 2 costs 0.9*2, too high by 2 costs 0.1*2
    assert linear.pinball_loss(y, np.array([8.0, 12.0]), 0.9) == pytest.approx(1.0)
    assert lgbm.pinball_loss(y, y, 0.5) == 0.0


def test_model_features():
    assert "on_sale" not in common.MODEL_FEATURES
    assert common.MODEL_FEATURES[-3:] == ["store_id", "dept_id", "cat_id"]


def test_time_folds(splits):
    _, s = splits
    weeks = np.concatenate([s["train"][3], s["val"][3]])
    folds = common.time_folds(s["train"][3], s["val"][3])
    assert len(folds) == 3
    for train_rows, valid_rows in folds:
        assert weeks[train_rows].max() < weeks[valid_rows].min()      # train always before valid
    last_valid = folds[-1][1]
    assert len(last_valid) == len(s["val"][3])                        # last fold = the val split
    assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])     # expanding


def test_ridge(splits):
    _, s = splits
    model, offsets = linear.fit_ridge(s["train"][0], s["train"][1], alpha=10.0)
    predictions = linear.predict_ridge(model, offsets, s["val"][0])
    check(predictions, len(s["val"][0]))
    assert offsets["q10"] <= offsets["q50"] <= offsets["q90"]


def test_quantreg(splits):
    _, s = splits
    preprocessor, params = linear.fit_quantreg(s["train"][0], s["train"][1])
    check(linear.predict_quantreg(preprocessor, params, s["val"][0]), len(s["val"][0]))


def test_lgbm_and_save_predictions(splits):
    feats, s = splits
    models = lgbm.fit_quantile_models(s["train"][0], s["train"][1], SMALL)
    val_p = lgbm.predict_quantiles(models, s["val"][0])
    test_p = lgbm.predict_quantiles(models, s["test"][0])
    check(test_p, len(s["test"][0]))
    frame = linear.to_frame(s["val"][2], val_p, s["test"][2], test_p)
    data.save_predictions(frame, "lgbm_unit", features=feats)     # validates the shared format


def test_shap_adds_up(splits):
    from src import explain
    _, s = splits
    model = lgbm.fit_quantile_models(s["train"][0], s["train"][1], SMALL)["q50"]
    table = explain.shap_table(model.booster_, s["test"][0], s["test"][2])
    features = list(s["test"][0].columns)
    rebuilt = table["base_value"] + table[features].sum(axis=1)
    assert np.allclose(rebuilt, table["prediction"], atol=1e-3)
