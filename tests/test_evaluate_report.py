import numpy as np
import pandas as pd
import pytest

from src import data, evaluate_report as er
from src.config import QUANTILE_COLS
from src.evaluate import mean_pinball_loss
from src.features import build_features
from src.models import baseline
from tests.conftest import make_weekly


@pytest.fixture(scope="module")
def feats():
    return build_features(make_weekly(n_series=8, n_weeks=140))


@pytest.fixture(scope="module")
def series_type(feats):
    return pd.DataFrame({"series_id": feats["series_id"].unique(), "adi": 1.0, "cv2": 0.2,
                         "demand_type": ["smooth", "intermittent"] * 4})


@pytest.fixture()
def pred_dir(feats, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PREDICTIONS_DIR", tmp_path)
    baseline.run(feats)                                     # writes naive and ma4
    wide = pd.read_parquet(tmp_path / "ma4.parquet")
    wide[QUANTILE_COLS] = wide[QUANTILE_COLS] * 1.5         # a fake "model" to compare
    wide.to_parquet(tmp_path / "wide.parquet", index=False)
    return tmp_path


def test_metrics_known_values():
    y = np.array([0.0, 10.0])
    Q = np.full((2, 7), 5.0)
    m = er.compute_metrics(y, Q)
    assert m["mae"] == 5.0 and m["wape"] == 1.0 and m["cov80"] == 0.0
    assert m["sl_q50"] == 0.5 and m["mape_pos"] == 0.5


def test_pinball_matches_evaluate_py(feats, pred_dir):
    test = feats[feats["split"] == "test"]
    p = er.load_predictions(pred_dir / "ma4.parquet").merge(test[["series_id", "origin_week", "y_next"]],
                                                            on=["series_id", "origin_week"])
    ours = er.compute_metrics(p["y_next"].to_numpy(float), p[QUANTILE_COLS].to_numpy(float))["mean_pinball"]
    assert ours == pytest.approx(mean_pinball_loss(p["y_next"].to_numpy(), p), rel=1e-12)


def test_report_and_files_written(feats, series_type, pred_dir, tmp_path):
    out = tmp_path / "reports" / "model_comparison.md"
    er.run(feats, series_type, pred_dir, out)
    text = out.read_text(encoding="utf-8")
    for needle in ["Validation winner", "naive", "ma4", "wide", "Demand type", "pinball vs"]:
        assert needle in text
    assert (out.parent / "model_metrics.csv").exists()
    assert (out.parent / "model_breakdown_test.csv").exists()


def test_selection_uses_validation_and_excludes_baselines(feats, series_type, pred_dir, tmp_path):
    out = tmp_path / "r" / "model_comparison.md"
    er.run(feats, series_type, pred_dir, out)
    line = [l for l in out.read_text(encoding="utf-8").splitlines() if "Validation winner" in l][0]
    assert "wide" in line and "naive" not in line and "ma4" not in line


def test_refuses_to_overwrite_foreign_report(feats, series_type, pred_dir, tmp_path):
    out = tmp_path / "model_comparison.md"
    out.write_text("# Member 3: Linear and LightGBM quantile models", encoding="utf-8")
    with pytest.raises(FileExistsError):
        er.run(feats, series_type, pred_dir, out)
    er.run(feats, series_type, pred_dir, out, force=True)
    assert er.MARKER in out.read_text(encoding="utf-8")


def test_incomplete_model_is_rejected(feats, series_type, pred_dir, tmp_path):
    pd.read_parquet(pred_dir / "ma4.parquet").iloc[:-3].to_parquet(pred_dir / "short.parquet", index=False)
    with pytest.raises(ValueError, match="short"):
        er.run(feats, series_type, pred_dir, tmp_path / "x.md", models=["ma4", "short"])


def test_fake_predictions_are_ignored(pred_dir):
    pd.read_parquet(pred_dir / "ma4.parquet").to_parquet(pred_dir / "fake.parquet", index=False)
    assert "fake" not in er.discover_models(pred_dir)