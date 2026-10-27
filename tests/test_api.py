"""API integration tests.

Uses real data files (lgbm.parquet, weekly.parquet, model_metrics.csv).
The TestClient runs the full FastAPI app in-process; no server needed.

Real series / week constants taken from lgbm.parquet:
    SERIES_ID   = "CA_1__FOODS_1_004"   (first series alphabetically)
    ORIGIN_WEEK = 252                   (first / minimum origin week)
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Constants – must match data in predictions/lgbm.parquet
# ---------------------------------------------------------------------------
SERIES_ID = "CA_1__FOODS_1_004"
ORIGIN_WEEK = 252          # minimum origin week present in lgbm.parquet
LAST_WEEK = 275            # maximum origin week present in lgbm.parquet


# ===========================================================================
# /health
# ===========================================================================
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "lgbm" in body["available_models"]
    assert body["default_model"] == "lgbm"


# ===========================================================================
# /series
# ===========================================================================
def test_series_lists_real_ids():
    r = client.get("/series")
    assert r.status_code == 200
    body = r.json()
    assert SERIES_ID in body["series_ids"]
    assert ORIGIN_WEEK in body["origin_weeks"]
    assert LAST_WEEK in body["origin_weeks"]


def test_series_explicit_model():
    r = client.get("/series", params={"model": "lgbm"})
    assert r.status_code == 200
    assert r.json()["model"] == "lgbm"


def test_series_unknown_model_404():
    r = client.get("/series", params={"model": "nope"})
    assert r.status_code == 404


def test_series_bad_model_name_422():
    r = client.get("/series", params={"model": "../x"})
    assert r.status_code == 422


# ===========================================================================
# /forecast
# ===========================================================================
def test_forecast_ok_and_monotonic():
    r = client.get("/forecast", params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK})
    assert r.status_code == 200
    body = r.json()
    qs = [body[k] for k in ("q10", "q25", "q50", "q75", "q90", "q95", "q99")]
    assert qs == sorted(qs), "Quantiles must be non-decreasing"


def test_forecast_unknown_series_404():
    r = client.get("/forecast", params={"series_id": "NO_SUCH_SERIES", "origin_week": ORIGIN_WEEK})
    assert r.status_code == 404


def test_forecast_unknown_week_404():
    r = client.get("/forecast", params={"series_id": SERIES_ID, "origin_week": 9999})
    assert r.status_code == 404


def test_forecast_unknown_model_404():
    r = client.get("/forecast", params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK, "model": "nope"})
    assert r.status_code == 404


def test_forecast_bad_model_name_422():
    r = client.get("/forecast", params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK, "model": "../x"})
    assert r.status_code == 422


# ===========================================================================
# /inventory
# ===========================================================================
def test_inventory_default_tau_is_075():
    """cu=0.30, co=0.10 → τ = 0.30 / 0.40 = 0.75"""
    r = client.get("/inventory", params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK})
    assert r.status_code == 200
    assert r.json()["tau"] == pytest.approx(0.75, abs=1e-4)


def test_inventory_on_hand_reduces_order():
    p = {"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK}
    a = client.get("/inventory", params={**p, "on_hand": 0}).json()
    b = client.get("/inventory", params={**p, "on_hand": 5}).json()
    assert b["order_qty"] <= a["order_qty"]
    assert a["order_up_to"] == pytest.approx(b["order_up_to"], abs=1e-6)


def test_inventory_higher_service_level_higher_S():
    p = {"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK}
    lo = client.get("/inventory", params={**p, "service_level": 0.25}).json()
    hi = client.get("/inventory", params={**p, "service_level": 0.95}).json()
    assert hi["order_up_to"] >= lo["order_up_to"]


def test_inventory_service_level_out_of_range_422():
    for bad in (0, 1, 1.5, -0.2):
        r = client.get(
            "/inventory",
            params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK, "service_level": bad},
        )
        assert r.status_code == 422, f"Expected 422 for service_level={bad}"


def test_inventory_negative_cost_422():
    r = client.get("/inventory", params={"series_id": SERIES_ID, "origin_week": ORIGIN_WEEK, "cu": -1})
    assert r.status_code == 422


# ===========================================================================
# /history
# ===========================================================================
def test_history_returns_points():
    r = client.get("/history", params={"series_id": SERIES_ID})
    assert r.status_code == 200
    body = r.json()
    assert body["series_id"] == SERIES_ID
    assert body["n_weeks"] > 0
    pts = body["history"]
    assert len(pts) == body["n_weeks"]
    # Check shape of first point
    first = pts[0]
    assert "week_idx" in first
    assert "y" in first
    assert "split" in first


def test_history_up_to_week_filter():
    r = client.get("/history", params={"series_id": SERIES_ID, "up_to_week": 260})
    assert r.status_code == 200
    body = r.json()
    assert all(p["week_idx"] <= 260 for p in body["history"])


def test_history_n_weeks_limit():
    r = client.get("/history", params={"series_id": SERIES_ID, "n_weeks": 10})
    assert r.status_code == 200
    assert r.json()["n_weeks"] == 10


def test_history_unknown_series_404():
    r = client.get("/history", params={"series_id": "NO_SUCH_SERIES"})
    assert r.status_code == 404


# ===========================================================================
# /models
# ===========================================================================
def test_models_returns_lgbm():
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    model_names = [m["model"] for m in body["metrics"]]
    assert "lgbm" in model_names


def test_models_split_filter_val():
    r = client.get("/models", params={"split": "val"})
    assert r.status_code == 200
    body = r.json()
    assert all(m["split"] == "val" for m in body["metrics"])


def test_models_split_filter_test():
    r = client.get("/models", params={"split": "test"})
    assert r.status_code == 200
    body = r.json()
    assert all(m["split"] == "test" for m in body["metrics"])


def test_models_bad_split_422():
    r = client.get("/models", params={"split": "train"})
    assert r.status_code == 422


def test_models_metrics_have_required_fields():
    r = client.get("/models")
    assert r.status_code == 200
    for m in r.json()["metrics"]:
        for field in ("model", "split", "mae", "rmse", "r2", "wape", "mean_pinball"):
            assert field in m, f"Missing field '{field}' in metric row"


# ===========================================================================
# /orders
# ===========================================================================
def test_orders_returns_all_series():
    r = client.get("/orders", params={"origin_week": ORIGIN_WEEK})
    assert r.status_code == 200
    body = r.json()
    assert body["origin_week"] == ORIGIN_WEEK
    assert body["n_series"] > 0
    assert len(body["orders"]) == body["n_series"]


def test_orders_tau_from_costs():
    """Default cu=0.30, co=0.10 → τ = 0.75"""
    r = client.get("/orders", params={"origin_week": ORIGIN_WEEK})
    body = r.json()
    assert body["tau"] == pytest.approx(0.75, abs=1e-4)


def test_orders_explicit_service_level():
    r = client.get("/orders", params={"origin_week": ORIGIN_WEEK, "service_level": 0.95})
    assert r.status_code == 200
    assert r.json()["tau"] == pytest.approx(0.95, abs=1e-4)


def test_orders_sorted_by_qty_desc():
    r = client.get("/orders", params={"origin_week": ORIGIN_WEEK})
    qtys = [o["order_qty"] for o in r.json()["orders"]]
    assert qtys == sorted(qtys, reverse=True)


def test_orders_all_series_present():
    """Every series in the forecast file should appear in /orders."""
    from app.api.deps import get_predictions
    df = get_predictions("lgbm")
    expected = set(df[df["origin_week"] == ORIGIN_WEEK]["series_id"].unique())
    r = client.get("/orders", params={"origin_week": ORIGIN_WEEK})
    returned = {o["series_id"] for o in r.json()["orders"]}
    assert expected == returned


def test_orders_last_week():
    r = client.get("/orders", params={"origin_week": LAST_WEEK})
    assert r.status_code == 200
    assert r.json()["n_series"] > 0


def test_orders_out_of_range_week_422():
    """origin_week < 252 should fail validation."""
    r = client.get("/orders", params={"origin_week": 100})
    assert r.status_code == 422


def test_orders_on_hand_reduces_qty():
    """Providing on_hand > 0 should reduce or equal order_qty vs on_hand=0."""
    r0 = client.get("/orders", params={"origin_week": ORIGIN_WEEK, "on_hand": 0})
    r5 = client.get("/orders", params={"origin_week": ORIGIN_WEEK, "on_hand": 5})
    # Match by series_id
    qty0 = {o["series_id"]: o["order_qty"] for o in r0.json()["orders"]}
    qty5 = {o["series_id"]: o["order_qty"] for o in r5.json()["orders"]}
    for sid in qty0:
        assert qty5[sid] <= qty0[sid], f"{sid}: on_hand=5 gave higher qty than on_hand=0"
