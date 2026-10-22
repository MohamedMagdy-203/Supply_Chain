from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
SID = "CA_1_ITEM_0"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_series_lists_ids_and_weeks():
    r = client.get("/series")
    assert r.status_code == 200
    assert SID in r.json()["series_ids"]


def test_forecast_ok_and_monotonic():
    r = client.get("/forecast", params={"series_id": SID, "origin_week": 1})
    assert r.status_code == 200
    b = r.json()
    qs = [b[k] for k in ("q10", "q25", "q50", "q75", "q90", "q95", "q99")]
    assert qs == sorted(qs)


def test_forecast_unknown_series_404():
    r = client.get("/forecast", params={"series_id": "NOPE", "origin_week": 1})
    assert r.status_code == 404


def test_forecast_unknown_model_404():
    r = client.get("/forecast", params={"series_id": SID, "origin_week": 1, "model": "nope"})
    assert r.status_code == 404


def test_forecast_bad_model_name_422():
    r = client.get("/forecast", params={"series_id": SID, "origin_week": 1, "model": "../x"})
    assert r.status_code == 422


def test_inventory_default_tau_is_075():
    r = client.get("/inventory", params={"series_id": SID, "origin_week": 1})
    assert r.status_code == 200
    assert r.json()["tau"] == 0.75


def test_inventory_on_hand_reduces_order():
    p = {"series_id": SID, "origin_week": 1}
    a = client.get("/inventory", params={**p, "on_hand": 0}).json()
    b = client.get("/inventory", params={**p, "on_hand": 5}).json()
    assert b["order_qty"] <= a["order_qty"]
    assert a["order_up_to"] == b["order_up_to"]


def test_inventory_higher_service_level_higher_S():
    p = {"series_id": SID, "origin_week": 1}
    lo = client.get("/inventory", params={**p, "service_level": 0.5}).json()
    hi = client.get("/inventory", params={**p, "service_level": 0.95}).json()
    assert hi["order_up_to"] > lo["order_up_to"]


def test_inventory_service_level_out_of_range_422():
    for bad in (0, 1, 1.5, -0.2):
        r = client.get("/inventory", params={"series_id": SID, "origin_week": 1, "service_level": bad})
        assert r.status_code == 422


def test_inventory_negative_cost_422():
    r = client.get("/inventory", params={"series_id": SID, "origin_week": 1, "cu": -1})
    assert r.status_code == 422