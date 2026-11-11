import pytest
from api.app import create_app
from fastapi.testclient import TestClient

OFFERS = [
    {"offer_id": f"o{i}", "supplier": s, "car_class": c, "supplier_rating": r, "cost_per_day": cost}
    for i, (s, c, r, cost) in enumerate([
        ("Alpha", "economy", 4.5, 30.0), ("Bravo", "compact", 4.2, 36.0),
        ("Cobalt", "suv", 3.8, 55.0), ("Delta", "mini", 4.0, 24.0),
        ("Ember", "van", 3.5, 62.0), ("Fjord", "economy", 4.6, 33.0)])
]
SESSION = {"partner": "Ryanair", "market": "IE", "device": "mobile", "returning_customer": False,
           "lead_days": 21, "trip_days": 5, "party_size": 3}


@pytest.fixture()
def client(small_model):
    _, path = small_model
    with TestClient(create_app(model_path=str(path), lam=10.0)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model_version"] == "test"


def test_rank_ml_returns_valid_plan(client):
    r = client.post("/rank", json={"session": SESSION, "offers": OFFERS})
    assert r.status_code == 200
    body = r.json()
    assert [o["position"] for o in body["offers"]] == list(range(1, 7))
    assert {o["offer_id"] for o in body["offers"]} == {o["offer_id"] for o in OFFERS}
    assert all(0 < o["p_book"] < 1 for o in body["offers"])
    assert all(0.08 <= o["markup"] <= 0.32 for o in body["offers"])  # Ryanair cap
    assert body["guardrails"]["partner"] == "Ryanair" and body["guardrails"]["markup_max"] == 0.32
    assert body["latency_ms"] < 500


def test_rank_control_is_cheapest_first_at_flat_markup(client):
    body = client.post("/rank", json={"session": SESSION, "offers": OFFERS, "policy": "control"}).json()
    prices = [o["price_per_day"] for o in body["offers"]]
    assert prices == sorted(prices)
    assert {o["markup"] for o in body["offers"]} == {0.2}


def test_validation_errors(client):
    bad = {"session": {**SESSION, "market": "XX"}, "offers": OFFERS}
    assert client.post("/rank", json=bad).status_code == 422
    assert client.post("/rank", json={"session": SESSION, "offers": OFFERS[:1]}).status_code == 422


def test_metrics_exposed(client):
    client.post("/rank", json={"session": SESSION, "offers": OFFERS})
    text = client.get("/metrics").text
    assert 'rank_requests_total{policy="ml"} 1' in text
    assert "rank_latency_seconds_count 1" in text


def test_partner_caps_differ(client):
    body = client.post("/rank", json={"session": {**SESSION, "partner": "eDreams ODIGEO"}, "offers": OFFERS}).json()
    assert all(o["markup"] <= 0.28 + 1e-9 for o in body["offers"])
    assert body["guardrails"]["markup_max"] == 0.28
    bad = {"session": {**SESSION, "partner": "Nobody"}, "offers": OFFERS}
    assert client.post("/rank", json=bad).status_code == 422
