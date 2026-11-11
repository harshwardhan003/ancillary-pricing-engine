import json

from api.app import create_app
from fastapi.testclient import TestClient

offers = [
    {"offer_id": f"o{i}", "supplier": s, "car_class": c, "supplier_rating": r, "cost_per_day": cost}
    for i, (s, c, r, cost) in enumerate([
        ("Alpha", "economy", 4.5, 30.0), ("Bravo", "compact", 4.2, 36.0),
        ("Cobalt", "suv", 3.8, 55.0), ("Delta", "mini", 4.0, 24.0)])
]
session = {"partner": "Ryanair", "market": "IE", "device": "mobile", "returning_customer": False,
           "lead_days": 21, "trip_days": 5, "party_size": 3}

with TestClient(create_app()) as client:
    print(client.get("/health").json())
    body = client.post("/rank", json={"session": session, "offers": offers}).json()
    print(json.dumps(body["guardrails"]))
    for o in body["offers"]:
        print(o)
    print("latency ms:", body["latency_ms"])
