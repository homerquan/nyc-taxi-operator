from datetime import datetime

from fastapi.testclient import TestClient

from backend.app.main import app, cached_forecast


def test_api_contracts(dataset, checkpoint, monkeypatch):
    data, _, _ = dataset
    monkeypatch.setenv("NYC_DATA_DIR", str(data))
    monkeypatch.setenv("NYC_ARTIFACTS_DIR", str(checkpoint))
    cached_forecast.cache_clear()
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ready"
    assert len(client.get("/api/zones").json()["features"]) == 3
    time = "2025-01-01T10:00:00-05:00"
    actual = client.get("/api/actuals", params={"as_of": time}).json()
    assert actual["available"] and actual["total_trips"] == 4
    response = client.get("/api/forecast", params={"as_of":time,"top_k":2})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert len(body["predictions"][0]["departure_bin_probabilities"]) == 4
    assert body["summary"]["predicted_total"] > 0
    route = client.get("/api/route", params={"as_of": time, "source_zone":1, "target_zone":2}).json()
    assert route["actual"]["trip_count"] == 3
    assert route["prediction"]["source_zone"] == 1
    assert client.get("/api/forecast", params={"as_of":"2025-01-01T10:00:00"}).status_code == 422
    assert client.get("/api/forecast", params={"as_of":time,"horizon_minutes":30}).status_code == 422
    assert client.get("/api/zone/999/forecast", params={"as_of":time}).status_code == 404
