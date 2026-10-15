"""Tests for the operational layer: metrics, API key auth, rate limiting."""

from fastapi.testclient import TestClient

from app import main as service

client = TestClient(service.app)


def test_metrics_endpoint():
    client.get("/health")  # make sure at least one request was counted
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")

    denied = client.post(
        "/predict", files={"file": ("x.jpg", b"x", "image/jpeg")}
    )
    assert denied.status_code == 401

    wrong = client.post(
        "/predict",
        headers={"X-API-Key": "wrong"},
        files={"file": ("x.jpg", b"x", "image/jpeg")},
    )
    assert wrong.status_code == 401

    # Correct key passes auth; the garbage payload then fails as 415,
    # which proves we reached the endpoint itself.
    allowed = client.post(
        "/predict",
        headers={"X-API-Key": "secret-key"},
        files={"file": ("x.jpg", b"x", "image/jpeg")},
    )
    assert allowed.status_code == 415

    # /health stays open even with auth enabled.
    assert client.get("/health").status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    service._hits.clear()

    statuses = [
        client.post(
            "/predict", files={"file": ("x.jpg", b"x", "image/jpeg")}
        ).status_code
        for _ in range(3)
    ]
    # First two reach the endpoint (415: not an image), third is throttled.
    assert statuses == [415, 415, 429]

    # Non-inference endpoints are never rate limited.
    assert client.get("/health").status_code == 200
    service._hits.clear()
