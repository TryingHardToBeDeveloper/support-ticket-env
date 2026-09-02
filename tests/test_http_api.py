"""Integration tests through the actual FastAPI serialization layer."""

from fastapi.testclient import TestClient

from server.app import app


def test_health_endpoint_and_security_headers():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_reset_rejects_invalid_task_id_over_http():
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/reset", json={"seed": 42, "task_id": 4})
    assert response.status_code == 422
    assert "task_id" in response.json()["detail"]


def test_state_response_contains_no_oracle_fields():
    with TestClient(app) as client:
        reset = client.post("/reset", json={"seed": 42, "task_id": 3})
        assert reset.status_code == 200
        response = client.get("/state")
    assert response.status_code == 200
    serialized = response.text
    assert "correct_category" not in serialized
    assert "correct_action" not in serialized
    assert "resolution_hint" not in serialized
