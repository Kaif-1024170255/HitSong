"""Integration and unit tests for FastAPI backend service in app/main.py."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance running application lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_pop_song():
    """Fixture providing a realistic pop song payload."""
    return {
        "tempo": 120.0,
        "energy": 0.80,
        "danceability": 0.75,
        "valence": 0.65,
        "loudness": -5.0,
        "acousticness": 0.15,
        "instrumentalness": 0.0,
        "speechiness": 0.05,
        "duration_sec": 210.0,
        "genre": "pop",
    }


def test_health_endpoint(client):
    """Verify that /api/health returns 200 with status ok and metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_name"] == "logistic_regression"
    assert "trained_at" in data


def test_schema_endpoint(client):
    """Verify that /api/schema returns feature definitions and slider bounds."""
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "sliders" in data
    assert "genres" in data
    assert "pop" in data["genres"]
    assert 0.0 < data["threshold"] < 1.0


def test_insights_endpoint(client):
    """Verify that /api/insights returns feature importances and model benchmark table."""
    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    assert "model_comparison" in data
    assert len(data["feature_importance"]) > 0
    assert len(data["model_comparison"]) > 0


def test_predict_success(client, valid_pop_song):
    """Verify that /api/predict returns probability in [0, 1] and contributions."""
    response = client.post("/api/predict", json=valid_pop_song)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["label"] in [0, 1]
    assert data["threshold"] == 0.48
    assert "contributions" in data
    assert len(data["contributions"]) == 10
    for c in data["contributions"]:
        assert "feature" in c
        assert "value" in c
        assert "effect" in c


def test_predict_out_of_range_field(client, valid_pop_song):
    """Verify that an out-of-range numeric field returns 422 with informative error."""
    payload = valid_pop_song.copy()
    payload["tempo"] = 999.0  # Training max is 217.282
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    messages = [err["message"] for err in data["detail"]]
    assert any("tempo" in m and "between" in m for m in messages)


def test_predict_missing_field(client, valid_pop_song):
    """Verify that a missing required field returns 422 naming the missing field."""
    payload = valid_pop_song.copy()
    del payload["danceability"]
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    fields = [err["field"] for err in data["detail"]]
    assert "danceability" in fields


def test_predict_invalid_genre(client, valid_pop_song):
    """Verify that an unknown genre category returns 422."""
    payload = valid_pop_song.copy()
    payload["genre"] = "nonexistent_martian_rock"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    fields = [err["field"] for err in data["detail"]]
    assert "genre" in fields


def test_static_root(client):
    """Verify that GET / serves the static web frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Hit Song Predictor" in response.text
