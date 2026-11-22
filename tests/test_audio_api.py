"""Integration tests for the audio estimation API endpoint."""

import io
import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def synthetic_mp3_bytes():
    """Generate in-memory bytes of a valid synthetic MP3."""
    sr = 22050
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    try:
        sf.write(path, y, sr, format="MP3")
        with open(path, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)

    return data


@pytest.fixture
def client():
    """Create a TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_estimate_audio_flag_disabled(synthetic_mp3_bytes, monkeypatch, client):
    """Verify 403 Forbidden is returned when ENABLE_AUDIO_UPLOAD is false."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "false")
    files = {"file": ("test.mp3", io.BytesIO(synthetic_mp3_bytes), "audio/mpeg")}
    res = client.post("/api/estimate-audio", files=files)
    assert res.status_code == 403
    assert "disabled" in res.json()["detail"].lower()


def test_schema_reflects_audio_upload_flag(monkeypatch, client):
    """Verify /api/schema returns enable_audio_upload matching environment."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    res_enabled = client.get("/api/schema")
    assert res_enabled.status_code == 200
    assert res_enabled.json().get("enable_audio_upload") is True

    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "false")
    res_disabled = client.get("/api/schema")
    assert res_disabled.status_code == 200
    assert res_disabled.json().get("enable_audio_upload") is False


def test_estimate_audio_invalid_extension(monkeypatch, client):
    """Verify 400 Bad Request on non-MP3 files."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    files = {"file": ("test.txt", io.BytesIO(b"not an audio file"), "text/plain")}
    res = client.post("/api/estimate-audio", files=files)
    assert res.status_code == 400
    assert "mp3" in res.json()["detail"].lower()


def test_estimate_audio_empty_file(monkeypatch, client):
    """Verify 400 Bad Request on empty MP3 file."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    files = {"file": ("empty.mp3", io.BytesIO(b""), "audio/mpeg")}
    res = client.post("/api/estimate-audio", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_corrupt_mp3_returns_400_clean_json(monkeypatch, client):
    """Verify 400 Bad Request with clean detail message on corrupt MP3 file."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    files = {"file": ("corrupt.mp3", io.BytesIO(b"NOT_REAL_MP3_DATA_1234567890"), "audio/mpeg")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 400
    assert "corrupt or unreadable" in res.json()["detail"].lower()


def test_oversized_mp3_rejected_cleanly(monkeypatch, client):
    """Verify 400 Bad Request on files exceeding 20 MB."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    oversized_data = b"x" * (20 * 1024 * 1024 + 1024)
    files = {"file": ("huge.mp3", io.BytesIO(oversized_data), "audio/mpeg")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 400
    assert "20mb" in res.json()["detail"].lower()


def test_extract_endpoint_alias(synthetic_mp3_bytes, monkeypatch, client):
    """Verify POST /api/extract endpoint returns same schema as /api/estimate-audio."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    files = {"file": ("track.mp3", io.BytesIO(synthetic_mp3_bytes), "audio/mpeg")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "estimates" in data
    assert "duration_sec" in data["estimates"]


def test_estimate_audio_success_and_clamping(synthetic_mp3_bytes, monkeypatch, client):
    """Verify successful estimation and that all values stay within training bounds."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")
    files = {"file": ("sample.mp3", io.BytesIO(synthetic_mp3_bytes), "audio/mpeg")}
    res = client.post("/api/estimate-audio", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "ok"
    assert data["filename"] == "sample.mp3"
    assert "estimates" in data

    estimates = data["estimates"]
    for feat in ["duration_sec", "tempo", "loudness", "energy", "danceability"]:
        assert feat in estimates
        entry = estimates[feat]
        assert "value" in entry
        assert "clamped" in entry
        assert "method" in entry
        b = entry["bounds"]
        assert b["min"] <= entry["value"] <= b["max"]


def test_estimated_features_pipe_into_predict(synthetic_mp3_bytes, monkeypatch, client):
    """Verify that estimated features pipe into /api/predict without triggering 422."""
    monkeypatch.setenv("ENABLE_AUDIO_UPLOAD", "true")

    # 1. Estimate from audio
    files = {"file": ("song.mp3", io.BytesIO(synthetic_mp3_bytes), "audio/mpeg")}
    res_est = client.post("/api/estimate-audio", files=files)
    assert res_est.status_code == 200
    est = res_est.json()["estimates"]

    # 2. Assemble payload with manual fields
    payload = {
        "duration_sec": est["duration_sec"]["value"],
        "tempo": est["tempo"]["value"],
        "loudness": est["loudness"]["value"],
        "energy": est["energy"]["value"],
        "danceability": est["danceability"]["value"],
        "valence": 0.50,
        "acousticness": 0.20,
        "instrumentalness": 0.05,
        "speechiness": 0.04,
        "genre": "pop",
    }

    # 3. Predict
    res_pred = client.post("/api/predict", json=payload)
    assert res_pred.status_code == 200
    pred = res_pred.json()
    assert "probability" in pred
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["label"] in [0, 1]
