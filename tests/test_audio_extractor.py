"""Unit tests for the audio extractor module."""

import io
import json
import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from src.audio_extractor import (
    load_audio,
    extract_duration,
    extract_tempo,
    extract_loudness,
    estimate_energy_proxy,
    estimate_danceability_proxy,
    clamp_feature,
    extract_features,
    extract_audio_features,
)


@pytest.fixture
def synthetic_mp3_path():
    """Create a temporary 3-second synthetic sine wave MP3 for unit tests."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    sf.write(path, y, sr, format="MP3")
    yield path

    if os.path.exists(path):
        os.remove(path)


def test_load_audio(synthetic_mp3_path):
    """Test loading MP3 into mono numpy array at target sample rate."""
    data, sr = load_audio(synthetic_mp3_path, target_sr=22050)
    assert isinstance(data, np.ndarray)
    assert sr == 22050
    assert len(data) > 0


def test_known_length_exact_duration():
    """Verify that a known-length file gives exact duration."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    try:
        sf.write(path, y, sr, format="MP3")
        data, srate = load_audio(path, target_sr=sr)
        dur = extract_duration(data, srate)
        assert abs(dur - 5.0) < 0.1
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_louder_copy_gives_higher_loudness():
    """Verify that a louder audio signal yields a higher LUFS measurement."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y_quiet = 0.05 * np.sin(2 * np.pi * 440 * t)
    y_loud = 0.50 * np.sin(2 * np.pi * 440 * t)

    lufs_quiet = extract_loudness(y_quiet, sr)
    lufs_loud = extract_loudness(y_loud, sr)
    assert lufs_loud > lufs_quiet


def test_120_bpm_click_track():
    """Verify that a synthetic 120 BPM click track returns tempo within 3 BPM."""
    sr = 22050
    duration = 8.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    # 120 BPM = 2 beats per second -> 0.5s intervals
    interval = int(0.5 * sr)
    click_len = int(0.02 * sr)
    click = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.02, click_len, endpoint=False))

    for idx in range(0, len(y) - click_len, interval):
        y[idx : idx + click_len] += click * 0.9

    bpm = extract_tempo(y, sr)
    assert abs(bpm - 120.0) <= 3.0


def test_corrupt_file_raises_clean_error():
    """Verify that a corrupt MP3 file raises a ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"CORRUPT_NOT_A_VALID_MP3_STREAM_1234567890")
        path = tmp.name
    try:
        with pytest.raises(ValueError, match="Corrupt or unreadable"):
            load_audio(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_estimate_proxies(synthetic_mp3_path):
    """Test energy and danceability proxies within [0, 1]."""
    data, sr = load_audio(synthetic_mp3_path, target_sr=22050)
    energy = estimate_energy_proxy(data)
    dance = estimate_danceability_proxy(data, sr)
    assert 0.0 <= energy <= 1.0
    assert 0.0 <= dance <= 1.0


def test_clamp_feature():
    """Test feature clamping against defined bounds."""
    bounds = {"min": 60.0, "max": 300.0}
    val_low, clamped_low = clamp_feature(45.0, bounds)
    assert val_low == 60.0
    assert clamped_low is True

    val_high, clamped_high = clamp_feature(400.0, bounds)
    assert val_high == 300.0
    assert clamped_high is True

    val_mid, clamped_mid = clamp_feature(180.0, bounds)
    assert val_mid == 180.0
    assert clamped_mid is False


def test_extract_audio_features_integration(synthetic_mp3_path):
    """Test full extraction pipeline with metadata bounds."""
    with open("models/final/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    bounds = meta["per_feature_min_max"]

    res = extract_features(synthetic_mp3_path, bounds)
    expected_feats = ["duration_sec", "tempo", "loudness", "energy", "danceability"]
    for feat in expected_feats:
        assert feat in res
        assert "value" in res[feat]
        assert "method" in res[feat]
        assert "clamped" in res[feat]
        b = bounds[feat]
        assert b["min"] <= res[feat]["value"] <= b["max"]
