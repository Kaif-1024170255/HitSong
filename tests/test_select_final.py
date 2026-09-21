"""Unit tests for src/select_final.py module and production deployment artifacts."""

import json
import os
import joblib
import pandas as pd
import pytest

from src.preprocess import ALL_MODEL_FEATURES, NUMERIC_FEATURES


def test_final_model_artifact_load_and_predict():
    """Verify that models/final/model.joblib loads and produces valid predictions."""
    model_path = "models/final/model.joblib"
    assert os.path.exists(model_path), f"Missing {model_path}"

    pipe = joblib.load(model_path)
    assert hasattr(pipe, "predict_proba")
    assert "preprocessor" in pipe.named_steps
    assert "model" in pipe.named_steps

    # Test prediction on single dummy row with exact schema
    dummy_input = pd.DataFrame([{
        "tempo": 120.0,
        "energy": 0.75,
        "danceability": 0.80,
        "valence": 0.50,
        "loudness": -5.0,
        "acousticness": 0.10,
        "instrumentalness": 0.0,
        "speechiness": 0.05,
        "duration_sec": 210.0,
        "genre": "pop"
    }])[ALL_MODEL_FEATURES]

    proba = pipe.predict_proba(dummy_input)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_final_metadata_structure_and_types():
    """Verify that models/final/metadata.json contains all required schema keys."""
    meta_path = "models/final/metadata.json"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "model_name", "chosen_threshold", "exact_ordered_feature_list",
        "genre_categories", "per_feature_min_max", "test_set_metrics", "trained_at"
    ]
    for k in required_keys:
        assert k in meta, f"Missing key {k} in metadata.json"

    assert meta["exact_ordered_feature_list"] == ALL_MODEL_FEATURES
    assert len(meta["genre_categories"]) > 100
    assert 0.0 < meta["chosen_threshold"] < 1.0

    metrics = meta["test_set_metrics"]
    for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        assert 0.0 <= metrics[m] <= 1.0

    for feat in NUMERIC_FEATURES:
        assert feat in meta["per_feature_min_max"]
        b = meta["per_feature_min_max"][feat]
        assert b["min"] < b["max"]


def test_selection_report_word_count():
    """Verify that reports/model_selection.md is strictly under 400 words."""
    report_path = "reports/model_selection.md"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.strip().split()
    assert len(words) < 400, f"model_selection.md exceeds 400 words ({len(words)})"
