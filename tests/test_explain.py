"""Unit tests for src/explain.py module."""

import json
import os
import joblib
import pandas as pd
import pytest

from src.explain import (
    compute_hypothesis_statistics,
    export_top_features_json,
    get_rf_impurity_importance,
)


def test_export_top_features_json(tmp_path):
    """Verify JSON export of top permutation features."""
    df_perm = pd.DataFrame({
        "feature": ["genre", "loudness", "danceability"],
        "importance": [0.12, 0.05, 0.02],
        "std": [0.01, 0.005, 0.002],
    })
    out_file = tmp_path / "top_features.json"
    export_top_features_json(df_perm, str(out_file), top_n=2)

    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["feature"] == "genre"
    assert data[0]["importance"] == 0.12


def test_get_rf_impurity_importance():
    """Verify Random Forest impurity importance aggregates genre into 10 features."""
    rf_pipe = joblib.load("models/random_forest.joblib")
    df_imp = get_rf_impurity_importance(rf_pipe)
    assert len(df_imp) == 10
    assert "genre" in df_imp["feature"].values
    assert "loudness" in df_imp["feature"].values
    assert df_imp["importance"].sum() == pytest.approx(1.0, rel=1e-3)


def test_compute_hypothesis_statistics():
    """Verify hypothesis statistical calculation keys and validity."""
    df = pd.read_parquet("data/processed/songs.parquet")
    stats_dict = compute_hypothesis_statistics(df)
    assert "val_r" in stats_dict
    assert "dur_r" in stats_dict
    assert stats_dict["n_year"] > 0
    assert stats_dict["dur_slope"] < 0
