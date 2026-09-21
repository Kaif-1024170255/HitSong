"""Unit tests for unsupervised audio clustering module in src/clustering.py."""

import os
import pandas as pd
import pytest

from src.clustering import build_cluster_profiles, fit_kmeans, load_and_scale_features
from src.preprocess import NUMERIC_FEATURES


def test_load_and_scale_features():
    """Verify that feature loading scales all 9 continuous audio features."""
    df, X_scaled, scaler = load_and_scale_features()
    assert len(df) == 7403
    assert X_scaled.shape == (7403, 9)
    assert hasattr(scaler, "mean_")


def test_fit_kmeans():
    """Verify that K-Means clusters reproduce deterministically for k=4."""
    _, X_scaled, _ = load_and_scale_features()
    km, labels = fit_kmeans(X_scaled, k=4)
    assert km.n_clusters == 4
    assert len(labels) == 7403
    assert set(labels) == {0, 1, 2, 3}


def test_cluster_profiles_artifact():
    """Verify that reports/tables/cluster_profiles.csv contains expected columns and rows."""
    path = "reports/tables/cluster_profiles.csv"
    assert os.path.exists(path), f"Missing {path}"
    df_prof = pd.read_csv(path)
    assert len(df_prof) == 4

    required_cols = ["cluster", "cluster_name", "size", "charted_rate", "genre_composition"] + NUMERIC_FEATURES
    for col in required_cols:
        assert col in df_prof.columns, f"Missing column {col} in cluster_profiles.csv"

    assert df_prof["size"].sum() == 7403


def test_clustering_figures_exist():
    """Verify that both required clustering visualization artifacts exist."""
    fig1 = "reports/figures/cluster_selection.png"
    fig2 = "reports/figures/clusters_pca.png"
    assert os.path.exists(fig1), f"Missing {fig1}"
    assert os.path.exists(fig2), f"Missing {fig2}"
    assert os.path.getsize(fig1) > 1000
    assert os.path.getsize(fig2) > 1000
