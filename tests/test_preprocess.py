"""Unit tests for src/preprocess.py module."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.preprocess import (
    ALL_MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    GROUP_COL,
    ANALYSIS_COL,
    build_preprocessor,
    build_pipeline,
    make_splits,
    summarise_chart_year_coverage,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Load the processed modelling dataset."""
    return pd.read_parquet("data/processed/songs.parquet")


@pytest.fixture(scope="module")
def split_data(dataset: pd.DataFrame):
    """Run make_splits once for test reuse."""
    return make_splits(dataset)


def test_artist_separation(split_data):
    """Test A: Mandatory check that train and test artist groups are completely disjoint."""
    _, _, _, _, groups_train, groups_test = split_data
    assert set(groups_train).isdisjoint(set(groups_test)), "Artist overlap detected between train and test sets!"


def test_class_balance(split_data):
    """Test B: Assert that positive-class rates in train and test differ by <= 2 percentage points."""
    _, _, y_train, y_test, _, _ = split_data
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    diff = abs(train_pos_rate - test_pos_rate)
    assert diff <= 0.02, f"Positive rate difference {diff:.4f} exceeds 0.02 threshold!"


def test_approximate_test_size(split_data):
    """Test C: Assert that test set is approximately 20% of total observations (tolerance: 3 percentage points)."""
    X_train, X_test, _, _, _, _ = split_data
    total_rows = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_rows
    # Tolerance of 3 percentage points (17% to 23%) accounts for atomic group assignment constraints
    assert abs(test_ratio - 0.20) <= 0.03, f"Test ratio {test_ratio:.4f} is outside [0.17, 0.23] bounds!"


def test_no_nans_after_preprocessing(split_data):
    """Test D: Fit preprocessor on X_train only and assert no NaNs in transformed matrices."""
    X_train, X_test, _, _, _, _ = split_data
    preprocessor = build_preprocessor(scale=True)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Handle sparse or dense outputs
    if sparse.issparse(X_train_trans):
        assert not np.isnan(X_train_trans.data).any(), "NaN found in transformed sparse X_train!"
        assert not np.isnan(X_test_trans.data).any(), "NaN found in transformed sparse X_test!"
    else:
        assert not np.isnan(X_train_trans).any(), "NaN found in transformed dense X_train!"
        assert not np.isnan(X_test_trans).any(), "NaN found in transformed dense X_test!"


def test_scaler_fitted_only_on_training(split_data):
    """Test E: Verify StandardScaler is fitted on X_train only and matches X_train numeric means."""
    X_train, _, _, _, _, _ = split_data
    preprocessor = build_preprocessor(scale=True)
    preprocessor.fit(X_train)

    scaler = preprocessor.named_transformers_["num"]
    expected_means = X_train[NUMERIC_FEATURES].mean().values
    assert np.allclose(scaler.mean_, expected_means, atol=1e-5), "Scaler mean does not match X_train numeric means!"


def test_feature_exclusions(split_data):
    """Test F: Assert X_train and X_test contain strictly the 9 audio features + genre, excluding leakage/helpers."""
    X_train, X_test, _, _, _, _ = split_data
    forbidden = [GROUP_COL, ANALYSIS_COL, TARGET]
    for col in forbidden:
        assert col not in X_train.columns, f"Forbidden column {col} present in X_train!"
        assert col not in X_test.columns, f"Forbidden column {col} present in X_test!"

    assert list(X_train.columns) == ALL_MODEL_FEATURES, "X_train feature list does not match canonical model features!"
    assert list(X_test.columns) == ALL_MODEL_FEATURES, "X_test feature list does not match canonical model features!"


def test_reproducibility(dataset: pd.DataFrame):
    """Test G: Assert that make_splits produces identical splits across multiple runs with random_state=42."""
    split1 = make_splits(dataset)
    split2 = make_splits(dataset)

    pd.testing.assert_frame_equal(split1[0], split2[0])
    pd.testing.assert_frame_equal(split1[1], split2[1])
    pd.testing.assert_series_equal(split1[2], split2[2])
    pd.testing.assert_series_equal(split1[3], split2[3])


def test_target_integrity(split_data):
    """Test H: Assert target contains only binary values {0, 1} and both classes exist in train and test."""
    _, _, y_train, y_test, _, _ = split_data
    assert set(y_train.unique()).issubset({0, 1}), "y_train contains non-binary values!"
    assert set(y_test.unique()).issubset({0, 1}), "y_test contains non-binary values!"
    assert (y_train == 0).sum() > 0 and (y_train == 1).sum() > 0, "y_train must contain both classes!"
    assert (y_test == 0).sum() > 0 and (y_test == 1).sum() > 0, "y_test must contain both classes!"


def test_chart_year_coverage_analysis(dataset: pd.DataFrame):
    """Test coverage summary helper for chart_year analysis."""
    stats = summarise_chart_year_coverage(dataset)
    assert stats["n_charted"] == (dataset[TARGET] == 1).sum()
    assert stats["n_non_charted_valid_year"] == 0
    assert stats["n_charted_valid_year"] + stats["n_charted_missing_year"] == stats["n_charted"]
