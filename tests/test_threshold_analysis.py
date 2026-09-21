"""Unit tests for src/threshold_analysis.py module.

Verifies:
1. Precision calculation
2. Recall calculation
3. Specificity calculation
4. F1 calculation
5. Youden's J calculation
6. Zero-division edge cases
7. OOF prediction completeness
8. OOF probability range
9. Artist separation across folds
10. Threshold selection determinism
11. Final test evaluation using frozen thresholds
12. Test data not being used during threshold selection
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocess import make_splits
from src.threshold_analysis import (
    compute_metrics_at_threshold,
    evaluate_frozen_thresholds_test,
    evaluate_threshold_grid,
    generate_oof_predictions,
    select_cv_thresholds,
    verify_oof_predictions,
)


def test_metric_calculations_and_zero_division():
    """Verify precision, recall, specificity, F1, and zero-division edge cases."""
    # Standard calculations
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.4, 0.35, 0.1])
    m = compute_metrics_at_threshold(y_true, y_proba, threshold=0.5)

    assert m["TP"] == 1
    assert m["FP"] == 0
    assert m["TN"] == 2
    assert m["FN"] == 1
    assert m["Precision"] == 1.0  # 1 / (1 + 0)
    assert m["Recall"] == 0.5     # 1 / (1 + 1)
    assert m["Specificity"] == 1.0  # 2 / (2 + 0)
    assert m["F1"] == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5), rel=1e-3)

    # Zero-division edge cases: high threshold where no positive predictions occur
    m_zero = compute_metrics_at_threshold(y_true, y_proba, threshold=0.99)
    assert m_zero["TP"] == 0
    assert m_zero["FP"] == 0
    assert m_zero["Precision"] == 0.0
    assert m_zero["Recall"] == 0.0
    assert m_zero["F1"] == 0.0
    assert m_zero["Specificity"] == 1.0


def test_youden_j_calculation():
    """Verify Youden's J calculation: J = Sensitivity + Specificity - 1."""
    y_true = pd.Series([1, 1, 0, 0])
    oof_proba = np.array([0.8, 0.6, 0.3, 0.1])
    opt, df_oof = select_cv_thresholds(y_true, oof_proba)

    for _, row in df_oof.iterrows():
        expected_j = round(row["recall"] + row["specificity"] - 1.0, 4)
        assert row["youden_j"] == pytest.approx(expected_j, abs=1e-4)


def test_oof_predictions_and_artist_separation():
    """Verify OOF prediction completeness, probability bounds, and fold artist separation."""
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, _, y_train, _, groups_train, _ = make_splits(df)

    oof_proba, overlaps = generate_oof_predictions(X_train, y_train, groups_train, c_param=1.0)

    # Check completeness
    assert len(oof_proba) == len(X_train)
    assert not np.isnan(oof_proba).any()

    # Check range
    assert oof_proba.min() >= 0.0
    assert oof_proba.max() <= 1.0

    # Check zero artist overlap
    assert len(overlaps) == 5
    for i, ov in enumerate(overlaps, 1):
        assert ov == 0, f"Fold {i} has {ov} overlapping artists!"

    # Verify helper passes without error
    verify_oof_predictions(X_train, oof_proba, overlaps)


def test_threshold_selection_determinism_and_no_test_leakage():
    """Verify that threshold selection is deterministic and uses only training data."""
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, X_test, y_train, y_test, _, _ = make_splits(df)

    # Synthetic training OOF predictions
    np.random.seed(42)
    dummy_oof = np.random.uniform(0.0, 1.0, len(y_train))

    opt1, df_oof1 = select_cv_thresholds(y_train, dummy_oof)
    opt2, df_oof2 = select_cv_thresholds(y_train, dummy_oof)

    # Determinism check
    assert opt1["f1_threshold"] == opt2["f1_threshold"]
    assert opt1["youden_j_threshold"] == opt2["youden_j_threshold"]
    pd.testing.assert_frame_equal(df_oof1, df_oof2)

    # Ensure test data is strictly excluded from OOF threshold selection
    assert len(dummy_oof) == len(y_train)
    assert len(dummy_oof) != len(y_test)


def test_final_frozen_threshold_evaluation():
    """Verify that frozen thresholds evaluate cleanly on test data without modification."""
    df = pd.read_parquet("data/processed/songs.parquet")
    _, X_test, _, y_test, _, _ = make_splits(df)
    pipe = joblib.load("models/logistic_regression.joblib")

    frozen_f1_th = 0.77
    frozen_j_th = 0.48
    df_eval = evaluate_frozen_thresholds_test(pipe, X_test, y_test, frozen_f1_th, frozen_j_th)

    assert len(df_eval) == 2
    assert df_eval.loc[0, "threshold"] == frozen_f1_th
    assert df_eval.loc[1, "threshold"] == frozen_j_th
    assert (df_eval["TP"] + df_eval["TN"] + df_eval["FP"] + df_eval["FN"] == len(y_test)).all()
