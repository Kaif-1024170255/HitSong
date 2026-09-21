"""Unit tests for src/evaluate.py module."""

import pandas as pd
import pytest

from src.evaluate import compute_specificity, format_markdown_table


def test_compute_specificity_normal():
    """Verify standard specificity calculations."""
    assert compute_specificity(80, 20) == 0.8
    assert compute_specificity(100, 0) == 1.0
    assert compute_specificity(0, 100) == 0.0


def test_compute_specificity_zero_division():
    """Verify specificity handles zero total negatives without raising error."""
    assert compute_specificity(0, 0) == 0.0


def test_format_markdown_table():
    """Verify markdown table formatter output."""
    df = pd.DataFrame({
        "Model": ["M1", "M2"],
        "ROC-AUC": [0.92, 0.85]
    })
    md = format_markdown_table(df)
    assert "| Model | ROC-AUC |" in md
    assert "| --- | --- |" in md
    assert "| M1 | 0.92 |" in md
    assert "| M2 | 0.85 |" in md
