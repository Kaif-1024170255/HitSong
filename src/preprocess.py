"""Preprocessing and dataset splitting module for Hit Song Predictor.

Defines the canonical model schema, artist-grouped train/test splitting,
pipeline preprocessors (StandardScaler and OneHotEncoder), and chart-year coverage analysis.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical Modelling Schema Constants
NUMERIC_FEATURES = [
    'tempo', 'energy', 'danceability', 'valence', 'loudness',
    'acousticness', 'instrumentalness', 'speechiness', 'duration_sec'
]
CATEGORICAL_FEATURES = ['genre']
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = 'charted'
GROUP_COL = 'primary_artist'
ANALYSIS_COL = 'chart_year'


def normalise_artist(artist: str) -> str:
    """Normalise primary artist name for grouping only."""
    if not isinstance(artist, str):
        return "unknown"
    return re.sub(r'\s+', ' ', artist.lower().strip())


def select_best_fold(df: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Select candidate fold with 0 artist overlap, ~20% test size, and closest positive rate."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    full_pos_rate = df[TARGET].mean()
    best_candidate = None
    min_dist = float('inf')

    for train_idx, test_idx in sgkf.split(df, df[TARGET], groups):
        overlap = len(set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx])))
        test_pct = len(test_idx) / len(df)
        pos_diff = abs(df[TARGET].iloc[test_idx].mean() - full_pos_rate)
        dist = overlap * 1000.0 + abs(test_pct - 0.20) * 10.0 + pos_diff * 5.0
        if dist < min_dist:
            min_dist = dist
            best_candidate = (train_idx, test_idx)
    return best_candidate


def print_split_summary(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, groups: pd.Series) -> float:
    """Print split statistics, artist overlap, class balance, and return positive rate difference."""
    n_total, n_train, n_test = len(df), len(train_idx), len(test_idx)
    train_y, test_y = df[TARGET].iloc[train_idx], df[TARGET].iloc[test_idx]
    train_groups, test_groups = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    overlap = len(train_groups.intersection(test_groups))

    diff = abs(train_y.mean() - test_y.mean())

    print("\n--- Split Statistics ---")
    print(f"Total rows: {n_total}")
    print(f"Training rows: {n_train} ({n_train / n_total * 100:.2f}%)")
    print(f"Test rows: {n_test} ({n_test / n_total * 100:.2f}%)")
    print(f"Full-dataset positive count: {(df[TARGET] == 1).sum()} ({df[TARGET].mean() * 100:.2f}%)")
    print(f"Training positive count: {(train_y == 1).sum()} ({train_y.mean() * 100:.2f}%)")
    print(f"Training negative count: {(train_y == 0).sum()} ({(train_y == 0).mean() * 100:.2f}%)")
    print(f"Test positive count: {(test_y == 1).sum()} ({test_y.mean() * 100:.2f}%)")
    print(f"Test negative count: {(test_y == 0).sum()} ({(test_y == 0).mean() * 100:.2f}%)")
    print(f"Unique normalised artists in full dataset: {groups.nunique()}")
    print(f"Unique normalised artists in training: {len(train_groups)}")
    print(f"Unique normalised artists in testing: {len(test_groups)}")
    print(f"Artists appearing in BOTH train and test: {overlap}")
    print(f"Absolute train/test positive-class rate difference: {diff * 100:.2f} percentage points")
    return diff


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform artist-grouped ~80/20 train/test split ensuring zero artist overlap.

    Rationale for Artist Grouping:
    Songs by the same artist may share production, vocal, genre, and stylistic
    characteristics. Allowing the same artist to appear in both train and test can
    make evaluation optimistic because the model is tested partly on artists represented
    during training. The grouped split therefore evaluates generalisation to unseen artists.
    This does not claim to completely eliminate all forms of leakage.

    Selection Procedure:
    Uses StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) to generate candidate
    folds. Folds are evaluated deterministically to select the split with zero artist overlap,
    test size closest to 20%, and test positive-class proportion closest to the full dataset.
    """
    groups = df[GROUP_COL].apply(normalise_artist)
    train_idx, test_idx = select_best_fold(df, groups)

    diff = print_split_summary(df, train_idx, test_idx, groups)
    if diff > 0.02:
        raise ValueError(f"Class imbalance difference ({diff * 100:.2f}pp) exceeds 2 percentage points threshold!")

    X_train = df.iloc[train_idx][ALL_MODEL_FEATURES].copy()
    X_test = df.iloc[test_idx][ALL_MODEL_FEATURES].copy()
    y_train = df.iloc[train_idx][TARGET].copy()
    y_test = df.iloc[test_idx][TARGET].copy()
    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def build_preprocessor(scale: bool = True, dense: bool = False) -> ColumnTransformer:
    """Build ColumnTransformer for numeric and categorical features."""
    num_step = StandardScaler() if scale else "passthrough"
    cat_step = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=not dense)
    return ColumnTransformer(
        transformers=[
            ("num", num_step, NUMERIC_FEATURES),
            ("cat", cat_step, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3 if not dense else 0.0,
    )


def build_pipeline(estimator, scale: bool = True, dense: bool = False) -> Pipeline:
    """Wrap estimator with preprocessor inside a scikit-learn Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor(scale=scale, dense=dense)),
        ("model", estimator),
    ])


def summarise_chart_year_coverage(df: pd.DataFrame) -> dict:
    """Analyse chart_year coverage. ANALYSIS ONLY; chart_year is never used for training."""
    charted_mask = df[TARGET] == 1
    non_charted_mask = df[TARGET] == 0

    n_charted = int(charted_mask.sum())
    n_charted_valid_year = int(df.loc[charted_mask, ANALYSIS_COL].notnull().sum())
    n_charted_missing_year = int(df.loc[charted_mask, ANALYSIS_COL].isnull().sum())
    n_non_charted_valid_year = int(df.loc[non_charted_mask, ANALYSIS_COL].notnull().sum())

    print("\n--- Chart Year Coverage Analysis ---")
    print(f"Number of charted observations: {n_charted}")
    print(f"Number of charted observations with valid chart_year: {n_charted_valid_year}")
    print(f"Number of charted observations missing chart_year: {n_charted_missing_year}")
    print(f"Number of non-charted observations with valid chart_year: {n_non_charted_valid_year}")

    return {
        "n_charted": n_charted,
        "n_charted_valid_year": n_charted_valid_year,
        "n_charted_missing_year": n_charted_missing_year,
        "n_non_charted_valid_year": n_non_charted_valid_year,
    }


def main() -> None:
    """Execute split and coverage checks on the processed modelling dataset."""
    print("=== Hit Song Predictor: Dataset Splitting & Preprocessing ===")
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, X_test, y_train, y_test, groups_train, groups_test = make_splits(df)

    overlap = len(set(groups_train).intersection(set(groups_test)))
    pos_diff = abs(y_train.mean() - y_test.mean())

    summarise_chart_year_coverage(df)

    print("\n--- Split Verification Checks ---")
    if overlap == 0:
        print("ARTIST OVERLAP CHECK: PASS")
    else:
        print(f"ARTIST OVERLAP CHECK: FAIL ({overlap} overlapping artists)")

    if pos_diff <= 0.02:
        print("CLASS BALANCE CHECK: PASS")
    else:
        print(f"CLASS BALANCE CHECK: FAIL ({pos_diff * 100:.2f}pp diff)")


if __name__ == "__main__":
    main()
