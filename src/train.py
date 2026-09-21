"""Model training and group-aware hyperparameter tuning module for Hit Song Predictor.

Trains eight classification models using StratifiedGroupKFold 5-fold cross-validation
on the training split produced by src.preprocess.make_splits(). Grouping on primary_artist
ensures zero artist leakage across inner CV folds and outer splits. Saves fitted pipelines
to models/<model_name>.joblib and evaluation tables to reports/tables/.
"""

import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import ALL_MODEL_FEATURES, build_pipeline, make_splits


def get_tree_configs() -> dict:
    """Return model configurations for decision trees and random forest."""
    return {
        "tree_gini": {
            "estimator": DecisionTreeClassifier(criterion="gini", random_state=42),
            "scale": False,
            "dense": False,
            "param_grid": {
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
            },
        },
        "tree_entropy": {
            "estimator": DecisionTreeClassifier(criterion="entropy", random_state=42),
            "scale": False,
            "dense": False,
            "param_grid": {
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "scale": False,
            "dense": False,
            "param_grid": {
                "model__max_depth": [8, 15, None],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        },
    }


def get_linear_configs() -> dict:
    """Return model configuration for Logistic Regression."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "scale": True,
            "dense": False,
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
    }


def get_svm_configs() -> dict:
    """Return model configurations for Support Vector Machines."""
    return {
        "svm_linear": {
            "estimator": SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            "scale": True,
            "dense": False,
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "svm_rbf": {
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            "scale": True,
            "dense": False,
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
            },
        },
    }


def get_other_configs() -> dict:
    """Return model configurations for Naive Bayes and K-Nearest Neighbors."""
    return {
        "naive_bayes": {
            "estimator": GaussianNB(),
            "scale": False,
            "dense": True,
            "param_grid": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "scale": True,
            "dense": False,
            "param_grid": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            },
        },
    }


def get_model_configs() -> dict:
    """Combine and return all eight model configurations in canonical order."""
    configs = {}
    configs.update(get_linear_configs())
    configs.update(get_tree_configs())
    configs.update(get_other_configs())
    configs.update(get_svm_configs())
    order = [
        "logistic_regression", "tree_gini", "tree_entropy", "random_forest",
        "naive_bayes", "knn", "svm_linear", "svm_rbf"
    ]
    return {k: configs[k] for k in order}


def verify_svm_safety(n_train: int, max_threshold: int = 50000) -> bool:
    """Verify training set row count is within safe limits for SVM training."""
    print(f"\n[SVM Safety Check] Training rows: {n_train} (Threshold: {max_threshold})")
    if n_train > max_threshold:
        raise RuntimeError(f"Training set size ({n_train}) exceeds safety threshold ({max_threshold})!")
    print("[SVM Safety Check] Passed: proceeding with SVM training.")
    return True


def verify_inner_cv_groups(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv: StratifiedGroupKFold) -> None:
    """Verify that no artist appears in both train and validation splits in any inner fold."""
    print("\n--- Inner CV Group Verification ---\n")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_artists = set(groups.iloc[train_idx])
        val_artists = set(groups.iloc[val_idx])
        overlap = len(train_artists.intersection(val_artists))
        print(f"Fold {fold} artist overlap: {overlap}")
        if overlap != 0:
            raise ValueError(f"Inner CV Fold {fold} has {overlap} overlapping artists!")
    print("\nINNER CV ARTIST SEPARATION CHECK: PASS")


def run_pipeline_checks(
    X_train: pd.DataFrame, X_test: pd.DataFrame, groups_train: pd.Series, groups_test: pd.Series
) -> None:
    """Run verification checks on feature set, leakage exclusion, missing values, and split separation."""
    assert "primary_artist" not in X_train.columns, "primary_artist leaked into X_train features!"
    assert "chart_year" not in X_train.columns, "chart_year leaked into X_train features!"
    assert list(X_train.columns) == ALL_MODEL_FEATURES, "X_train columns do not match canonical schema!"
    assert X_train.isnull().sum().sum() == 0, "Missing values found in X_train!"
    assert X_test.isnull().sum().sum() == 0, "Missing values found in X_test!"

    outer_overlap = len(set(groups_train).intersection(set(groups_test)))
    assert outer_overlap == 0, f"Outer train/test artist overlap detected ({outer_overlap})!"


def train_single_model(
    name: str, config: dict, X_train: pd.DataFrame, y_train: pd.Series,
    groups_train: pd.Series, cv: StratifiedGroupKFold
) -> dict:
    """Fit group-aware GridSearchCV on training split, save pipeline, and return metrics."""
    print(f"\n{'=' * 60}\nModel: {name}\nParameter Grid: {config['param_grid']}")
    pipeline = build_pipeline(config["estimator"], scale=config["scale"], dense=config["dense"])
    grid = GridSearchCV(pipeline, config["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True)

    t0 = time.perf_counter()
    grid.fit(X_train, y_train, groups=groups_train)
    fit_time = time.perf_counter() - t0

    mean_auc = grid.cv_results_["mean_test_score"][grid.best_index_]
    std_auc = grid.cv_results_["std_test_score"][grid.best_index_]
    print(f"Fit Time: {fit_time:.2f}s")
    print(f"Best Group-Aware CV ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"Best Hyperparameters: {grid.best_params_}\n{'=' * 60}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, os.path.join("models", f"{name}.joblib"))

    return {
        "model": name,
        "best_cv_roc_auc": mean_auc,
        "cv_std": std_auc,
        "fit_time_seconds": fit_time,
        "cv_folds": 5,
        "group_aware_cv": True,
        "best_params": str(grid.best_params_),
    }


def save_reports(results: list[dict], output_dir: str = "reports/tables") -> pd.DataFrame:
    """Save training_summary.csv and best_params.csv sorted by group-aware CV ROC-AUC."""
    os.makedirs(output_dir, exist_ok=True)
    df_summary = pd.DataFrame(results)
    df_summary = df_summary.sort_values(by="best_cv_roc_auc", ascending=False).reset_index(drop=True)
    df_summary.to_csv(os.path.join(output_dir, "training_summary.csv"), index=False)

    df_best = df_summary[["model", "best_cv_roc_auc", "cv_std", "best_params"]].copy()
    df_best.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    return df_summary


def print_summary_table(df_summary: pd.DataFrame) -> None:
    """Print formatted summary table of model performance ranked by group-aware CV ROC-AUC."""
    print(f"\n{'=' * 80}")
    print("FINAL TRAINING SUMMARY (Ranked by Group-Aware CV ROC-AUC)")
    print(f"{'=' * 80}\n")
    header = f"{'Model':<22} | {'Best CV ROC-AUC':<15} | {'Std':<8} | {'Time (s)':<10} | {'Best Hyperparameters'}"
    print(header)
    print("-" * 80)
    for _, row in df_summary.iterrows():
        auc_str = f"{row['best_cv_roc_auc']:.4f}"
        std_str = f"{row['cv_std']:.4f}"
        time_str = f"{row['fit_time_seconds']:.2f}"
        print(f"{row['model']:<22} | {auc_str:<15} | {std_str:<8} | {time_str:<10} | {row['best_params']}")
    print(f"{'=' * 80}\n")


def main() -> None:
    """Execute model training and group-aware hyperparameter tuning on training split."""
    print("=== Hit Song Predictor: Model Training & Group-Aware Hyperparameter Tuning ===")
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, X_test, y_train, _, groups_train, groups_test = make_splits(df)

    run_pipeline_checks(X_train, X_test, groups_train, groups_test)

    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    verify_inner_cv_groups(X_train, y_train, groups_train, inner_cv)

    verify_svm_safety(len(X_train))

    configs = get_model_configs()
    results = []
    for name, config in configs.items():
        res = train_single_model(name, config, X_train, y_train, groups_train, inner_cv)
        results.append(res)

    df_summary = save_reports(results)
    print_summary_table(df_summary)


if __name__ == "__main__":
    main()
