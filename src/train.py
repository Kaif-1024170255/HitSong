"""Model training and hyperparameter tuning module for Hit Song Predictor.

Trains eight classification models using 5-fold cross-validation on the
training split produced by src.preprocess.make_splits(). Saves full fitted
pipelines to models/<model_name>.joblib and metrics to reports/tables/.
"""

import os
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.preprocess import make_splits, build_pipeline


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
    configs.update(get_svm_configs())
    configs.update(get_tree_configs())
    configs.update(get_other_configs())
    order = [
        "logistic_regression", "tree_gini", "tree_entropy", "random_forest",
        "naive_bayes", "knn", "svm_linear", "svm_rbf"
    ]
    return {k: configs[k] for k in order}


def verify_svm_safety(n_train: int, max_threshold: int = 50000) -> bool:
    """Verify training set row count is within safe limits for SVM training."""
    print(f"\n[SVM Safety Check] Training rows: {n_train} (Threshold: {max_threshold})")
    if n_train > max_threshold:
        raise RuntimeError(
            f"Training set size ({n_train}) exceeds safety threshold ({max_threshold})!"
        )
    print("[SVM Safety Check] Passed: proceeding with SVM training.")
    return True


def train_single_model(name: str, config: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit GridSearchCV on training split, save pipeline artifact, and return results."""
    print(f"\n{'=' * 60}\nModel: {name}\nParameter Grid: {config['param_grid']}")
    pipeline = build_pipeline(config["estimator"], scale=config["scale"], dense=config["dense"])
    grid = GridSearchCV(pipeline, config["param_grid"], cv=5, scoring="roc_auc", n_jobs=-1, return_train_score=True)

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    mean_auc = grid.cv_results_["mean_test_score"][grid.best_index_]
    std_auc = grid.cv_results_["std_test_score"][grid.best_index_]
    print(f"Fit Time: {fit_time:.2f}s | Best CV ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"Best Hyperparameters: {grid.best_params_}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, os.path.join("models", f"{name}.joblib"))

    return {
        "model_name": name,
        "best_hyperparameters": str(grid.best_params_),
        "best_cv_roc_auc_mean": mean_auc,
        "best_cv_roc_auc_std": std_auc,
        "training_time_sec": fit_time,
    }


def save_reports(results: list[dict], output_dir: str = "reports/tables") -> pd.DataFrame:
    """Save training_summary.csv and best_params.csv sorted by CV ROC-AUC."""
    os.makedirs(output_dir, exist_ok=True)
    df_summary = pd.DataFrame(results)
    df_summary = df_summary.sort_values(by="best_cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    df_summary.to_csv(os.path.join(output_dir, "training_summary.csv"), index=False)

    df_best = df_summary[["model_name", "best_hyperparameters"]].rename(
        columns={"model_name": "model", "best_hyperparameters": "best_params"}
    )
    df_best.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    return df_summary


def print_summary_table(df_summary: pd.DataFrame) -> None:
    """Print clean formatted summary table of model performance ranked by CV ROC-AUC."""
    print(f"\n{'=' * 80}")
    print("FINAL TRAINING SUMMARY (Ranked by CV ROC-AUC)")
    print(f"{'=' * 80}")
    header = f"{'Model':<22} | {'Best CV ROC-AUC':<20} | {'Time (s)':<10} | {'Best Hyperparameters'}"
    print(header)
    print("-" * len(header))
    for _, row in df_summary.iterrows():
        auc_str = f"{row['best_cv_roc_auc_mean']:.4f} +/- {row['best_cv_roc_auc_std']:.4f}"
        print(f"{row['model_name']:<22} | {auc_str:<20} | {row['training_time_sec']:<10.2f} | {row['best_hyperparameters']}")
    print(f"{'=' * 80}\n")


def main() -> None:
    """Execute model training pipeline for all 8 models on training split."""
    print("=== Hit Song Predictor: Model Training & Hyperparameter Tuning ===")
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, _, y_train, _, _, _ = make_splits(df)

    verify_svm_safety(len(X_train))

    configs = get_model_configs()
    results = []
    for name, config in configs.items():
        res = train_single_model(name, config, X_train, y_train)
        results.append(res)

    df_summary = save_reports(results)
    print_summary_table(df_summary)


if __name__ == "__main__":
    main()
