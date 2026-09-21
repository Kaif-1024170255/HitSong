"""Final model selection and API artifact serialization module for Hit Song Predictor.

Selects the final deployment pipeline, generates models/final/model.joblib and
models/final/metadata.json, and writes the concise reasoned selection report
in reports/model_selection.md.
"""

import datetime
import json
import os
import joblib
import pandas as pd

from src.preprocess import ALL_MODEL_FEATURES, NUMERIC_FEATURES


def extract_feature_bounds(df: pd.DataFrame) -> dict:
    """Calculate min, max, step, and median defaults for UI slider boundaries."""
    bounds = {}
    for feat in NUMERIC_FEATURES:
        is_int_like = feat in ["duration_sec", "tempo"]
        step = 1.0 if is_int_like else 0.01
        bounds[feat] = {
            "min": round(float(df[feat].min()), 3),
            "max": round(float(df[feat].max()), 3),
            "step": step,
            "default": round(float(df[feat].median()), 3),
        }
    return bounds


def get_genre_categories(pipe) -> list[str]:
    """Extract alphabetically sorted genre categories learned by the preprocessor."""
    preproc = pipe.named_steps["preprocessor"]
    ohe = preproc.named_transformers_["cat"]
    return sorted(list(ohe.categories_[0]))


def build_metadata(pipe, df: pd.DataFrame, df_final: pd.DataFrame, df_comp: pd.DataFrame) -> dict:
    """Construct complete production metadata dictionary for FastAPI backend and web UI."""
    j_row = df_final[df_final["threshold_method"].str.contains("Youden")].iloc[0]
    logreg_row = df_comp[df_comp["model"] == "logistic_regression"].iloc[0]

    return {
        "model_name": "logistic_regression",
        "chosen_threshold": float(j_row["threshold"]),
        "threshold_method": "out_of_fold_youden_j_maximizing",
        "exact_ordered_feature_list": ALL_MODEL_FEATURES,
        "genre_categories": get_genre_categories(pipe),
        "per_feature_min_max": extract_feature_bounds(df),
        "test_set_metrics": {
            "accuracy": float(j_row["accuracy"]),
            "precision": float(j_row["precision"]),
            "recall": float(j_row["recall"]),
            "specificity": float(j_row["specificity"]),
            "f1": float(j_row["f1"]),
            "roc_auc": float(logreg_row["ROC-AUC"]),
            "pr_auc": float(logreg_row["PR-AUC"]),
            "TP": int(j_row["TP"]),
            "TN": int(j_row["TN"]),
            "FP": int(j_row["FP"]),
            "FN": int(j_row["FN"]),
        },
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def save_model_artifacts(pipe, metadata: dict, output_dir: str = "models/final") -> None:
    """Serialize full fitted pipeline and production metadata to target directory."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(output_dir, "model.joblib"))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_selection_report(metadata: dict, save_path: str = "reports/model_selection.md") -> None:
    """Write concise, reasoned model selection report strictly under 400 words."""
    m = metadata["test_set_metrics"]
    th = metadata["chosen_threshold"]

    content = f"""# Final Model Selection and Deployment Rationale

### Selected Model and Operational Threshold
- **Selected Model**: Regularized Logistic Regression (`C=1.0`, `class_weight='balanced'`)
- **Operational Threshold**: `{th:.2f}` (Selected via 5-fold training out-of-fold Youden's J maximization)

### Selection Considerations
Model selection evaluated: (1) held-out ROC-AUC, (2) PR-AUC, (3) recall, (4) specificity, (5) interpretability, (6) model complexity, and (7) deployment simplicity.

Accuracy alone is insufficient for this imbalanced dataset. A classifier predicting non-chart for every song achieves approximately 90.9% accuracy while identifying zero hits.

### Multi-Criteria Decision Rationale
Random Forest achieved higher test PR-AUC (0.619 vs 0.606) and higher default precision (0.816) and specificity (0.993), but lower default recall (0.296). As a 300-tree ensemble, it presents a substantially more complex decision structure than Logistic Regression.

Logistic Regression was selected for deployment based on the project's stated emphasis on recall, interpretability, and computational simplicity, while acknowledging that other models performed better on some metrics:
1. **Screening Priorities**: In candidate song screening, false negatives represent missed hits, while false positives represent tracks requiring review. At threshold {th:.2f}, the model identified {m['TP']} of {m['TP'] + m['FN']} charted test tracks ({m['recall']*100:.1f}% recall) and {m['TN']:,} of {m['TN'] + m['FP']:,} non-charted tracks ({m['specificity']*100:.1f}% specificity).
2. **Transparent Model Associations**: Logistic Regression offers transparent model-level interpretation. Standardized numeric coefficients describe changes in log-odds associated with a one-standard-deviation increase in features, while genre coefficients are relative to reference categories. These reflect associative relationships, not causal effects.
3. **Computational Simplicity**: Logistic Regression has a simpler computational structure and lower inference footprint than the 300-tree Random Forest.
4. **Validation Integrity**: Artist-grouped splitting evaluates generalization to unseen artists (artist overlap = 0 across outer splits and inner CV folds), ensuring evaluation without artist leakage.

The operational threshold of {th:.2f} was selected exclusively from 5-fold artist-grouped out-of-fold training predictions maximizing Youden's J; the held-out test set was evaluation-only. This threshold represents an operational operating point balancing false positives and false negatives, rather than a universally optimal threshold.
"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    """Execute final model selection, save models/final artifacts, and report metadata."""
    print("=== Hit Song Predictor: Final Model Selection & Serialization ===")
    df = pd.read_parquet("data/processed/songs.parquet")
    df_comp = pd.read_csv("reports/tables/model_comparison.csv")
    df_final = pd.read_csv("reports/tables/final_threshold_evaluation.csv")

    pipe = joblib.load("models/logistic_regression.joblib")
    metadata = build_metadata(pipe, df, df_final, df_comp)

    save_model_artifacts(pipe, metadata, output_dir="models/final")
    generate_selection_report(metadata, save_path="reports/model_selection.md")

    print("\n--- Serialized Production Metadata (models/final/metadata.json) ---")
    print(json.dumps(metadata, indent=2))
    print("\nArtifacts successfully written:")
    print("  - models/final/model.joblib")
    print("  - models/final/metadata.json")
    print("  - reports/model_selection.md\n")


if __name__ == "__main__":
    main()
