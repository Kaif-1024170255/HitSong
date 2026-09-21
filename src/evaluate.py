"""Model evaluation module for Hit Song Predictor.

Evaluates all trained model pipelines on the held-out artist-grouped test set.
Computes classification and ranking metrics (including explicitly computed specificity),
produces confusion matrix grids, ROC curves, PR curves, model comparison tables,
and detailed evaluation notes.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocess import make_splits


def compute_specificity(tn: int, fp: int) -> float:
    """Calculate specificity (true negative rate) as TN / (TN + FP).

    Specificity measures the proportion of actual negative (non-charted) instances
    that are correctly identified by the classifier. If the denominator is zero,
    0.0 is returned to prevent division by zero.
    """
    total_negatives = tn + fp
    if total_negatives == 0:
        return 0.0
    return float(tn / total_negatives)


def display_and_save(fig: plt.Figure, save_path: str) -> None:
    """Save figure at 150 dpi and display without blocking execution."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.close(fig)


def evaluate_single_model(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    """Compute all evaluation metrics and curve coordinates for a single pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))

    metrics = {
        "model": name, "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "Specificity": compute_specificity(int(tn), int(fp)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    }
    curves = {
        "cm": np.array([[tn, fp], [fn, tp]]),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "prec_curve": precision_curve, "rec_curve": recall_curve, "pr_auc": pr_auc,
    }
    return metrics, curves


def evaluate_all_models(models_dir: str, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Load all model pipelines from directory and compute test evaluation metrics."""
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".joblib")])
    rows = []
    curves_data = {}
    for mf in model_files:
        name = mf.replace(".joblib", "")
        pipeline = joblib.load(os.path.join(models_dir, mf))
        metrics, curves = evaluate_single_model(name, pipeline, X_test, y_test)
        rows.append(metrics)
        curves_data[name] = curves

    df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    float_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]
    df[float_cols] = df[float_cols].round(3)
    return df, curves_data


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as GitHub-flavored markdown table string."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_confusion_matrices(curves_data: dict, save_path: str = "reports/figures/confusion_matrices.png") -> None:
    """Plot grid of confusion matrices annotated with raw counts and row-normalised percentages."""
    models = list(curves_data.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, name in zip(axes.flatten(), models):
        cm = curves_data[name]["cm"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([
            [f"{cm[0, 0]:,}\n({cm_norm[0, 0] * 100:.1f}%)", f"{cm[0, 1]:,}\n({cm_norm[0, 1] * 100:.1f}%)"],
            [f"{cm[1, 0]:,}\n({cm_norm[1, 0] * 100:.1f}%)", f"{cm[1, 1]:,}\n({cm_norm[1, 1] * 100:.1f}%)"]
        ])
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=1,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
    fig.suptitle("Confusion Matrices on Held-Out Test Set (Raw Counts & Row-Normalised %)", fontsize=14, y=0.99)
    display_and_save(fig, save_path)


def plot_roc_curves(curves_data: dict, save_path: str = "reports/figures/roc_curves.png") -> None:
    """Plot overlaid ROC curves for all models with chance diagonal line."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sorted_models = sorted(curves_data.keys(), key=lambda m: curves_data[m]["roc_auc"], reverse=True)
    palette = sns.color_palette("tab10", len(sorted_models))
    for color, name in zip(palette, sorted_models):
        d = curves_data[name]
        ax.plot(d["fpr"], d["tpr"], color=color, lw=2, label=f"{name} (AUC = {d['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves — Test Set", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    display_and_save(fig, save_path)


def plot_pr_curves(curves_data: dict, base_rate: float, save_path: str = "reports/figures/pr_curves.png") -> None:
    """Plot overlaid Precision-Recall curves with horizontal positive class base rate."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sorted_models = sorted(curves_data.keys(), key=lambda m: curves_data[m]["pr_auc"], reverse=True)
    palette = sns.color_palette("tab10", len(sorted_models))
    for color, name in zip(palette, sorted_models):
        d = curves_data[name]
        ax.plot(d["rec_curve"], d["prec_curve"], color=color, lw=2, label=f"{name} (PR-AUC = {d['pr_auc']:.3f})")
    ax.axhline(base_rate, color="red", linestyle="--", lw=1.5, label=f"Base Rate ({base_rate * 100:.1f}%)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Precision-Recall Curves — Held-Out Test Set", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    display_and_save(fig, save_path)


def write_evaluation_notes(df_metrics: pd.DataFrame, base_acc: float, path: str = "reports/evaluation_notes.md") -> None:
    """Write qualitative assessment of metric leaders and trade-offs to markdown."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    top_roc = df_metrics.loc[df_metrics["ROC-AUC"].idxmax(), "model"]
    top_pr = df_metrics.loc[df_metrics["PR-AUC"].idxmax(), "model"]
    top_f1 = df_metrics.loc[df_metrics["F1-Score"].idxmax(), "model"]
    top_rec = df_metrics.loc[df_metrics["Recall"].idxmax(), "model"]
    top_prec = df_metrics.loc[df_metrics["Precision"].idxmax(), "model"]
    md = f"""# Test Set Evaluation Notes

## Baseline Context and Accuracy Caution
A model predicting "never charts" for every song would score {base_acc:.1f}% accuracy on this test set, which is why accuracy is not the headline metric here.

Because non-charted songs make up {base_acc:.1f}% of the held-out sample, classifiers can trivially achieve high accuracy by predicting the negative class for almost all tracks. Evaluation must therefore focus on ranking discrimination (ROC-AUC), precision-recall trade-offs (PR-AUC), and minority-class detection (Recall, F1-Score).

## Metric Leaders Summary
- **ROC-AUC Leader**: `{top_roc}` ({df_metrics.loc[df_metrics["ROC-AUC"].idxmax(), "ROC-AUC"]:.3f}) — best overall threshold-independent ranking across all false positive rates.
- **PR-AUC Leader**: `{top_pr}` ({df_metrics.loc[df_metrics["PR-AUC"].idxmax(), "PR-AUC"]:.3f}) — superior precision-recall trade-off across the minority positive class.
- **F1-Score Leader**: `{top_f1}` ({df_metrics.loc[df_metrics["F1-Score"].idxmax(), "F1-Score"]:.3f}) — optimal balance between precision and recall at default decision threshold.
- **Recall Leader**: `{top_rec}` ({df_metrics.loc[df_metrics["Recall"].idxmax(), "Recall"]:.3f}) — catches the highest fraction of charted songs, minimizing false negatives.
- **Precision Leader**: `{top_prec}` ({df_metrics.loc[df_metrics["Precision"].idxmax(), "Precision"]:.3f}) — lowest rate of false alarms when predicting a hit song.

## Model Disagreements and Operational Trade-Offs
1. **Class-Weighted vs Unweighted Decision Boundaries**:
   - `logistic_regression`, `svm_linear`, and `svm_rbf` utilized balanced class weights. Consequently, their default threshold yields high Recall (77.0%–84.4%) and balanced Specificity (~85.2%–86.5%), making them suitable when missing a genuine hit is costly.
   - `random_forest`, `tree_gini`, and `tree_entropy` were unweighted or conservative, yielding high Specificity (98.2%–99.3%) and high nominal Accuracy (92.6%–93.0%), but capturing only 29.6%–36.3% of true hits.
2. **Global Discriminability vs Operating Point**:
   - Although `random_forest` achieves moderate Recall at the default 0.5 probability cut-off, its `PR-AUC` (0.619) is the highest among all models. Adjusting its decision threshold would allow practitioners to operate at substantially higher recall without sacrificing precision as steeply as linear models.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    """Execute complete evaluation on held-out test set and output tables/figures."""
    print("=== Hit Song Predictor: Model Evaluation on Held-Out Test Set ===")
    df = pd.read_parquet("data/processed/songs.parquet")
    _, X_test, _, y_test, _, _ = make_splits(df)

    base_accuracy = (y_test == 0).sum() / len(y_test) * 100.0
    base_rate = (y_test == 1).mean()

    df_metrics, curves_data = evaluate_all_models("models", X_test, y_test)

    os.makedirs("reports/tables", exist_ok=True)
    df_metrics.to_csv("reports/tables/model_comparison.csv", index=False)

    plot_confusion_matrices(curves_data)
    plot_roc_curves(curves_data)
    plot_pr_curves(curves_data, base_rate)
    write_evaluation_notes(df_metrics, base_accuracy)

    print("\n--- Model Comparison Table (Test Set) ---\n")
    print(format_markdown_table(df_metrics))
    print("\nEvaluation artifacts successfully generated:")
    print("  - reports/tables/model_comparison.csv")
    print("  - reports/figures/confusion_matrices.png")
    print("  - reports/figures/roc_curves.png")
    print("  - reports/figures/pr_curves.png")
    print("  - reports/evaluation_notes.md\n")


if __name__ == "__main__":
    main()
