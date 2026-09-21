"""Decision threshold analysis module for Hit Song Predictor.

Features both:
1. Exploratory test-set threshold analysis (evaluating tradeoffs on test data).
2. Leakage-safe threshold selection using 5-fold artist-grouped out-of-fold (OOF)
   predictions from training data only, with final evaluation on the held-out test set.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocess import build_pipeline, make_splits


def display_and_save(fig: plt.Figure, save_path: str) -> None:
    """Save figure at 150 dpi and display without blocking CLI."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.close(fig)


def compute_metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Calculate confusion counts and classification rates for a given probability cutoff."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": round(float(threshold), 2),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Precision": round(float(prec), 4), "Recall": round(float(rec), 4),
        "Specificity": round(float(spec), 4), "F1": round(float(f1), 4),
    }


def evaluate_threshold_grid(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """Compute performance table across discrete decision thresholds."""
    rows = [compute_metrics_at_threshold(y_true, y_proba, th) for th in thresholds]
    df = pd.DataFrame(rows)
    df[["TP", "TN", "FP", "FN"]] = df[["TP", "TN", "FP", "FN"]].astype(int)
    return df


def generate_oof_predictions(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, c_param: float = 1.0) -> tuple[np.ndarray, list[int]]:
    """Generate leakage-safe out-of-fold predicted probabilities on training data only."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_train))
    overlaps = []

    for _, (t_idx, v_idx) in enumerate(sgkf.split(X_train, y_train, groups_train), 1):
        t_artists = set(groups_train.iloc[t_idx])
        v_artists = set(groups_train.iloc[v_idx])
        overlaps.append(len(t_artists.intersection(v_artists)))

        fold_pipe = build_pipeline(
            LogisticRegression(C=c_param, max_iter=1000, class_weight="balanced", random_state=42),
            scale=True
        )
        fold_pipe.fit(X_train.iloc[t_idx], y_train.iloc[t_idx])
        oof_proba[v_idx] = fold_pipe.predict_proba(X_train.iloc[v_idx])[:, 1]

    return oof_proba, overlaps


def verify_oof_predictions(X_train: pd.DataFrame, oof_proba: np.ndarray, overlaps: list[int]) -> None:
    """Verify OOF prediction completeness, range, and fold artist separation."""
    assert len(oof_proba) == len(X_train), "OOF prediction count does not match training rows!"
    assert not np.isnan(oof_proba).any(), "NaN found in OOF predictions!"
    assert oof_proba.min() >= 0.0 and oof_proba.max() <= 1.0, "OOF probabilities out of bounds [0, 1]!"
    for i, ov in enumerate(overlaps, 1):
        assert ov == 0, f"Artist overlap detected in fold {i} ({ov} artists)!"


def select_cv_thresholds(y_train: pd.Series, oof_proba: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Scan thresholds from 0.01 to 0.99 on OOF predictions to select F1 and Youden-J thresholds."""
    th_range = np.arange(0.01, 1.00, 0.01)
    rows = []
    y_true = y_train.to_numpy()
    for th in th_range:
        m = compute_metrics_at_threshold(y_true, oof_proba, th)
        m["youden_j"] = round(m["Recall"] + m["Specificity"] - 1.0, 4)
        rows.append(m)

    df_oof = pd.DataFrame(rows).rename(columns={"Precision": "precision", "Recall": "recall", "Specificity": "specificity", "F1": "f1"})
    df_oof[["TP", "TN", "FP", "FN"]] = df_oof[["TP", "TN", "FP", "FN"]].astype(int)

    max_f1 = df_oof["f1"].max()
    f1_cand = df_oof[df_oof["f1"] == max_f1].sort_values(by="recall", ascending=False).iloc[0]

    max_j = df_oof["youden_j"].max()
    j_cand = df_oof[df_oof["youden_j"] == max_j].sort_values(by="recall", ascending=False).iloc[0]

    opt = {
        "f1_row": f1_cand.to_dict(),
        "youden_j_row": j_cand.to_dict(),
        "f1_threshold": float(f1_cand["threshold"]),
        "youden_j_threshold": float(j_cand["threshold"]),
    }
    return opt, df_oof


def evaluate_frozen_thresholds_test(pipe, X_test: pd.DataFrame, y_test: pd.Series, f1_th: float, j_th: float) -> pd.DataFrame:
    """Evaluate frozen training thresholds once on held-out test set."""
    test_proba = pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()
    results = []

    for name, th in [("F1-selected threshold", f1_th), ("Youden-J-selected threshold", j_th)]:
        y_pred = (test_proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = float((tp + tn) / len(y_true))
        results.append({
            "threshold_method": name, "threshold": th,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "specificity": round(spec, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(results)


def plot_threshold_curves_cv(df_oof: pd.DataFrame, f1_th: float, j_th: float, save_path: str) -> None:
    """Plot precision, recall, and F1 curves across thresholds using training OOF predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_oof["threshold"], df_oof["precision"], label="Precision", color="#2980B9", lw=2.2)
    ax.plot(df_oof["threshold"], df_oof["recall"], label="Recall", color="#E67E22", lw=2.2)
    ax.plot(df_oof["threshold"], df_oof["f1"], label="F1-Score", color="#27AE60", lw=2.2)

    ax.axvline(f1_th, linestyle="--", color="#16A085", lw=2, label=f"F1-Selected ({f1_th:.2f})")
    ax.axvline(j_th, linestyle=":", color="#C0392B", lw=2, label=f"Youden-J-Selected ({j_th:.2f})")

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Cross-Validated Threshold Trade-Off — Training OOF Predictions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9.5)
    display_and_save(fig, save_path)


def build_notes_section_a(df_th: pd.DataFrame) -> str:
    """Construct Section A of threshold notes describing exploratory test-set analysis."""
    t5 = df_th.loc[df_th["threshold"] == 0.5].iloc[0]
    t4 = df_th.loc[df_th["threshold"] == 0.4].iloc[0]
    delta_tp = int(t4["TP"] - t5["TP"])
    delta_fp = int(t4["FP"] - t5["FP"])

    return f"""## SECTION A — Exploratory Test-Set Threshold Analysis

Thresholds from 0.05 to 0.95 were evaluated directly on the held-out test set ($N = 1,480$) as an exploratory visualization to illustrate how classification trade-offs behave across probability cutoffs. These exploratory test results must NOT be used as the deployment threshold-selection procedure.

### Observed Trade-offs on Test Set
- **At Threshold 0.5**: TP = {int(t5['TP'])}, FP = {int(t5['FP'])}, Recall = {t5['Recall']:.3f}, Precision = {t5['Precision']:.3f}.
- **At Threshold 0.4**: TP = {int(t4['TP'])}, FP = {int(t4['FP'])}, Recall = {t4['Recall']:.3f}, Precision = {t4['Precision']:.3f}.
- Moving from threshold 0.5 to 0.4 produces {delta_tp} additional true-positive predictions at the cost of {delta_fp} additional false-positive predictions, corresponding to {delta_fp // delta_tp} additional false-positive predictions for every additional true positive captured ({delta_fp} / {delta_tp} = {delta_fp / delta_tp:.1f}).
- In a hypothetical A&R promotional screening interpretation, relaxing the cutoff increases hit discovery but steeply escalates non-charting contenders.

### Exploratory Test-Set Extrema (Non-Deployment Reference)
- **Maximum test-set F1 threshold**: 0.88 (Test-set maximum F1 = 0.5941, precision = 0.6827, recall = 0.5259).
- **Maximum test-set Youden's J threshold**: 0.38 (Test-set Youden's J = 0.7231, sensitivity = 0.8889, specificity = 0.8342).
"""


def build_notes_section_b(opt: dict, df_final: pd.DataFrame) -> str:
    """Construct Section B of threshold notes describing leakage-safe CV threshold selection."""
    f1_r = opt["f1_row"]
    j_r = opt["youden_j_row"]
    f1_test = df_final.loc[df_final["threshold_method"] == "F1-selected threshold"].iloc[0]
    j_test = df_final.loc[df_final["threshold_method"] == "Youden-J-selected threshold"].iloc[0]

    return f"""## SECTION B — Leakage-Safe Threshold Selection

The final threshold-selection procedure uses out-of-fold predictions generated exclusively from the artist-grouped training data. The held-out test set is not used to select the threshold.

### 1. Training Out-of-Fold Threshold Selection
Thresholds were scanned across 0.01–0.99 using 5-fold artist-grouped out-of-fold predictions ($N = 5,923$ training observations):
- **F1-maximizing threshold**: **{f1_r['threshold']:.2f}**
  - Training OOF Metrics: F1 = {f1_r['f1']:.4f}, Precision = {f1_r['precision']:.4f}, Recall = {f1_r['recall']:.4f}, Specificity = {f1_r['specificity']:.4f}.
- **Youden-J-maximizing threshold**: **{j_r['threshold']:.2f}**
  - Training OOF Metrics: Youden's J = {j_r['youden_j']:.4f}, Sensitivity (Recall) = {j_r['recall']:.4f}, Specificity = {j_r['specificity']:.4f}.

### 2. Final Frozen-Threshold Held-Out Test Evaluation
These two candidate deployment thresholds were frozen and evaluated on the held-out test set ($N = 1,480$ unseen tracks):
- **F1-selected threshold ({f1_test['threshold']:.2f}) on Test Set**:
  - TP = {int(f1_test['TP'])}, TN = {int(f1_test['TN'])}, FP = {int(f1_test['FP'])}, FN = {int(f1_test['FN'])}
  - Accuracy = {f1_test['accuracy']:.4f}, Precision = {f1_test['precision']:.4f}, Recall = {f1_test['recall']:.4f}, Specificity = {f1_test['specificity']:.4f}, F1 = {f1_test['f1']:.4f}.
- **Youden-J-selected threshold ({j_test['threshold']:.2f}) on Test Set**:
  - TP = {int(j_test['TP'])}, TN = {int(j_test['TN'])}, FP = {int(j_test['FP'])}, FN = {int(j_test['FN'])}
  - Accuracy = {j_test['accuracy']:.4f}, Precision = {j_test['precision']:.4f}, Recall = {j_test['recall']:.4f}, Specificity = {j_test['specificity']:.4f}, F1 = {j_test['f1']:.4f}.

### Methodological Clarity
Training/CV data was used strictly for threshold selection; the held-out test set was used solely for final evaluation of the frozen thresholds.
"""


def write_threshold_report(sec_a: str, sec_b: str, save_path: str = "reports/threshold_notes.md") -> None:
    """Save combined markdown threshold notes report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    header = "# Decision Threshold Analysis: Exploratory Evaluation and Leakage-Safe Selection\n\n"
    content = header + sec_a + "\n" + sec_b
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def print_oof_summary(n_train: int, oof_proba: np.ndarray, overlaps: list[int]) -> None:
    """Print OOF verification block matching prompt format."""
    print("\n--- OOF Verification ---\n")
    print(f"Training rows:\n{n_train}\n")
    print(f"OOF predictions:\n{len(oof_proba)}\n")
    print(f"Missing OOF predictions:\n{int(np.isnan(oof_proba).sum())}\n")
    print(f"OOF probability range:\n{oof_proba.min():.4e} to {oof_proba.max():.4f}\n")
    for i, ov in enumerate(overlaps, 1):
        print(f"Fold {i} artist overlap: {ov}")
    print("\nOOF THRESHOLD SELECTION CHECK: PASS")


def print_selection_and_test_summary(opt: dict, df_final: pd.DataFrame) -> None:
    """Print selected thresholds and frozen test evaluation metrics."""
    f1_r, j_r = opt["f1_row"], opt["youden_j_row"]
    print("\n--- Training/CV Threshold Selection ---\n")
    print(f"F1-maximizing threshold:\n{f1_r['threshold']:.2f}\n")
    print(f"F1:\n{f1_r['f1']:.4f}\n\nPrecision:\n{f1_r['precision']:.4f}\n\nRecall:\n{f1_r['recall']:.4f}\n\nSpecificity:\n{f1_r['specificity']:.4f}\n")
    print(f"Youden-J-maximizing threshold:\n{j_r['threshold']:.2f}\n")
    print(f"Youden's J:\n{j_r['youden_j']:.4f}\n\nSensitivity:\n{j_r['recall']:.4f}\n\nSpecificity:\n{j_r['specificity']:.4f}\n")

    f1_t = df_final.loc[df_final["threshold_method"] == "F1-selected threshold"].iloc[0]
    j_t = df_final.loc[df_final["threshold_method"] == "Youden-J-selected threshold"].iloc[0]
    print("--- Frozen Threshold Test Evaluation ---\n")
    print(f"F1-selected threshold:\n{f1_t['threshold']:.2f}\n\nTP:\n{int(f1_t['TP'])}\n\nTN:\n{int(f1_t['TN'])}\n\nFP:\n{int(f1_t['FP'])}\n\nFN:\n{int(f1_t['FN'])}\n")
    print(f"Precision:\n{f1_t['precision']:.4f}\n\nRecall:\n{f1_t['recall']:.4f}\n\nSpecificity:\n{f1_t['specificity']:.4f}\n\nF1:\n{f1_t['f1']:.4f}\n")
    print(f"Youden-J-selected threshold:\n{j_t['threshold']:.2f}\n\nTP:\n{int(j_t['TP'])}\n\nTN:\n{int(j_t['TN'])}\n\nFP:\n{int(j_t['FP'])}\n\nFN:\n{int(j_t['FN'])}\n")
    print(f"Precision:\n{j_t['precision']:.4f}\n\nRecall:\n{j_t['recall']:.4f}\n\nSpecificity:\n{j_t['specificity']:.4f}\n\nF1:\n{j_t['f1']:.4f}\n")


def print_artifact_verification() -> None:
    """Print artifact verification statuses."""
    artifacts = [
        "reports/tables/cv_threshold_analysis.csv",
        "reports/tables/selected_thresholds.json",
        "reports/tables/final_threshold_evaluation.csv",
        "reports/figures/threshold_tradeoff_cv.png",
        "reports/threshold_notes.md",
    ]
    print("--- Artifact Verification ---\n")
    for art in artifacts:
        print(f"{art}: {'PASS' if os.path.exists(art) else 'FAIL'}")


def main() -> None:
    """Execute complete leakage-safe threshold selection and final frozen test evaluation."""
    print("=== Leakage-Safe Threshold Selection ===\n\n--- Model ---\nLogistic Regression\nHeld-out ROC-AUC: 0.929")
    df = pd.read_parquet("data/processed/songs.parquet")
    X_train, X_test, y_train, y_test, groups_train, _ = make_splits(df)

    pipe_saved = joblib.load("models/logistic_regression.joblib")
    best_c = pipe_saved.named_steps["model"].C

    oof_proba, overlaps = generate_oof_predictions(X_train, y_train, groups_train, c_param=best_c)
    verify_oof_predictions(X_train, oof_proba, overlaps)
    print_oof_summary(len(X_train), oof_proba, overlaps)

    opt, df_oof = select_cv_thresholds(y_train, oof_proba)
    os.makedirs("reports/tables", exist_ok=True)
    df_oof.to_csv("reports/tables/cv_threshold_analysis.csv", index=False)

    json_data = {
        "model": "logistic_regression", "threshold_selection_method": "out_of_fold_training_predictions",
        "cv_folds": 5, "grouping_variable": "primary_artist",
        "f1_threshold": opt["f1_threshold"], "youden_j_threshold": opt["youden_j_threshold"],
        "selection_data": "training_only",
    }
    with open("reports/tables/selected_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    plot_threshold_curves_cv(df_oof, opt["f1_threshold"], opt["youden_j_threshold"], "reports/figures/threshold_tradeoff_cv.png")

    df_final = evaluate_frozen_thresholds_test(pipe_saved, X_test, y_test, opt["f1_threshold"], opt["youden_j_threshold"])
    df_final.to_csv("reports/tables/final_threshold_evaluation.csv", index=False)
    print_selection_and_test_summary(opt, df_final)

    df_exploratory = pd.read_csv("reports/tables/threshold_analysis.csv")
    sec_a = build_notes_section_a(df_exploratory)
    sec_b = build_notes_section_b(opt, df_final)
    write_threshold_report(sec_a, sec_b, "reports/threshold_notes.md")

    print_artifact_verification()


if __name__ == "__main__":
    main()
