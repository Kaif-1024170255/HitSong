# Final Model Selection and Deployment Rationale

### Selected Model and Operational Threshold
- **Selected Model**: Regularized Logistic Regression (`C=1.0`, `class_weight='balanced'`)
- **Operational Threshold**: `0.48` (Selected via 5-fold training out-of-fold Youden's J maximization)

### Selection Considerations
Model selection evaluated: (1) held-out ROC-AUC, (2) PR-AUC, (3) recall, (4) specificity, (5) interpretability, (6) model complexity, and (7) deployment simplicity.

Accuracy alone is insufficient for this imbalanced dataset. A classifier predicting non-chart for every song achieves approximately 90.9% accuracy while identifying zero hits.

### Multi-Criteria Decision Rationale
Random Forest achieved higher test PR-AUC (0.619 vs 0.606) and higher default precision (0.816) and specificity (0.993), but lower default recall (0.296). As a 300-tree ensemble, it presents a substantially more complex decision structure than Logistic Regression.

Logistic Regression was selected for deployment based on the project's stated emphasis on recall, interpretability, and computational simplicity, while acknowledging that other models performed better on some metrics:
1. **Screening Priorities**: In candidate song screening, false negatives represent missed hits, while false positives represent tracks requiring review. At threshold 0.48, the model identified 115 of 135 charted test tracks (85.2% recall) and 1,153 of 1,345 non-charted tracks (85.7% specificity).
2. **Transparent Model Associations**: Logistic Regression offers transparent model-level interpretation. Standardized numeric coefficients describe changes in log-odds associated with a one-standard-deviation increase in features, while genre coefficients are relative to reference categories. These reflect associative relationships, not causal effects.
3. **Computational Simplicity**: Logistic Regression has a simpler computational structure and lower inference footprint than the 300-tree Random Forest.
4. **Validation Integrity**: Artist-grouped splitting evaluates generalization to unseen artists (artist overlap = 0 across outer splits and inner CV folds), ensuring evaluation without artist leakage.

The operational threshold of 0.48 was selected exclusively from 5-fold artist-grouped out-of-fold training predictions maximizing Youden's J; the held-out test set was evaluation-only. This threshold represents an operational operating point balancing false positives and false negatives, rather than a universally optimal threshold.
