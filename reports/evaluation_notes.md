# Test Set Evaluation Notes

## Baseline Context and Accuracy Caution
A model predicting "never charts" for every song would score 90.9% accuracy on this test set, which is why accuracy is not the headline metric here.

Because non-charted songs make up 90.9% of the held-out sample, classifiers can trivially achieve high accuracy by predicting the negative class for almost all tracks. Evaluation must therefore focus on ranking discrimination (ROC-AUC), precision-recall trade-offs (PR-AUC), and minority-class detection (Recall, F1-Score).

## Metric Leaders Summary
- **ROC-AUC Leader**: `logistic_regression` (0.929) — best overall threshold-independent ranking across all false positive rates.
- **PR-AUC Leader**: `random_forest` (0.619) — superior precision-recall trade-off across the minority positive class.
- **F1-Score Leader**: `logistic_regression` (0.524) — optimal balance between precision and recall at default decision threshold.
- **Recall Leader**: `naive_bayes` (0.911) — catches the highest fraction of charted songs, minimizing false negatives.
- **Precision Leader**: `random_forest` (0.816) — lowest rate of false alarms when predicting a hit song.

## Model Disagreements and Operational Trade-Offs
1. **Class-Weighted vs Unweighted Decision Boundaries**:
   - `logistic_regression`, `svm_linear`, and `svm_rbf` utilized balanced class weights. Consequently, their default threshold yields high Recall (77.0%–84.4%) and balanced Specificity (~85.2%–86.5%), making them suitable when missing a genuine hit is costly.
   - `random_forest`, `tree_gini`, and `tree_entropy` were unweighted or conservative, yielding high Specificity (98.2%–99.3%) and high nominal Accuracy (92.6%–93.0%), but capturing only 29.6%–36.3% of true hits.
2. **Global Discriminability vs Operating Point**:
   - Although `random_forest` achieves moderate Recall at the default 0.5 probability cut-off, its `PR-AUC` (0.619) is the highest among all models. Adjusting its decision threshold would allow practitioners to operate at substantially higher recall without sacrificing precision as steeply as linear models.
