# Decision Threshold Analysis: Exploratory Evaluation and Leakage-Safe Selection

## SECTION A — Exploratory Test-Set Threshold Analysis

Thresholds from 0.05 to 0.95 were evaluated directly on the held-out test set ($N = 1,480$) as an exploratory visualization to illustrate how classification trade-offs behave across probability cutoffs. These exploratory test results must NOT be used as the deployment threshold-selection procedure.

### Observed Trade-offs on Test Set
- **At Threshold 0.5**: TP = 114, FP = 186, Recall = 0.844, Precision = 0.380.
- **At Threshold 0.4**: TP = 116, FP = 218, Recall = 0.859, Precision = 0.347.
- Moving from threshold 0.5 to 0.4 produces 2 additional true-positive predictions at the cost of 32 additional false-positive predictions, corresponding to 16 additional false-positive predictions for every additional true positive captured (32 / 2 = 16.0).
- In a hypothetical A&R promotional screening interpretation, relaxing the cutoff increases hit discovery but steeply escalates non-charting contenders.

### Exploratory Test-Set Extrema (Non-Deployment Reference)
- **Maximum test-set F1 threshold**: 0.88 (Test-set maximum F1 = 0.5941, precision = 0.6827, recall = 0.5259).
- **Maximum test-set Youden's J threshold**: 0.38 (Test-set Youden's J = 0.7231, sensitivity = 0.8889, specificity = 0.8342).

## SECTION B — Leakage-Safe Threshold Selection

The final threshold-selection procedure uses out-of-fold predictions generated exclusively from the artist-grouped training data. The held-out test set is not used to select the threshold.

### 1. Training Out-of-Fold Threshold Selection
Thresholds were scanned across 0.01–0.99 using 5-fold artist-grouped out-of-fold predictions ($N = 5,923$ training observations):
- **F1-maximizing threshold**: **0.77**
  - Training OOF Metrics: F1 = 0.6293, Precision = 0.5466, Recall = 0.7416, Specificity = 0.9385.
- **Youden-J-maximizing threshold**: **0.48**
  - Training OOF Metrics: Youden's J = 0.7831, Sensitivity (Recall) = 0.9071, Specificity = 0.8760.

### 2. Final Frozen-Threshold Held-Out Test Evaluation
These two candidate deployment thresholds were frozen and evaluated on the held-out test set ($N = 1,480$ unseen tracks):
- **F1-selected threshold (0.77) on Test Set**:
  - TP = 91, TN = 1252, FP = 93, FN = 44
  - Accuracy = 0.9074, Precision = 0.4946, Recall = 0.6741, Specificity = 0.9309, F1 = 0.5705.
- **Youden-J-selected threshold (0.48) on Test Set**:
  - TP = 115, TN = 1153, FP = 192, FN = 20
  - Accuracy = 0.8568, Precision = 0.3746, Recall = 0.8519, Specificity = 0.8572, F1 = 0.5204.

### Methodological Clarity
Training/CV data was used strictly for threshold selection; the held-out test set was used solely for final evaluation of the frozen thresholds.
