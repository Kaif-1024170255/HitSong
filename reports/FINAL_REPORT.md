# Hit Song Predictor: Academic Machine Learning Research Report

**Authors**: Machine Learning Project Team  
**Subject**: Binary Supervised Classification of Billboard Hot 100 Hits from Audio Features  
**Dataset Time Horizon**: 2000–2023  
**Artifact Directory**: `reports/`  

---

## 1. Introduction

The modern recorded music industry operates under intense commercial uncertainty. With tens of thousands of tracks uploaded daily to global streaming platforms, commercial record labels and Artists and Repertoire (A&R) executives face substantial opportunity costs in scouting and capital allocation. Traditional music discovery has historically relied on executive intuition, tastemakers, and subjective auditions.

This project investigates whether quantitative audio engineering features—measuring continuous musical properties such as tempo, energy, danceability, valence, and loudness—contain statistically robust signals that distinguish commercial hits on the Billboard Hot 100 from non-charting recorded music. As an academic project, the focus is placed on strict methodological rigor: avoiding data and artist leakage, preventing overly optimistic accuracy evaluation on imbalanced distributions, evaluating multi-criteria trade-offs, and deploying an explainable system.

---

## 2. Problem Statement

Given a set of standardized musical properties and genre categorization for an unreleased or newly published song, formulate a binary supervised classification problem:

$$\hat{y} \in \{0, 1\} = f(\mathbf{x})$$

where:
- $\mathbf{x} \in \mathbb{R}^9 \times \mathcal{G}$ represents the vector of 9 continuous Spotify audio descriptors alongside a categorical genre identifier $\mathcal{G} \in \{1, \dots, 114\}$.
- $y = 1$ denotes that the song charted on the historical Billboard Year-End Hot 100 between 2000 and 2023.
- $y = 0$ denotes that the song did not achieve a Year-End Hot 100 placement.

### Key Technical & Commercial Challenges
1. **Severe Natural Class Imbalance**: In the natural commercial distribution, charting songs constitute fewer than 1% of recorded tracks (0.84% in our raw candidate pool). Trivial majority-class classifiers achieve $>99\%$ accuracy while providing zero commercial utility.
2. **Artist Leakage**: Popular superstar artists (e.g., Drake, Taylor Swift) possess built-in promotional reach and radio distribution. If the same artist appears in both training and evaluation splits, models inadvertently memorize artist identity rather than learning generalizable acoustic patterns.
3. **Asymmetric Error Costs**: For an A&R screening tool, a **False Negative** (failing to identify a future chart hit) represents a catastrophic loss of streaming royalties and catalog equity. In contrast, a **False Positive** (auditing a non-charting track) incurs a manageable screening overhead.

---

## 3. Objectives

1. **Construct a Leakage-Free Dataset**: Formulate a clean, verified modelling dataset with an empirically controlled 10:1 negative-to-positive ratio, removing ambiguous records and isolating artist metadata strictly for grouping.
2. **Execute Artist-Grouped Partitioning**: Enforce an 80/20 train/test split with **zero artist overlap**, guaranteeing that test performance reflects generalization to completely unseen artists.
3. **Train & Tune Multi-Family Classifiers**: Systematically benchmark 8 distinct model families using artist-grouped 5-fold cross-validation (`StratifiedGroupKFold`) with ROC-AUC scoring.
4. **Conduct Rigorous Explainability & Hypothesis Testing**: Contrast impurity-based feature rankings with held-out permutation importances and standardized logistic coefficients, testing whether musical hypotheses are supported by empirical evidence.
5. **Formulate a Leakage-Safe Operational Threshold**: Select operational decision thresholds exclusively from training out-of-fold predictions, evaluating the frozen threshold once on held-out test data.
6. **Deliver an End-to-End Production Service**: Build a validated, accessible FastAPI service and manuscript-style web client complying with the visual design specification.

---

## 4. Dataset

### Source Data & Filtering
The raw data originates from historical Billboard Year-End Hot 100 charts (2000–2023) mapped to Spotify's Web API audio features.

| Dataset Metric | Value | Source File / Verification |
|:---|:---:|:---|
| **Raw Candidate Pool** | 80,237 tracks | `reports/tables/dataset_summary.csv` |
| **Natural Positive Count** | 673 tracks (0.84%) | `reports/tables/dataset_summary.csv` |
| **Natural Negative Pool** | 79,564 tracks (99.16%) | `reports/tables/dataset_summary.csv` |
| **Ambiguous Excluded Tracks** | 11 tracks | `reports/tables/ambiguous_excluded.csv` |
| **Final Downsampled Total** | **7,403 tracks** | `reports/tables/dataset_summary.csv` |
| **Final Positive Count** | 673 tracks (9.09%) | `data/processed/songs.parquet` |
| **Final Negative Count** | 6,730 tracks (90.91%) | `data/processed/songs.parquet` |
| **Controlled Class Ratio** | 10:1 Negative-to-Positive | `reports/tables/dataset_summary.csv` |

![Class Distribution (Natural vs Modeling)](figures/class_balance_modeling.png)

### Canonical Feature Schema
The modelling schema consists strictly of 10 predictive features, with metadata columns strictly quarantined:

- **9 Continuous Audio Features**:
  - `tempo` (BPM: 30.2 to 217.3, median 122.0)
  - `energy` (0.002 to 1.000, median 0.688)
  - `danceability` (0.061 to 0.980, median 0.586)
  - `valence` (0.000 to 0.992, median 0.454)
  - `loudness` (dB: -38.6 to +0.91, median -6.94)
  - `acousticness` (0.000 to 0.996, median 0.170)
  - `instrumentalness` (0.000 to 0.999, median 0.000)
  - `speechiness` (0.023 to 0.959, median 0.049)
  - `duration_sec` (seconds: 60.7 to 555.4, median 215.2)
- **1 Categorical Feature**: `genre` (114 distinct musical subgenres)
- **Grouping Column (Non-Predictive)**: `primary_artist` (4,581 unique normalized artists)
- **Temporal Column (Non-Predictive)**: `chart_year` (preserved for post-hoc coverage audits)

---

## 5. Pre-Processing

### Artist-Grouped Outer Split
To evaluate true commercial generalization to new and emerging talent, songs were partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on normalized `primary_artist`:

| Partition | Total Songs | Charted ($y=1$) | Non-Charted ($y=0$) | Positive Rate | Unique Artists | Artist Overlap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training Set** | 5,923 (80.01%) | 538 (9.08%) | 5,385 (90.92%) | 9.08% | 3,657 | — |
| **Test Set** | 1,480 (19.99%) | 135 (9.12%) | 1,345 (90.88%) | 9.12% | 924 | **0** |

The absolute difference in positive class rates between train and test is just **0.04 percentage points**, confirming stratified preservation of the 10:1 ratio while completely eliminating artist contamination.

### Data Leakage Prevention
1. **Transformer Fitting**: `StandardScaler` was fit exclusively on the training fold ($\mu_{\text{train}}, \sigma_{\text{train}}$) and applied without refitting to test data.
2. **One-Hot Encoding**: `OneHotEncoder(drop='first', handle_unknown='ignore')` was fit on training categories, ensuring unseen test categories are encoded cleanly without throwing dimensional errors.
3. **Temporal Audit**: A time-based split was audited and rejected because non-charting catalog tracks lack verified release timestamps; attempting a chronological split would have introduced temporal selection bias.

---

## 6. Exploratory Data Analysis (EDA)

Exploratory data analysis uncovered several distinct distributional properties:

### Target Feature Correlations
Pearson correlation coefficients against binary chart success (`charted`):

| Feature | Pearson $r$ | $p$-value | Empirical Finding |
|:---|:---:|:---:|:---|
| **`loudness`** | **+0.168** | $7.08 \times 10^{-48}$ | Significant positive correlation; commercial mastering favors loudness |
| **`instrumentalness`** | **-0.164** | $9.92 \times 10^{-46}$ | Strong negative correlation; chart hits require prominent vocals |
| **`acousticness`** | **-0.139** | $3.08 \times 10^{-33}$ | Strong negative correlation; acoustic tracks rarely penetrate modern Hot 100 |
| **`danceability`** | **+0.134** | $3.71 \times 10^{-31}$ | Significant positive correlation; rhythmic movement supports radio play |
| **`energy`** | +0.062 | $8.13 \times 10^{-8}$ | Moderate positive correlation with commercial intensity |
| **`valence`** | +0.048 | $3.85 \times 10^{-5}$ | Mild positive correlation; minor trend toward musical positivity |
| **`duration_sec`** | -0.035 | $2.77 \times 10^{-3}$ | Mild negative correlation; radio edits penalize extended durations |
| **`speechiness`** | -0.016 | 0.178 | Statistically non-significant linear relationship |
| **`tempo`** | -0.015 | 0.191 | Statistically non-significant linear relationship |

![Correlation Heatmap](figures/correlation_heatmap.png)

### Key Acoustic Patterns
- **Loudness & Vocals**: Charted tracks cluster tightly at higher loudness (median $-5.3\text{ dB}$ vs $-7.2\text{ dB}$ for non-charted) and zero instrumentalness (median $0.00$ vs $0.00$ with lower upper quartile).
- **Tempo Invariance**: Billboard success occurs across all tempo ranges (median $120.0\text{ BPM}$ vs $122.0\text{ BPM}$), indicating tempo alone does not dictate hit potential.

![Danceability vs Energy](figures/danceability_vs_energy.png)

---

## 7. Models & Hyperparameter Tuning

Eight distinct classifier architectures spanning linear, non-linear, ensemble, and instance-based methods were trained. Hyperparameters were tuned using an inner **5-fold StratifiedGroupKFold cross-validation** on the training data using ROC-AUC as the optimization objective:

| Model | Tuned Hyperparameters (`best_params.csv`) | Best CV ROC-AUC | CV Std Dev | Training Time |
|:---|:---|:---:|:---:|:---:|
| **Logistic Regression** | `{'model__C': 1.0}`, `class_weight='balanced'` | **0.9469** | $\pm 0.0086$ | 6.10s |
| **SVM (RBF Kernel)** | `{'model__C': 10.0, 'model__gamma': 'auto'}` | 0.9463 | $\pm 0.0080$ | 72.12s |
| **Linear SVM** | `{'model__C': 1.0}`, `class_weight='balanced'` | 0.9447 | $\pm 0.0085$ | 36.64s |
| **Random Forest** | `{'model__max_depth': None, 'model__max_features': 'log2'}` | 0.9424 | $\pm 0.0099$ | 46.67s |
| **K-Nearest Neighbors** | `{'model__n_neighbors': 15}` | 0.8950 | $\pm 0.0133$ | 10.03s |
| **Gaussian Naive Bayes** | `{'model__var_smoothing': 1e-07}` | 0.8933 | $\pm 0.0116$ | 1.59s |
| **Decision Tree (Gini)** | `{'model__max_depth': 12, 'model__min_samples_leaf': 20}` | 0.8585 | $\pm 0.0271$ | 2.77s |
| **Decision Tree (Entropy)** | `{'model__max_depth': 8, 'model__min_samples_leaf': 20}` | 0.8503 | $\pm 0.0266$ | 2.86s |

---

## 8. Held-Out Test Evaluation

All models were evaluated on the held-out test set (1,480 tracks; 135 charted, 1,345 non-charted; 0 artist overlap). Specificity was computed explicitly via the formula:

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

### Complete Benchmark Comparison (`reports/tables/model_comparison.csv`)

| Model | ROC-AUC | PR-AUC | Recall | Specificity | Precision | Accuracy | F1-Score | TP | TN | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.929** | 0.606 | **0.844** | 0.862 | 0.380 | 0.860 | **0.524** | 114 | 1159 | 186 | 21 |
| **SVM (RBF Kernel)** | 0.924 | 0.595 | 0.770 | 0.865 | 0.364 | 0.856 | 0.494 | 104 | 1163 | 182 | 31 |
| **Linear SVM** | 0.924 | 0.602 | 0.793 | 0.852 | 0.350 | 0.847 | 0.485 | 107 | 1146 | 199 | 28 |
| **Random Forest** | 0.918 | **0.619** | 0.296 | **0.993** | **0.816** | **0.930** | 0.435 | 40 | 1336 | 9 | 95 |
| **K-Nearest Neighbors** | 0.877 | 0.509 | 0.370 | 0.987 | 0.735 | 0.930 | 0.493 | 50 | 1327 | 18 | 85 |
| **Gaussian Naive Bayes** | 0.875 | 0.305 | 0.911 | 0.787 | 0.300 | 0.798 | 0.451 | 123 | 1058 | 287 | 12 |
| **Decision Tree (Entropy)**| 0.821 | 0.443 | 0.341 | 0.989 | 0.754 | 0.930 | 0.469 | 46 | 1330 | 15 | 89 |
| **Decision Tree (Gini)** | 0.821 | 0.513 | 0.363 | 0.982 | 0.671 | 0.926 | 0.471 | 49 | 1321 | 24 | 86 |

![ROC Curves](figures/roc_curves.png)
![Precision-Recall Curves](figures/pr_curves.png)
![Confusion Matrices Grid](figures/confusion_matrices.png)

---

## 9. Model Comparison & Selection Rationale

### Multi-Criteria Trade-Off Analysis
1. **Accuracy is Deceptive**: Random Forest, KNN, and Decision Trees achieve 92.6%–93.0% accuracy at default 0.5 cutoffs. However, a dummy model predicting "never charts" achieves **90.9% accuracy** on this test set. Random Forest's high accuracy masks an unacceptably low recall of **29.6%** (missing 95 of 135 true hits).
2. **PR-AUC vs. Recall**: Random Forest achieves the top PR-AUC (**0.619** vs Logistic Regression's **0.606**), showing superior precision-recall trade-offs at high-confidence thresholds. However, in an initial A&R screening funnel, hit capture (Recall) is the paramount operational constraint.
3. **Interpretability & Complexity**: Random Forest is a 300-tree ensemble with hundreds of split boundaries. Logistic Regression provides transparent, standardized log-odds coefficients directly explainable to non-technical stakeholders.
4. **Final Selection**: Regularized **Logistic Regression** (`C=1.0`, `class_weight='balanced'`) was selected for production deployment. It achieves the highest test ROC-AUC (**0.929**), robust test PR-AUC (**0.606**), sub-millisecond inference latency, and captures the majority of hits under operational thresholding.

---

## 10. Explainability & Hypothesis Testing

### Feature Importance Comparison
To prevent the known bias of impurity-based importance toward high-cardinality features, Random Forest Gini impurity importances were contrasted with held-out test permutation importances ($n=10$ repeats, `random_state=42`):

| Feature | Gini Impurity Importance | Test Permutation Importance (ROC-AUC Drop) | Permutation Std Dev |
|:---|:---:|:---:|:---:|
| **`genre`** | **0.312** | **0.1345** | $\pm 0.0100$ |
| **`loudness`** | 0.098 | **0.0267** | $\pm 0.0069$ |
| **`instrumentalness`** | 0.088 | **0.0238** | $\pm 0.0085$ |
| **`duration_sec`** | 0.081 | 0.0151 | $\pm 0.0026$ |
| **`acousticness`** | 0.071 | 0.0147 | $\pm 0.0059$ |
| **`danceability`** | 0.053 | 0.0052 | $\pm 0.0052$ |
| **`energy`** | 0.051 | 0.0019 | $\pm 0.0038$ |
| **`valence`** | 0.046 | 0.0012 | $\pm 0.0030$ |

![Feature Importance Comparison](figures/feature_importance_rf.png)

### Standardized Logistic Regression Coefficients
Because numeric features were standardized ($\mu=0, \sigma=1$), logistic regression coefficients directly communicate changes in log-odds associated with a 1-standard-deviation increase in each feature:

- **`instrumentalness` ($\beta = -1.868$)**: The single largest negative audio association; heavy instrumentation without vocals dramatically reduces chart odds.
- **`loudness` ($\beta = +1.349$)**: The strongest positive audio association; tracks mastered to commercial volume standards exhibit significantly higher log-odds of charting.
- **`duration_sec` ($\beta = -0.803$)**: Radio programming formats penalize excessive track lengths.
- **`danceability` ($\beta = +0.552$)**: Upbeat, rhythmic track structures positively support commercial adoption.
- **`acousticness` ($\beta = -0.525$)**: Organic acoustic profiles are associated with lower chart probability.
- **`energy` ($\beta = -0.475$)** & **`valence` ($\beta = -0.384$)**: Once loudness is controlled for, raw energy and musical happiness show slight negative associations.

![Logistic Regression Coefficients](figures/logreg_coefficients.png)

---

## 11. Leakage-Safe Threshold Analysis

### Methodology
Evaluating multiple thresholds directly on the test set causes test-set overfitting. Therefore, candidate thresholds were derived strictly from **5-fold artist-grouped out-of-fold (OOF) training predictions** and frozen before evaluating once on the held-out test set:

1. **Youden's J Statistic**: $J = \text{Sensitivity} + \text{Specificity} - 1$. Maximized at threshold **`0.48`** on training OOF predictions.
2. **F1-Score Maximization**: Maximized at threshold **`0.77`** on training OOF predictions.

![Training OOF Threshold Trade-Off](figures/threshold_tradeoff_cv.png)

### Final Frozen Test Evaluation (`reports/tables/final_threshold_evaluation.csv`)

| Threshold Method | Threshold | Accuracy | Precision | Recall | Specificity | F1-Score | TP | TN | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Youden-J Selected (Deployment)** | **0.48** | 0.8568 | 0.3746 | **0.8519** | 0.8572 | 0.5204 | **115** | 1,153 | 192 | **20** |
| **F1-Selected** | **0.77** | 0.9074 | 0.4946 | 0.6741 | 0.9309 | 0.5705 | 91 | 1,252 | 93 | 44 |

### Commercial A&R Decision Framing
- At threshold **`0.48`**, the system captures **115 of 135 true chart hits (85.2% recall)** while successfully filtering out **1,153 of 1,345 non-charting tracks (85.7% specificity)**.
- Raising the threshold to **`0.77`** increases precision from 37.5% to 49.5% and reduces false positives from 192 to 93, but incurs **24 additional false negatives (missing 44 hits vs. 20)**.
- For commercial scouting, missing a breakthrough hit carries an immense opportunity cost; threshold **0.48** represents the superior operational operating point.

---

## 12. Unsupervised Audio Clustering (Bonus Analysis)

To determine whether tracks naturally cluster by audio properties alone, K-Means clustering was executed on the 9 standardized continuous audio features ($k=2 \dots 10$):

- **Optimal Cluster Selection**: $k=4$ was selected based on the inertia elbow curve leveling ($42,067.74$) and peak multi-cluster silhouette score ($0.1794$).
- **2D PCA Projection**: Component 1 explains 30.9% of variance; Component 2 explains 17.8% (48.7% total explained variance).

![Cluster Selection Curves](figures/cluster_selection.png)
![PCA 2D Cluster Projection](figures/clusters_pca.png)

### Cluster Profiles (`reports/tables/cluster_profiles.csv`)

| Cluster | Descriptive Name | Size (% Total) | Charted Rate | Dominant Genres (Counts) |
|:---:|:---|:---:|:---:|:---|
| **0** | **Upbeat Danceable / Pop** | 3,348 (45.2%) | **13.92%** | dance (107), country (77), salsa (76), pagode (72) |
| **1** | **Acoustic / Downtempo / Ambient** | 1,492 (20.2%) | **1.94%** | romance (69), sleep (63), new-age (62), tango (60) |
| **2** | **Intense / Fast / Rock & Metal** | 2,315 (31.3%) | **6.83%** | grindcore (96), black-metal (79), hardstyle (75), heavy-metal (65) |
| **3** | **High-Speechiness / Spoken Word** | 248 (3.3%) | **8.06%** | comedy (76), j-dance (15), hardcore (12), french (10) |

**Key Finding**: The unsupervised clusters **cut sharply across commercial genre boundaries**. Upbeat danceable tracks from country, pop, and latin cluster together in Cluster 0 (which contains 466 of the 673 total hits), demonstrating that acoustic production parameters unite commercial tracks across disparate cultural categories.

---

## 13. Prediction System Architecture

### FastAPI Backend (`app/main.py`)
- **Startup Lifecycle**: Model pipeline (`models/final/model.joblib`) and metadata schema (`models/final/metadata.json`) load once at startup into `app.state`.
- **CORS & Security**: Pydantic input models bound all numeric fields to real training boundaries. Out-of-range inputs return informative 422 JSON errors naming the offending parameter.
- **Explainability API**: Computes local feature contributions ($\text{effect}_i = x_{\text{standardized}, i} \times \beta_i$) returned dynamically with every prediction.

### Manuscript Web Client (`web/`)
Built with plain HTML, CSS, and Vanilla JavaScript conforming to `design.md`:
- Left-aligned manuscript paper aesthetic with Fraunces serif typography and 5-line staff dividers.
- Dynamic sliders and presets ("Pop single", "Acoustic ballad", "Ambient instrumental").
- Horizontal meter with brass threshold indicator at 0.48 and diverging contribution bar charts.
- Accessible keyboard navigation (full sequential Tab order and visible 2px indigo focus rings).

---

## 14. Conclusion & Limitations

### Summary of Achievements
1. Built an artist-leakage-free classification pipeline achieving **0.929 test ROC-AUC** and **0.606 PR-AUC** on completely unseen artists.
2. Formulated a leakage-safe decision threshold of **0.48** capturing **85.2%** of historical chart hits.
3. Proved that commercial chart success is strongly associated with high loudness, prominent vocal presence (low instrumentalness), and high danceability.
4. Delivered a fully tested, accessible, and explainable software deployment.

### System Boundaries & What the Model Cannot Do
- **Not a Virality Predictor**: The model evaluates acoustic feature viability, not marketing budget, TikTok virality, playlist placement, or radio plugger campaigns.
- **Not Causal Advice**: Positive coefficients reflect empirical statistical associations in historical data, not causal guarantees that artificially boosting loudness will produce a hit.
- **Dynamic Cultural Shifts**: The model reflects commercial music patterns between 2000 and 2023; emergent subgenres and production shifts require periodic model retraining.
