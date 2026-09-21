# Hit Song Predictor: Presentation Slide Notes

Comprehensive speaking notes, verified metrics, and figure assignments for an 8-slide academic presentation. Every statistic traces directly to artifacts in `reports/tables/` and figures in `reports/figures/`.

---

## Slide 1: Project Overview & The Hit Song Problem
- **Slide Title**: Can Audio Features Predict a Billboard Hit?
- **Assigned Figure**: `reports/figures/class_balance_modeling.png`
- **Bullet Points & Real Numbers**:
  - **The Commercial Problem**: A&R music scouting faces high search costs and severe asymmetric risk: missing a breakthrough chart hit costs millions in catalog equity, while auditing a non-charting track has negligible overhead.
  - **The Supervised Task**: Binary classification (`charted`: 0 vs 1) predicting Billboard Year-End Hot 100 placement from 10 Spotify musical features (9 continuous audio metrics + genre).
  - **The Imbalance Challenge**: The natural commercial candidate pool contains only **0.84% hits** (673 hits out of 80,237 tracks). Evaluated on a controlled 10:1 modeling dataset of **7,403 tracks** (**673 hits / 9.09%** vs **6,730 non-hits / 90.91%**).
  - **Core Engineering Constraint**: Strict artist-grouped evaluation to ensure models predict musical acoustic quality rather than memorizing superstar artist identities.
- **Presenter Speaking Script**:
  > *"Welcome everyone. Our project asks a fundamental question in music technology: do the audio engineering properties of a song predict whether it will chart on the Billboard Hot 100? In commercial A&R, this is an asymmetric decision problem. Missing a hit is catastrophic, but auditing a non-hit is cheap. The natural data is brutally imbalanced—fewer than 1% of songs chart. We constructed a rigorously controlled 10:1 dataset of 7,403 songs, where our core methodological constraint was preventing artist leakage so we test generalization to completely unseen artists."*

---

## Slide 2: Dataset Construction & Artist Leakage Prevention
- **Slide Title**: Leakage-Free Dataset & Artist-Grouped Splitting
- **Assigned Figure**: `reports/figures/correlation_heatmap.png`
- **Bullet Points & Real Numbers**:
  - **Source Filtering**: Filtered 80,237 raw tracks down to **7,403 songs** with a 10:1 ratio; excluded **11 ambiguous records** with conflicting chart metadata.
  - **Artist Grouping**: The dataset contains **4,581 unique normalized artists**. Partitioned using `StratifiedGroupKFold` (80% train / 20% test).
  - **Zero Leakage**: **3,657 training artists** and **924 test artists** with **0 artist overlap**. No artist in the test set was ever seen during training or preprocessing.
  - **Split Stratification**: Training set has **5,923 songs** (538 hits / 9.08%); test set has **1,480 songs** (135 hits / 9.12%). The train/test positive rate difference is just **0.04 percentage points**.
  - **Temporal Integrity**: Chronological splitting was audited and rejected because non-charting tracks lack verified release years; time splitting would have introduced severe selection bias.
- **Presenter Speaking Script**:
  > *"If an ML model sees Drake or Taylor Swift in both training and testing, it doesn't learn musical features—it learns artist fame. To prevent this artist leakage, we identified 4,581 unique artists and enforced an artist-grouped split. There is exactly zero artist overlap between our 5,923 training tracks and 1,480 test tracks. Our transformers were fit strictly on training folds, and we verified that the positive rate difference between splits is a mere 0.04 percentage points."*

---

## Slide 3: Exploratory Data Analysis & Acoustic Drivers of Chart Success
- **Slide Title**: Exploratory Data Analysis: What Separates Hits from Flops?
- **Assigned Figure**: `reports/figures/danceability_vs_energy.png` (or `reports/figures/feature_boxplots.png`)
- **Bullet Points & Real Numbers**:
  - **Mastering & Loudness ($r = +0.168, p = 7.08 \times 10^{-48}$)**: The strongest linear predictor; hits undergo aggressive commercial compression and mastering (median $-5.3\text{ dB}$ vs $-7.2\text{ dB}$).
  - **Vocal Dominance ($r = -0.164, p = 9.92 \times 10^{-46}$)**: High instrumentalness almost guarantees a song will not chart; commercial hits require prominent vocal presence.
  - **Acousticness Penalty ($r = -0.139, p = 3.08 \times 10^{-33}$)**: Organic, unplugged acoustic arrangements face significant commercial headwinds on the Hot 100.
  - **Rhythm & Danceability ($r = +0.134, p = 3.71 \times 10^{-31}$)**: Strong rhythmic beat structure strongly correlates with chart performance.
  - **Tempo Invariance ($r = -0.015, p = 0.191$)**: Tempo shows no statistically significant linear correlation with charting (median $120.0\text{ BPM}$ vs $122.0\text{ BPM}$); hits succeed across all tempo bands.
- **Presenter Speaking Script**:
  > *"Our exploratory analysis uncovered striking acoustic drivers. Loudness has the highest positive correlation (+0.168, p < 10^-47), driven by commercial mastering standards. Conversely, instrumentalness (-0.164) and acousticness (-0.139) show strong negative correlations—modern Billboard hits demand prominent vocals and synthetic or polished production. Interestingly, tempo has virtually zero correlation (-0.015, p = 0.19); songs chart at 80 BPM just as well as at 130 BPM."*

---

## Slide 4: Group-Aware Model Training & Inner Cross-Validation
- **Slide Title**: 8 Model Families & Group-Aware Cross-Validation
- **Assigned Figure**: `reports/figures/decision_tree.png`
- **Bullet Points & Real Numbers**:
  - **Inner CV Design**: 5-fold `StratifiedGroupKFold` on the training fold (never touching the test set) to prevent artist leakage during hyperparameter tuning.
  - **Scoring Objective**: Tuned for **ROC-AUC** rather than accuracy to prioritize discriminative ranking across class-imbalanced thresholds.
  - **Linear vs Non-Linear Performance**:
    - **Logistic Regression**: Best CV ROC-AUC **0.9469 $\pm$ 0.0086** (`C=1.0`, `class_weight='balanced'`, fit time 6.1s).
    - **SVM (RBF Kernel)**: CV ROC-AUC **0.9463 $\pm$ 0.0080** (`C=10.0, gamma='auto'`, fit time 72.1s).
    - **Linear SVM**: CV ROC-AUC **0.9447 $\pm$ 0.0085** (`C=1.0`, `class_weight='balanced'`, fit time 36.6s).
    - **Random Forest**: CV ROC-AUC **0.9424 $\pm$ 0.0099** (`max_depth=None, max_features='log2'`, fit time 46.7s).
  - **Tree Impurity Comparison**: Decision tree with Gini criterion (CV 0.8585) slightly outperformed Entropy (CV 0.8503), both constrained to `min_samples_leaf=20`.
- **Presenter Speaking Script**:
  > *"We benchmarked 8 model families using group-aware 5-fold cross-validation so that no artist ever appeared in both the training and validation fold of hyperparameter tuning. Tuning for ROC-AUC revealed that regularized linear models performed exceptionally well: Logistic Regression led cross-validation with 0.9469 ROC-AUC, closely matched by RBF SVM at 0.9463 and Random Forest at 0.9424. Training took just 6.1 seconds for Logistic Regression compared to over 72 seconds for the RBF kernel."*

---

## Slide 5: Held-Out Test Evaluation & The "Accuracy Trap"
- **Slide Title**: Held-Out Test Benchmark: Exposing the Accuracy Trap
- **Assigned Figures**: `reports/figures/roc_curves.png` & `reports/figures/confusion_matrices.png`
- **Bullet Points & Real Numbers**:
  - **The Held-Out Test Set**: 1,480 unseen songs (135 hits, 1,345 non-hits; 0 artist overlap).
  - **The "Accuracy Trap"**: A baseline model predicting "never charts" achieves **90.9% accuracy** while capturing zero hits. Random Forest achieved **93.0% accuracy** at default 0.5 cutoff, but its recall was a disastrous **29.6%** (missing 95 of 135 true hits!).
  - **Discriminative Ranking**:
    - **Logistic Regression**: Top test ROC-AUC **0.929** (84.4% recall, 86.2% specificity, 114/135 hits caught).
    - **SVM (RBF)**: Test ROC-AUC **0.924** (77.0% recall, 86.5% specificity, 104/135 hits caught).
    - **Random Forest**: Test ROC-AUC **0.918**, top PR-AUC **0.619** (high precision of 0.816 and 0.993 specificity, but low default recall).
  - **Explicit Metric Definitions**: Specificity was explicitly calculated as $\text{TN}/(\text{TN}+\text{FP})$ in all pipeline reports.
- **Presenter Speaking Script**:
  > *"When we evaluate on our 1,480 held-out test songs, we expose the classic 'accuracy trap'. Random Forest appears impressive with 93.0% accuracy, but it missed 95 out of 135 actual hits—a recall of only 29.6%! In commercial scouting, missing 70% of hits is fatal. Logistic Regression achieved the highest test ROC-AUC at 0.929, catching 114 of 135 hits (84.4% recall) with 86.2% specificity. While Random Forest achieved the top PR-AUC (0.619 vs 0.606), Logistic Regression provides far superior hit screening."*

---

## Slide 6: Model Explainability: Impurity vs. Permutation Importance
- **Slide Title**: Explainability: What Actually Drives the Predictions?
- **Assigned Figures**: `reports/figures/feature_importance_rf.png` & `reports/figures/logreg_coefficients.png`
- **Bullet Points & Real Numbers**:
  - **Impurity Bias Addressed**: Random Forest Gini impurity importance overvalues high-cardinality features. Contrasted with held-out test permutation importance ($n=10$ repeats, `random_state=42`).
  - **Top Permutation Features (Mean ROC-AUC Drop)**:
    - `genre`: **0.1345 $\pm$ 0.0100** (Primary commercial baseline)
    - `loudness`: **0.0267 $\pm$ 0.0069** (Crucial modern production factor)
    - `instrumentalness`: **0.0238 $\pm$ 0.0085** (Vocal presence requirement)
    - `duration_sec`: **0.0151 $\pm$ 0.0026** (Radio length constraints)
    - `acousticness`: **0.0147 $\pm$ 0.0059** (Acoustic texture penalty)
  - **Standardized Logistic Regression Coefficients ($\mu=0, \sigma=1$)**:
    - `instrumentalness`: **-1.868** (Dominant negative association)
    - `loudness`: **+1.349** (Dominant positive association)
    - `duration_sec`: **-0.803** (Strong radio penalty for long songs)
    - `danceability`: **+0.552** (Positive rhythmic contribution)
    - `acousticness`: **-0.525** (Negative acoustic penalty)
- **Presenter Speaking Script**:
  > *"To ensure executive explainability, we investigated why our models make their predictions. In-sample Gini importance is known to bias toward high-cardinality features, so we ran test-set permutation importance. Shuffling genre causes a 0.135 drop in ROC-AUC, followed by loudness (0.027) and instrumentalness (0.024). Looking at our standardized logistic regression coefficients, instrumentalness has a massive negative weight of -1.868, while loudness has a positive weight of +1.349. These coefficients represent associative relationships in log-odds, giving A&R teams fully transparent rationales."*

---

## Slide 7: Leakage-Safe Operational Threshold Selection
- **Slide Title**: Leakage-Safe Threshold Tuning & The A&R Commercial Trade-Off
- **Assigned Figure**: `reports/figures/threshold_tradeoff_cv.png`
- **Bullet Points & Real Numbers**:
  - **Leakage Prevention**: Tuning decision thresholds directly on the test set is methodological cheating. We derived candidate thresholds exclusively from **5-fold training out-of-fold (OOF) predictions** and froze them.
  - **Youden's J Optimization ($J = \text{Recall} + \text{Specificity} - 1$)**: Maximized on training OOF predictions at threshold **`0.48`**.
  - **F1-Score Optimization**: Maximized on training OOF predictions at threshold **`0.77`**.
  - **Frozen Test Set Evaluation (`reports/tables/final_threshold_evaluation.csv`)**:
    - **Selected Threshold 0.48**: **85.19% Recall**, **85.72% Specificity**, **37.46% Precision**, **115 Hits Caught**, **192 False Positives**, **20 Missed Hits**.
    - **F1 Threshold 0.77**: **67.41% Recall**, **93.09% Specificity**, **49.46% Precision**, **91 Hits Caught**, **93 False Positives**, **44 Missed Hits**.
  - **The A&R Bottom Line**: Moving from 0.77 to 0.48 catches **24 additional genuine chart hits** while incurring 99 extra tracks to review—an optimal trade-off for commercial scouting.
- **Presenter Speaking Script**:
  > *"A default 0.5 threshold is arbitrary. To select a deployment threshold safely, we tuned on training out-of-fold cross-validation predictions—the test set was never touched during selection. Maximizing Youden's J gave us an operational threshold of 0.48. When evaluated once on our frozen test set, this threshold achieves 85.2% recall and 85.7% specificity, catching 115 hits and missing only 20. If we had picked the F1-maximizing threshold of 0.77, we would have cut false positives from 192 to 93, but we would have missed 24 extra hits. In music discovery, catching those 24 hits is worth reviewing 99 extra tracks."*

---

## Slide 8: Deployment Architecture & Unsupervised Acoustic Clustering
- **Slide Title**: Production Deployment & Acoustic Clustering Archetypes
- **Assigned Figures**: `reports/figures/clusters_pca.png` & `reports/figures/cluster_selection.png`
- **Bullet Points & Real Numbers**:
  - **FastAPI Backend (`app/main.py`)**: Sub-millisecond inference, Pydantic range validation returning explicit 422 errors, dynamic schema loading, and local feature contribution calculation ($\text{effect}_i = x_{\text{std}, i} \times \beta_i$).
  - **Manuscript Frontend (`web/`)**: Vanilla HTML/CSS/JS conforming to `design.md`—cream/black manuscript paper theme, Fraunces typography, 250ms debounced sliders, accessible focus rings.
  - **Unsupervised K-Means Clustering ($k=4$, silhouette 0.1794, PCA 48.7% variance)**:
    - **Cluster 0 ("Upbeat Danceable / Pop", 45.2% of songs)**: **13.92% chart rate** (466 of 673 total hits). High danceability (0.67), loudness ($-6.5\text{ dB}$).
    - **Cluster 1 ("Acoustic / Downtempo", 20.2% of songs)**: **1.94% chart rate**. High acousticness (0.77), low loudness ($-14.6\text{ dB}$).
    - **Cluster 2 ("Intense Rock & Metal", 31.3% of songs)**: **6.83% chart rate**. High energy (0.80), fast tempo ($132.7\text{ BPM}$).
    - **Cluster 3 ("Spoken Word / Comedy", 3.3% of songs)**: **8.06% chart rate**. High speechiness (0.57).
  - **Key Architectural Insight**: Audio clusters **cut sharply across commercial genre boundaries**; acoustic production style unites commercial pop, country, and dance tracks into a single high-hit-rate cluster.
- **Presenter Speaking Script**:
  > *"Finally, we deployed this pipeline as a production web service. Our FastAPI backend provides sub-millisecond inference and computes real-time feature contributions, while our vanilla frontend adheres to an accessible manuscript-paper aesthetic. As an unsupervised bonus, we ran K-Means clustering on the continuous audio features. At k=4, we discovered that audio clusters cut sharply across official genre labels: Cluster 0 groups commercial pop, country, dance, and salsa tracks with high danceability and loudness, containing over 69% of all chart hits in our dataset. Thank you, and we look forward to your questions."*
