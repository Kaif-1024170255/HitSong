# Hit Song Predictor — Viva & Oral Defense Preparation Guide

This guide contains **15 rigorous technical questions** an academic examiner is likely to ask regarding this project. Every answer provides the precise, mathematically sound justification and is strictly grounded in the verified metrics and empirical results recorded in `reports/tables/`.

---

### Question 1: Why is raw classification accuracy a misleading evaluation metric for this project?

**Examiner Intent:** Testing understanding of class imbalance, base rate fallacy, and metric alignment with domain objectives.

**Exact Defense Answer:**
> "Raw classification accuracy is fundamentally misleading here due to severe class imbalance. In our candidate pool of 80,237 tracks, genuine hits comprise only 0.84% (673 tracks). In our curated 10:1 modeling dataset (7,403 tracks), the hit rate is 9.09% (673 hits vs 6,730 non-hits).
>
> On our held-out test set of 1,480 tracks (135 hits, 1,345 non-hits), a trivial majority-class dummy classifier that predicts *'Never a Hit'* for every song achieves an accuracy of **90.88%** (1,345/1,480), yet has a recall of **0.0%**, missing every single hit.
>
> Furthermore, compare our tuned Random Forest and Logistic Regression models in `reports/tables/model_comparison.csv`:
> - Random Forest achieved **93.0% accuracy**, but caught only 40 hits and missed 95 (Recall = **29.6%**, F1 = **0.435**).
> - Logistic Regression achieved **86.0% accuracy**, but caught 114 hits and missed only 21 (Recall = **84.4%**, F1 = **0.524**).
>
> Random Forest achieved higher accuracy simply by predicting the majority class 90% of the time. In commercial music scouting, missing a hit is an existential opportunity cost, whereas reviewing a false positive costs minutes of listening time. Therefore, we evaluated models using ROC-AUC (0.929 for LogReg), PR-AUC (0.606), and Recall, rather than accuracy."

---

### Question 2: Why was `class_weight='balanced'` applied, and how does it alter the optimization objective?

**Examiner Intent:** Testing understanding of cost-sensitive learning versus resampling, and mathematical implementation in loss functions.

**Exact Defense Answer:**
> "We applied `class_weight='balanced'` to prevent majority-class gradient dominance without having to generate synthetic minority samples via SMOTE, which often creates unrealistic acoustic feature combinations.
>
> Mathematically, scikit-learn adjusts the loss penalty inversely proportional to class frequencies:
> $$w_j = \frac{N}{2 \cdot N_j}$$
> For our training set ($N = 5,923$, $N_0 = 5,385$, $N_1 = 538$):
> - Non-hit weight $w_0 = \frac{5923}{2 \times 5385} \approx 0.550$
> - Hit weight $w_1 = \frac{5923}{2 \times 538} \approx 5.505$
>
> This creates an exact **10:1 penalty ratio** in the cross-entropy loss function:
> $$\mathcal{L}(\theta) = -\sum_{i=1}^{N} \left[ w_1 y_i \log(\hat{p}_i) + w_0 (1 - y_i) \log(1 - \hat{p}_i) \right]$$
> Whenever the model misclassifies a true hit as a non-hit, the optimization gradient penalizes the model 10 times more heavily than misclassifying a non-hit. This shifted the decision boundary to achieve 84.4% test recall on Logistic Regression compared to unweighted models which collapsed toward low-recall majority predictions."

---

### Question 3: What data leakage was identified and eliminated, and how did you diagnose it?

**Examiner Intent:** Probing data hygiene, leakage vectors, and knowledge of grouping structures in tabular/audio data.

**Exact Defense Answer:**
> "The primary leakage risk identified was **artist-level identity leakage**. Multiple songs in our dataset are performed by the same prolific artists (e.g., Drake, Taylor Swift, The Beatles). If songs by the same artist appear in both the training and test splits, classifiers can memorize artist-specific acoustic production signatures (such as specific vocal equalization, stem compression, or mastering loudness) rather than learning universal acoustic patterns of hit songs.
>
> We diagnosed this by inspecting the artist distribution across our 7,403 songs and finding 4,581 unique artists with multi-song clusters.
>
> To eliminate this leakage completely, we used `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on `primary_artist`. This guaranteed:
> 1. **Zero artist overlap**: Exactly 0 artists appear in both the 5,923-track train set (3,657 artists) and the 1,480-track test set (924 artists).
> 2. **Identical class balance**: Train hit rate is 9.08% (538/5,923) and test hit rate is 9.12% (135/1,480), a difference of only 0.04 percentage points.
>
> Transformers (such as `StandardScaler` and `OneHotEncoder`) were also fitted strictly on training folds and transformed on validation/test folds inside scikit-learn `Pipeline` objects, preventing distribution leakage."

---

### Question 4: Why did the Decision Tree splitting criterion (Gini Impurity vs Entropy / Information Gain) barely change the results?

**Examiner Intent:** Assessing theoretical understanding of decision tree splitting mathematics and empirical behavior.

**Exact Defense Answer:**
> "In our 5-fold cross-validation and held-out test evaluation, Gini Impurity and Entropy produced virtually identical results:
> - CV ROC-AUC: **0.8585 ± 0.0271** (Gini) vs **0.8503 ± 0.0266** (Entropy)
> - Held-out Test ROC-AUC: **0.821** for both
> - Held-out Test Accuracy: **0.926** (Gini) vs **0.930** (Entropy)
> - Held-out Test F1-Score: **0.471** (Gini) vs **0.469** (Entropy)
>
> This occurs because Gini Impurity $I_G(p) = 1 - \sum p_i^2 = 2p(1-p)$ and Entropy $H(p) = -\sum p_i \log_2(p_i)$ are mathematically monotonic functions of each other across virtually the entire probability range $[0, 1]$.
>
> For binary classification, both functions are symmetric concave curves peaking at $p = 0.5$ ($I_G = 0.5$, $H = 1.0$) and reaching 0 at $p \in \{0, 1\}$. In practice, they select the exact same feature and split threshold on over 95% of node evaluations. The tree structure was constrained primarily by regularization hyperparameters (`max_depth=8-12`, `min_samples_leaf=20`), which dominated tree geometry and rendered the choice of impurity metric statistically indistinguishable."

---

### Question 5: Why did K-Nearest Neighbors (KNN) strictly require feature scaling, while Tree-based models did not?

**Examiner Intent:** Understanding distance-based geometry versus axis-aligned orthogonal partitioning.

**Exact Defense Answer:**
> "KNN computes metric proximity using Euclidean distance:
> $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{k=1}^{D} (u_k - v_k)^2}$$
> In our unscaled audio feature space, `duration_sec` ranges from 30 to 600 seconds ($\sigma^2 \approx 10^3$), and `loudness` ranges from -35 dB to 0 dB ($\sigma^2 \approx 25$). Meanwhile, features like `danceability`, `speechiness`, and `acousticness` are bounded strictly in $[0.0, 1.0]$ ($\sigma^2 \approx 0.05$).
>
> Without `StandardScaler`, a 10-second difference in song duration contributes $(10)^2 = 100$ to the squared distance, completely overpowering a massive 0.8 difference in acousticness ($(0.8)^2 = 0.64$) by a factor of 150. KNN would effectively degenerate into a 1-dimensional search over song length and loudness. `StandardScaler` standardizes all features to $\mu=0, \sigma=1$, giving each acoustic dimension equal geometric influence.
>
> In contrast, Decision Trees and Random Forests use rank-order, axis-aligned splits ($x_k \le \theta$). Monotonic transformations like z-score scaling do not alter the ordering of feature values, leaving tree splits and loss reductions completely invariant."

---

### Question 6: What does your threshold analysis mean commercially for an A&R (Artists and Repertoire) department?

**Examiner Intent:** Testing whether you can translate mathematical probability thresholds into real-world business economics.

**Exact Defense Answer:**
> "The default classification threshold of 0.5 is an arbitrary mathematical artifact. In our project, threshold tuning navigates an explicit commercial tradeoff between talent discovery and screening bandwidth (`reports/tables/final_threshold_evaluation.csv`):
>
> 1. **High-Recall Screening Funnel ($t = 0.48$, Youden's $J = 0.709$):**
>    - Captures **85.19% of hits** (115 of 135 hits caught; only 20 missed).
>    - Precision is **37.46%** (192 false positives).
>    - *Commercial Meaning:* An A&R team screens 307 flagged candidates ($115 + 192$) to discover 115 charting hits. That is **1 hit discovered per 2.67 tracks reviewed**, compared to the blind baseline rate of 1 hit per 11 tracks. Because signing a hit yields millions in streaming royalties while listening to an extra candidate takes 3 minutes, this is the optimal deployment threshold.
>
> 2. **Precision-Targeted Signing Funnel ($t = 0.77$, Max F1 = 0.570):**
>    - Precision increases to **49.46%** (only 93 false alarms).
>    - Recall drops to **67.41%** (misses 44 genuine hits).
>    - *Commercial Meaning:* Useful only when signing capital or studio time is severely rationed and the label cannot afford unvalidated studio sessions."

---

### Question 7: What are the fundamental limitations of this model? What can it NOT do?

**Examiner Intent:** Checking intellectual honesty, boundary awareness, and avoidance of overclaiming.

**Exact Defense Answer:**
> "Our model is strictly an **acoustic viability classifier**; it cannot and does not predict total commercial outcome. Specifically:
> 1. **No Non-Acoustic Variables:** The model has zero knowledge of marketing spend, TikTok virality, artist social media reach, playlist placement (e.g., Spotify's *Today's Top Hits*), or label distribution power. A song with ideal audio features will fail without marketing.
> 2. **No Causal Guarantees:** Logistic regression coefficients reflect observational associations, not causal interventions. Compressing dynamic range to make an ambient piano ballad +5 dB louder will ruin the track rather than guarantee a Billboard charting.
> 3. **Market Specificity:** Labels are derived from the US Billboard Hot 100. It cannot be applied directly to regional streaming markets (e.g., UK Official Charts, Latin America, or South Asia) where acoustic preferences differ.
> 4. **Temporal Drift:** Production aesthetics evolve over time (e.g., shorter track lengths in the streaming era). The model represents an aggregate representation from 1960–2023."

---

### Question 8: Why did you choose Logistic Regression as the final production model over Random Forest, even though Random Forest had a slightly higher PR-AUC (0.619 vs 0.606)?

**Examiner Intent:** Testing model selection criteria beyond single metrics — interpretability, calibration, and error distributions.

**Exact Defense Answer:**
> "We selected Logistic Regression over Random Forest based on three clear engineering and commercial justifications:
> 1. **Catastrophic Recall Deficit in Random Forest:** At default operating settings, Random Forest missed **95 out of 135 hits** on the test set (Recall = **29.6%**, F1 = **0.435**). In contrast, Logistic Regression captured **114 out of 135 hits** (Recall = **84.4%**, F1 = **0.524**). For an A&R screening tool, capturing 84.4% of commercial opportunities is superior to capturing 29.6%.
> 2. **Higher Global Discrimination (ROC-AUC):** Logistic Regression achieved a higher held-out test ROC-AUC (**0.929**) than Random Forest (**0.918**), and a higher 5-fold CV ROC-AUC (**0.9469 ± 0.0086** vs **0.9424 ± 0.0099**).
> 3. **Governance and Explainability:** Logistic Regression is fully glass-box. Its standardized log-odds coefficients (e.g., instrumentalness = -1.868, loudness = +1.349, duration = -0.803) allow real-time feature attribution in our FastAPI endpoint and plain-language defensibility to creative music executives, whereas Random Forest is a 100-tree ensemble with complex non-linear interactions."

---

### Question 9: Why did you perform 5-fold StratifiedGroupKFold on the training data instead of standard KFold or StratifiedKFold?

**Examiner Intent:** Understanding group structure, cross-validation variance, and statistical validation integrity.

**Exact Defense Answer:**
> "Standard K-Fold randomly assigns rows to folds, ignoring both class proportions and grouped structures. In our dataset, this would lead to two fatal flaws:
> 1. **Artist Contamination:** Songs by the same artist would be split between training and validation folds, leaking artist identities and generating artificially inflated cross-validation scores.
> 2. **Class Imbalance Instability:** Standard GroupKFold groups by artist but ignores class labels, resulting in folds with wildly fluctuating hit counts (e.g., some folds having 5% hits and others 15%).
>
> `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` solves both problems simultaneously: it assigns all songs by a given artist to a single fold (guaranteeing 0 artist leakage across folds) while utilizing a constraint satisfaction algorithm to ensure that every fold maintains the ~9.08% hit rate. This ensured our hyperparameter selection was robust and fully generalizable to unseen artists."

---

### Question 10: Why was a temporal train/test split rejected in favor of the artist-grouped split?

**Examiner Intent:** Investigating temporal data considerations and recognizing practical data limitations.

**Exact Defense Answer:**
> "While music trends evolve over time, a temporal split was methodologically invalid on this dataset due to **asymmetric temporal metadata**:
> 1. The positive class (Billboard Hot 100) contains verified charting dates from 1960 to 2023 (`chart_year`).
> 2. The negative class (Spotify 1.2M pool) contains over 65% missing release years, or contains reissue/remaster dates (e.g., a 1971 Led Zeppelin track tagged as 2014 remastered).
>
> Splitting on year would have concentrated clean recent tracks in the test set while leaving corrupted historical records in the training set, creating severe temporal selection bias between positive and negative classes.
>
> Furthermore, the central commercial question in music A&R is: *'If an unsigned artist submits an audio demo, can audio features predict if they will chart?'* The artist-grouped split directly tests generalization to unseen artists without confounding temporal artifacts."

---

### Question 11: Why did you downsample the negative pool to a 10:1 ratio instead of training on the full 80,237 tracks or using a 1:1 balance?

**Examiner Intent:** Testing understanding of sample efficiency, computational complexity, and calibration realism.

**Exact Defense Answer:**
> "The negative pool was downsampled to a 10:1 ratio (6,730 non-hits to 673 hits; 7,403 total) for two reasons:
> 1. **Computational tractability without information loss:** In the raw pool of 80,237 tracks, the hit ratio is 0.84% (118:1). Fitting models like Support Vector Machines with an RBF kernel scales quadratically with sample size ($O(N^2)$ to $O(N^3)$). Training SVM on 80,000 samples requires over 6.4 billion kernel computations, taking hours, whereas 7,403 samples trains in seconds while retaining identical decision boundaries when paired with `class_weight='balanced'`.
> 2. **Avoiding the distortion of 1:1 undersampling:** Artificially balancing to 1:1 (673 hits vs 673 non-hits) would discard 99% of non-hit acoustic diversity and severely distort predicted probability calibration, causing the model to drastically over-predict hit likelihood in production where hits are naturally rare. The 10:1 ratio preserves vast negative acoustic diversity while remaining computationally efficient."

---

### Question 12: Why did Random Forest's Mean Decrease in Impurity (Gini importance) disagree with Permutation Feature Importance?

**Examiner Intent:** Probing understanding of feature importance biases in ensemble tree models.

**Exact Defense Answer:**
> "Mean Decrease in Impurity (MDI / Gini importance) is computed during tree construction on training data. It is well-documented to suffer from a severe structural bias toward **continuous features with high cardinality** (like `loudness`, `tempo`, and `duration_sec`). Because continuous features have thousands of unique values, they offer many more candidate split thresholds that can arbitrarily reduce training node impurity, even if the split captures pure noise.
>
> In contrast, **Permutation Feature Importance** measures the drop in model performance (ROC-AUC) on the unseen test set when a feature column is randomly shuffled.
>
> Permutation importance revealed that one-hot encoded `genre` was by far the single most predictive feature (test ROC-AUC drop of **0.1345**), followed by `loudness` (**0.0267**) and `instrumentalness` (**0.0238**). MDI ranked continuous audio features artificially high because of split-point cardinality bias, whereas permutation importance accurately reflected out-of-sample predictive power."

---

### Question 13: How was your final threshold selection procedure kept strictly leakage-safe?

**Examiner Intent:** Examining validation hygiene, avoiding test-set optimization, and adherence to clean ML workflows.

**Exact Defense Answer:**
> "A common methodological error is tuning the classification threshold directly on the held-out test set, which leaks test labels into model decision rules and produces over-optimistic test metrics.
>
> We enforced a strictly leakage-safe protocol:
> 1. **Out-of-Fold (OOF) Training Predictions:** We generated out-of-fold probability predictions exclusively on the training set (5,923 songs) using the 5 artist-grouped cross-validation folds.
> 2. **Threshold Optimization on OOF Only:** The optimal Youden's $J$ threshold ($t = 0.48$) and max-$F_1$ threshold ($t = 0.77$) were computed solely from these OOF training probabilities.
> 3. **Threshold Freezing:** Both thresholds were frozen and serialized into `models/final/metadata.json`.
> 4. **Single-Pass Test Evaluation:** The held-out test set (1,480 songs) was evaluated exactly once using the frozen threshold of 0.48, yielding:
>    - Accuracy = **0.8568**
>    - Precision = **0.3746**
>    - Recall = **0.8519**
>    - Specificity = **0.8572**
>    - F1 = **0.5204**
>
> Test Youden's $J$ was $0.8519 + 0.8572 - 1 = 0.7091$, closely tracking the training OOF $J$ of $0.7483$, confirming that our threshold did not overfit."

---

### Question 14: What did unsupervised K-Means clustering reveal about commercial genres versus acoustic features?

**Examiner Intent:** Testing insights from unsupervised learning, dimensionality reduction, and domain synthesis.

**Exact Defense Answer:**
> "Using K-Means on the 9 standardized continuous audio features (excluding genre and chart status), elbow and silhouette analysis identified $k = 4$ clusters as optimal (silhouette score **0.1794**, inertia **42,067.74**).
>
> PCA projection (PC1 explaining **30.9%** variance, PC2 **17.8%**, totaling **48.7%**) revealed four distinct acoustic archetypes (`reports/tables/cluster_profiles.csv`):
> - **Cluster 0 ('Upbeat Danceable / Pop'):** High danceability (0.697), high valence (0.648), high loudness (-6.4 dB). Chart hit rate = **13.92%**.
> - **Cluster 1 ('Acoustic / Downtempo'):** High acousticness (0.751), low energy (0.281). Chart hit rate = **1.94%**.
> - **Cluster 2 ('Intense Rock & Metal'):** High energy (0.835), low acousticness (0.076), moderate danceability. Chart hit rate = **6.83%**.
> - **Cluster 3 ('Spoken Word & Rap'):** Extremely high speechiness (0.804), low musical energy. Chart hit rate = **8.06%**.
>
> Crucially, these clusters **cut directly across official genre boundaries**. Pop songs appeared in both Cluster 0 (dance pop) and Cluster 1 (acoustic piano ballads); Rock appeared in both Cluster 2 (hard rock) and Cluster 1 (acoustic folk rock). This proves that commercial genre tags are marketing labels, whereas audio features define the true underlying physical acoustic profile."

---

### Question 15: How does your deployment system ensure low latency, input safety, and end-to-end explainability?

**Examiner Intent:** Assessing software engineering rigor, production architecture, API validation, and explainable AI integration.

**Exact Defense Answer:**
> "Our deployment system is built with FastAPI and vanilla HTML/CSS/JavaScript with three core architectural safeguards:
> 1. **Dynamic Schema & Single Source of Truth:** The frontend contains zero hardcoded feature ranges or defaults. On initialization, it calls `GET /api/schema`, which reads min, max, and step parameters directly from `models/final/metadata.json`.
> 2. **Strict Input Validation & HTTP 422 Handling:** On `POST /api/predict`, Pydantic models validate that every audio feature lies strictly within empirical domain bounds (e.g., `loudness` between -60 and 5 dB, `danceability` between 0.0 and 1.0). Out-of-bounds inputs return structured HTTP 422 errors identifying the exact field and allowed range.
> 3. **Real-Time Local Feature Attribution in Under 5ms:** Because Logistic Regression is linear in the log-odds space:
>    $$\log \left( \frac{p}{1-p} \right) = \beta_0 + \sum_{i=1}^{D} \beta_i \left( \frac{x_i - \mu_i}{\sigma_i} \right)$$
>    For every prediction, the API computes the exact signed local contribution $c_i = \beta_i \cdot z_i$ for each feature. The frontend displays these effects directly to the user (e.g. *'+0.42 log-odds from Loudness'*, *'-0.61 log-odds from Instrumentalness'*), providing instant, auditable explanations with sub-5ms inference latency without needing slow approximations like KernelSHAP."

---
*All metrics, splits, hyperparameters, and tables cited in this document are verified against `reports/tables/` and generated under `random_state=42`.*
