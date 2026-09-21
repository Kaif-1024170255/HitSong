# Model Explainability and Hypothesis Verification

## 1. Random Forest Feature Importance: Impurity vs Permutation
- **A. Observed Results**:
  - `genre`: +0.1345 ± 0.0100
  - `loudness`: +0.0267 ± 0.0069
  - `instrumentalness`: +0.0238 ± 0.0085
  - `duration_sec`: +0.0151 ± 0.0026
  - `acousticness`: +0.0147 ± 0.0059
  - `danceability`: +0.0052 ± 0.0052
  - `energy`: +0.0019 ± 0.0038
  - `valence`: +0.0012 ± 0.0030
- **B. Interpretation**:
  - Permutation importance measures the change in held-out test ROC-AUC when a feature's information is disrupted. It therefore provides an out-of-sample measure of that feature's usefulness to this particular model and dataset.
  - In contrast, tree-based Gini impurity importance can favor continuous or high-cardinality variables because such variables provide more potential split points.
  - The two importance measures answer different questions and should not be interpreted as equivalent quantities or as proof of causal influence.
- **C. Limitations**: Permutation importance is model-specific and reflects predictive association conditional on all other available features.

## 2. Logistic Regression Standardized Coefficients
- **A. Observed Results**: `loudness` (+1.349), `danceability` (+0.552), `duration_sec` (+0.201), `tempo` (+0.027), `speechiness` (+0.015), `valence` (-0.216), `acousticness` (-0.491), `energy` (-0.620), `instrumentalness` (-1.868).
- **B. Interpretation**:
  - Numeric features were standardized using StandardScaler; a coefficient represents the change in log-odds associated with approximately one standard-deviation increase in that feature, while holding other model features constant.
  - One-hot encoded genre features are interpreted relative to the omitted/reference genre category. Raw coefficient magnitudes for numeric features and one-hot genre indicators are not directly comparable.
  - Magnitudes represent conditional model associations and predictive contributions, not causal influence.
- **C. Limitations**: Logistic regression models log-odds linearly and does not capture complex feature interactions.

## 3. Top Levels of the Tuned Gini Decision Tree (`tree_gini`)
- **A. Observed Structure**: Displays the top three levels of the tuned `tree_gini` model (Gini Decision Tree), showing split features, decision thresholds, Gini impurity, sample counts, and class proportions.
- **B. Interpretation & Limitations**: The tree depicts recursive binary partitioning boundaries used by the model for classification; these splits describe model decision logic, not real-world causal mechanisms.

## 4. Empirical Hypothesis Testing

### Hypothesis 1: Valence ("Upbeat songs chart better")
- **A. Observed Data**: Charted mean valence = 0.505; non-charted mean valence = 0.462.
- **B. Statistical Result**: Welch's t = 4.850 (p = 1.46e-06); point-biserial r = 0.0478 (p = 3.85e-05).
- **C. Model-Based Evidence**: Multivariate Logistic Regression coefficient beta = -0.216; Random Forest test permutation importance = +0.0012 ± 0.0030 ROC-AUC drop.
- **D. Interpretation & Verdict**:
  - **Verdict**: Valence differs statistically between charted and non-charted tracks in the univariate analysis, but the association is weak and the multivariate model provides limited evidence that valence independently contributes strongly to chart prediction.
  - The univariate analysis detects a statistically significant difference in means, but the point-biserial correlation is small (r = 0.0478). Statistical significance (p < 0.05) does not imply a large practical effect.
  - When controlling for other features in the multivariate Logistic Regression, the coefficient becomes negative (beta = -0.216), and Random Forest permutation importance is minimal (+0.0012).
  - Therefore, valence shows a weak univariate association with limited independent predictive contribution; valence alone provides limited predictive information about chart status in this dataset.
- **E. Limitations**: Valence is an algorithmic estimate of musical mood; confounding with genre and production loudness accounts for the discrepancy between univariate significance and multivariate importance.

### Hypothesis 2: Duration and Chart Year ("Songs are getting shorter")
- **A. Observed Data**: Analyzed n = 670 charted observations with valid `chart_year` from 2000–2023. Mean duration 2000–2005 = 246.6 seconds; mean duration 2018–2023 = 197.5 seconds (difference of approximately 49 seconds).
- **B. Statistical Result**: Pearson r = -0.4390 (p = 6.11e-33); Spearman rho = -0.4621 (p = 9.38e-37); fitted linear regression slope = -2.64 seconds per chart year.
- **C. Methodological Scope**: `chart_year` refers to the year associated with the Billboard chart observation. It should not be interpreted as the song's release year. This is a within-charted-songs temporal analysis (n = 670) and does not evaluate non-charted tracks over time.
- **D. Interpretation & Verdict**:
  - **Verdict**: Among the analyzed charted tracks from 2000–2023, song duration shows a statistically significant negative association with chart year.
  - The fitted linear regression corresponds to an estimated decrease of approximately 2.64 seconds in duration per chart year.
  - The mean duration of analyzed charted tracks decreased from 246.6 seconds in 2000–2005 to 197.5 seconds in 2018–2023, a difference of approximately 49 seconds.
- **E. Limitations**: This analysis describes temporal patterns among charted observations with available chart years; it does not establish causation and should not be generalized automatically to all released songs.
