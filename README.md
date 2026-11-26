# Hit Song Predictor

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-52%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

A binary supervised machine learning system predicting whether a newly recorded track will chart on the **Billboard Hot 100** using quantitative audio features and genre classification. This project emphasizes empirical correctness, zero artist-data leakage, multi-criteria model evaluation, transparent explainability, and accessible production deployment.

---

## 1. Project Overview & Commercial Context

Commercial record labels and Artists and Repertoire (A&R) teams evaluate thousands of demo tracks each week. Traditional scouting depends heavily on subjective tastemakers and intuition. The **Hit Song Predictor** tests whether quantitative audio features (tempo, energy, danceability, loudness, valence, acousticness, instrumentalness, speechiness, duration) contain statistically robust signals separating Billboard Year-End Hot 100 hits from non-charting music.

- **Task**: Binary supervised classification (`charted`: 1 = Charted on Billboard Year-End Hot 100, 0 = Non-Charted).
- **Core Challenge**: Severe natural class imbalance (only ~0.84% of candidate tracks chart), high risk of artist leakage across splits, and asymmetric operational error costs (missing a genuine chart hit is commercially far costlier than auditing a non-charting song).
- **Academic Standard**: Zero data leakage, stratified artist grouping, strict Out-of-Fold (OOF) threshold selection, multi-criteria trade-off evaluation, and open-source audio extraction without paid cloud APIs.

---

## 2. Dataset Architecture & Leakage-Free Hygiene

The modeling dataset integrates historical Billboard Year-End Hot 100 charts (2000–2023) mapped to Spotify Web API audio features stored in `data/processed/songs.parquet`:

- **Raw Candidate Pool**: 80,237 tracks (673 hits, 79,564 non-charting tracks).
- **Controlled Modeling Ratio**: Downsampled at a 10:1 negative-to-positive ratio (`random_state=42`) to create a learnable decision boundary without distorting rank fidelity. 11 ambiguous tracks were permanently removed.
- **Total Processed Records**: **7,403 songs**
  - **Charted Hits ($y=1$)**: 673 songs (9.09%)
  - **Non-Charted ($y=0$)**: 6,730 songs (90.91%)
- **Zero-Artist-Leakage Outer Partition**:
  - Partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on normalized `primary_artist` (4,581 unique artists).
  - **Training Set (80.01%)**: 5,923 songs (538 charted, 5,385 non-charted; 3,657 unique artists).
  - **Held-Out Test Set (19.99%)**: 1,480 songs (135 charted, 1,345 non-charted; 924 unique artists).
  - **Artist Overlap between Train and Test**: **EXACTLY ZERO**. Test metrics represent true generalization to unseen talent.

### Predictive vs. Quarantined Schema
- **9 Continuous Audio Features**: `tempo`, `energy`, `danceability`, `valence`, `loudness`, `acousticness`, `instrumentalness`, `speechiness`, `duration_sec`.
- **1 Categorical Feature**: `genre` (114 one-hot encoded subgenres).
- **Restricted Metadata (Never Used Predictively)**:
  - `primary_artist`: Quarantined exclusively as a grouping key for cross-validation splits to prevent models from memorizing superstar marketing power.
  - `chart_year`: Preserved strictly for post-hoc coverage audits; excluded from modeling because unreleased/non-charting catalog tracks lack verified release timestamps.

---

## 3. Machine Learning Pipeline & Benchmark Results

### 8 Classifier Architectures Benchmarked
Eight distinct model families were tuned using inner 5-fold `StratifiedGroupKFold` cross-validation on training data optimizing for ROC-AUC, then evaluated on the frozen held-out test split (1,480 songs):

| Rank | Model Architecture | ROC-AUC | PR-AUC | Recall | Specificity | Precision | Accuracy | F1-Score | TP | TN | FP | FN |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression (Selected)** | **0.929** | 0.606 | **0.844** | 0.862 | 0.380 | 0.860 | **0.524** | 114 | 1,159 | 186 | 21 |
| 2 | **SVM (RBF Kernel)** | 0.924 | 0.595 | 0.770 | 0.865 | 0.364 | 0.856 | 0.494 | 104 | 1,163 | 182 | 31 |
| 3 | **Linear SVM** | 0.924 | 0.602 | 0.793 | 0.852 | 0.350 | 0.847 | 0.485 | 107 | 1,146 | 199 | 28 |
| 4 | **Random Forest** | 0.918 | **0.619** | 0.296 | **0.993** | **0.816** | **0.930** | 0.435 | 40 | 1,336 | 9 | 95 |
| 5 | **K-Nearest Neighbors** | 0.877 | 0.509 | 0.370 | 0.987 | 0.735 | 0.930 | 0.493 | 50 | 1,327 | 18 | 85 |
| 6 | **Gaussian Naive Bayes** | 0.875 | 0.305 | 0.911 | 0.787 | 0.300 | 0.798 | 0.451 | 123 | 1,058 | 287 | 12 |
| 7 | **Decision Tree (Entropy)** | 0.821 | 0.443 | 0.341 | 0.989 | 0.754 | 0.930 | 0.469 | 46 | 1,330 | 15 | 89 |
| 8 | **Decision Tree (Gini)** | 0.821 | 0.513 | 0.363 | 0.982 | 0.671 | 0.926 | 0.471 | 49 | 1,321 | 24 | 86 |

### Why Logistic Regression Won (Multi-Criteria Evaluation)
1. **Accuracy is Deceptive on Imbalanced Data**: Random Forest achieves 93.0% accuracy, but a dummy model predicting "never charts" achieves **90.9% accuracy**. Random Forest misses **95 out of 135 true chart hits** (Recall = 29.6%), which is commercially fatal for scouting.
2. **Top ROC-AUC**: Logistic Regression achieved the highest test ROC-AUC (**0.929**), proving superior overall ranking discrimination across unseen artists.
3. **Interpretability & Speed**: Unlike a 300-tree black box ensemble, Logistic Regression provides direct standardized log-odds coefficients ($\beta$) that can be audited by non-technical stakeholders and executed with sub-millisecond inference latency.

---

## 4. Leakage-Safe Operational Threshold Selection

Tuning decision thresholds directly on test data leads to severe test-set overfitting. Operational thresholds were identified exclusively from **5-fold training out-of-fold (OOF) predictions** and evaluated once on the frozen held-out test split:

| Threshold Strategy | Selected $\tau$ | Accuracy | Precision | Recall | Specificity | F1-Score | Hits Captured (TP/135) | Missed Hits (FN) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Youden-J (Deployment)** | **0.48** | 0.8568 | 0.3746 | **0.8519** | 0.8572 | 0.5204 | **115 / 135 (85.2%)** | **20** |
| **F1-Max Maximizing** | **0.77** | 0.9074 | 0.4946 | 0.6741 | 0.9309 | 0.5705 | 91 / 135 (67.4%) | 44 |

- **Operational Selection ($\tau = 0.48$)**: Derived by maximizing Youden's $J = \text{Sensitivity} + \text{Specificity} - 1$. Captures **85.2% of all true hits** while rejecting **85.7% of non-hits**, minimizing commercially catastrophic false negatives.

---

## 5. Explainability & Standardized Coefficients

Standardized logistic coefficients ($\beta$) communicate the change in log-odds of charting associated with a $+1\sigma$ increase in each continuous feature:

- **`instrumentalness` ($\beta = -1.868$)**: The single largest negative audio association; commercial Hot 100 hits overwhelmingly require prominent human vocals.
- **`loudness` ($\beta = +1.349$)**: The strongest positive audio association; commercial tracks mastered to modern loudness standards exhibit significantly higher log-odds of charting.
- **`duration_sec` ($\beta = -0.803$)**: Radio and streaming playlist formats penalize excessive song lengths.
- **`danceability` ($\beta = +0.552$)**: Regular 4/4 rhythmic structures strongly support commercial adoption.
- **`acousticness` ($\beta = -0.525$)**: Organic acoustic instrumentation shows strong negative association with chart entry.
- **`energy` ($\beta = -0.475$) & `valence` ($\beta = -0.384$)**: Once mastering loudness is controlled for, raw waveform energy and musical happiness show slight negative associations.

---

## 6. Unsupervised Audio Feature Clustering

K-Means clustering on the 9 standardized audio features identified $k=4$ natural acoustic archetypes (2D PCA explains 48.7% of feature variance):

- **Cluster 0 (Upbeat Pop / Dance)**: 3,348 songs (45.2% of data), **13.92% hit rate (466 total hits)**. Features high danceability and loudness. Cuts across country, pop, latin, and dance genres.
- **Cluster 1 (Acoustic / Downtempo)**: 1,492 songs (20.2% of data), **1.94% hit rate (29 hits)**. High acousticness, low loudness.
- **Cluster 2 (Intense Rock & Metal)**: 2,315 songs (31.3% of data), **6.83% hit rate (158 hits)**. High energy, low danceability.
- **Cluster 3 (High-Speechiness / Spoken Word)**: 248 songs (3.3% of data), **8.06% hit rate (20 hits)**. Dominated by comedy and spoken rhythm.

---

## 7. Audio Feature Extraction Pipeline (Experimental Feature)

An optional, privacy-respecting audio extraction pipeline enables users to upload local MP3 tracks and estimate features locally without third-party APIs. Uploaded audio is processed in temporary memory and immediately deleted.

### Feature Flag (`ENABLE_AUDIO_UPLOAD`)
- **Default State**: Disabled (`ENABLE_AUDIO_UPLOAD=false`). The app functions identically to the pre-upload baseline (upload block is hidden, `/api/extract` returns `403 Forbidden`).
- **Enabled State**: Set `ENABLE_AUDIO_UPLOAD=true` to display the drag-and-drop upload block with experimental disclaimers.

### Extraction Taxonomy & DSP Methods
| Feature | Taxonomy Tier | Signal Processing Method / Library | Notes |
|:---|:---|:---|:---|
| **`duration_sec`** | Directly Measurable | Container sample count divided by sample rate ($N / f_s$) via `soundfile`/`librosa` | Physical exact measurement |
| **`tempo`** | Directly Measurable | Onset envelope autocorrelation (`librosa.feature.tempo`) with octave folding into [50, 180] BPM | Genuine beat tracking |
| **`loudness`** | Directly Measurable | Integrated loudness in **LUFS** (ITU-R BS.1770-4) via `pyloudnorm.Meter` | International broadcast standard (distinct from Spotify) |
| **`energy`** | Acoustic Proxy | Frame-wise RMS amplitude normalized by 0.20: $\min(\max(\text{RMS} / 0.20, 0), 1)$ | Signal energy proxy |
| **`danceability`** | Acoustic Proxy | Autocorrelation pulse clarity peak prominence across [0.3s, 1.2s] beat lags | Non-saturating pulse clarity heuristic |
| **`valence`** | Manual Only | **Omitted by backend**. Requires deep emotional modeling. | User sets via slider |
| **`acousticness`** | Manual Only | **Omitted by backend**. Requires statistical ML classifier. | User sets via slider |
| **`instrumentalness`** | Manual Only | **Omitted by backend**. Requires vocal source separation. | User sets via slider |
| **`speechiness`** | Manual Only | **Omitted by backend**. Requires voice activity detection. | User sets via slider |
| **`genre`** | Manual Only | **Never inferred from audio**. Must be explicitly chosen by the user. | User sets via dropdown |

---

## 8. Empirical Audio Validation & Audit Results

To evaluate whether open-source DSP estimates match Spotify's proprietary API features, 20 MP3 files corresponding to songs in the held-out test split were audited (`reports/audio_validation.csv`, `reports/audio_validation.md`).

**The audio extraction pipeline failed the equivalence acceptance gate:**
- **Automatic Features ($\rho \ge 0.60$)**: **1 / 9 features** (`duration_sec`: $\rho = 0.7188$; all others failed).
- **Model Label Agreement**: **55.0%** (11 of 20 tracks agreed).
- **Mean Absolute Probability Difference**: **0.3451** ($34.51$ percentage points).
- **Hit Share Discrepancy**: **45.0 pp** (Spotify: 65.0%, Extracted: 20.0%).
- **Overall Verdict**: **FAIL** (Integrated strictly as an *experimental convenience tool* with explicit UI warnings).

### Root Causes of Discrepancies
1. **LUFS Broadcast Standards vs. Spotify Proprietary Scale**: Standard ITU-R BS.1770-4 integrated loudness averages $-14.22$ LUFS on local files compared to $-7.31$ dB on Spotify's streaming masters (an offset of $\sim -6.91$ dB). This systematically depresses charting log-odds in the model.
2. **Track Version & Edit Mismatches**: Local MP3 files frequently diverge from Spotify album single masters (e.g. *Daddy Come Home* was an extended compilation track with a $+156.26$ s difference; *My Way* was a video cut with a $+63.27$ s outro).
3. **Psychoacoustic Gap**: Physical RMS amplitude cannot capture human perceptual energy, which depends on spectral flux, high-frequency density, and dynamic contrast.

---

## 9. Repository Structure

```text
HitSong/
├── data/
│   ├── raw/                       # Untouched raw source CSVs
│   └── processed/                 # Cleaned parquet dataset (songs.parquet)
├── src/                           # Standalone, importable pipeline modules
│   ├── preprocess.py              # Artist-grouped train/test splitting & transformers
│   ├── train.py                   # 5-fold StratifiedGroupKFold inner tuning (8 models)
│   ├── evaluate.py                # Held-out test evaluation & curve plotting
│   ├── explain.py                 # Feature importances, coefficients, & hypotheses
│   ├── threshold_analysis.py      # 5-fold OOF leakage-safe threshold selection
│   ├── select_final.py            # Final model serialization & metadata generation
│   ├── clustering.py              # Unsupervised K-Means audio feature clustering
│   ├── audio_extractor.py         # Local Tier 1 MP3 audio extraction & proxies
│   └── generate_pdf_report.py     # Publication-grade 14-page PDF report generator
├── models/                        # Serialized pipeline artifacts
│   └── final/
│       ├── model.joblib           # Final fitted Logistic Regression pipeline
│       └── metadata.json          # Dynamic schema, slider bounds, threshold, and genres
├── reports/                       # Academic documentation & figures
│   ├── HitSong_Comprehensive_Report.pdf # Complete 14-page technical PDF report
│   ├── FINAL_REPORT.md            # Comprehensive academic research report
│   ├── audio_validation.md        # 20-track audio extraction validation report
│   ├── audio_validation.csv       # Per-song validation metrics & probability differences
│   ├── audio_feature_metrics.csv  # Per-feature correlation and error metrics
│   ├── figures/                   # High-resolution publication plots (150 dpi)
│   └── tables/                    # Generated metrics and parameter summaries
├── app/                           # FastAPI production backend
│   └── main.py                    # REST API endpoints (/health, /schema, /predict, /extract)
├── web/                           # Vanilla static frontend (design.md compliant)
│   ├── index.html                 # Manuscript-style accessible interface
│   ├── style.css                  # Fluid layout, custom sliders, diverging bars, upload block
│   └── app.js                     # Debounced interaction, presets, live inference, MP3 upload
├── tests/                         # Automated test suite (52 passing tests)
│   ├── test_preprocess.py         # Split integrity, artist separation, zero NaNs
│   ├── test_evaluate.py           # Metric calculations, zero-division safety
│   ├── test_explain.py            # Importance aggregation and hypothesis stats
│   ├── test_threshold_analysis.py # OOF determinism, zero test leakage
│   ├── test_select_final.py       # Pipeline serialization and metadata schema
│   ├── test_api.py                # Core API endpoints, 422 validation, static serving
│   ├── test_clustering.py         # K-Means pipeline, PCA projection, profiles
│   ├── test_audio_extractor.py    # Synthetic audio extraction & clamping tests
│   └── test_audio_api.py          # Upload endpoint, 20MB limit, flag gating, corrupt MP3s
├── AGENTS.md                      # Binding engineering rules and reproducibility constraints
├── design.md                      # Binding visual and interaction design specification
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # Project documentation
```

---

## 10. Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/Kaif-1024170255/HitSong.git
cd HitSong

# 2. Create and activate virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 11. End-to-End Execution & Reproduction

Run each pipeline step standalone from the repository root:

```bash
# Step 1: Preprocess dataset and generate artist-grouped train/test split (80/20)
python -m src.preprocess

# Step 2: Train and tune all 8 classifiers using 5-fold StratifiedGroupKFold CV
python -m src.train

# Step 3: Evaluate models on the held-out test set
python -m src.evaluate

# Step 4: Run explainability analysis (importances, coefficients, hypotheses)
python -m src.explain

# Step 5: Execute leakage-safe threshold selection on training OOF predictions
python -m src.threshold_analysis

# Step 6: Select final model and serialize models/final/ production bundle
python -m src.select_final

# Step 7: Run unsupervised audio feature clustering (K-Means & PCA projection)
python -m src.clustering

# Step 8: Generate the 14-page publication-quality PDF report
python -m src.generate_pdf_report

# Step 9: Run the complete automated test suite (52 tests)
pytest tests/ -v
```

---

## 12. Running the Web Application

### Normal Mode (Standard Manual Predictor)
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Experimental Audio Upload Mode
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
```text
http://127.0.0.1:8000
```

### API Endpoints
- `GET /api/health`: Service health, model identification, and training timestamp.
- `GET /api/schema`: Dynamically returns feature list, slider min/max/step/default boundaries, and 114 approved genres.
- `GET /api/insights`: Returns feature importances and model comparison benchmark records.
- `POST /api/predict`: Validates payload against training boundaries and returns probability, binary label, operational threshold, and local feature contributions.

---

## 7. Headline Benchmark Results

### Held-Out Test Set Performance (`reports/tables/model_comparison.csv`)
Evaluated on 1,480 unseen songs (135 charted, 1,345 non-charted; 0 artist overlap with training data):

| Rank | Model | ROC-AUC | PR-AUC | Recall | Specificity | Precision | Accuracy | F1-Score | TP | TN | FP | FN |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression** | **0.929** | 0.606 | **0.844** | 0.862 | 0.380 | 0.860 | **0.524** | 114 | 1159 | 186 | 21 |
| 2 | **SVM (RBF Kernel)** | 0.924 | 0.595 | 0.770 | 0.865 | 0.364 | 0.856 | 0.494 | 104 | 1163 | 182 | 31 |
| 3 | **Linear SVM** | 0.924 | 0.602 | 0.793 | 0.852 | 0.350 | 0.847 | 0.485 | 107 | 1146 | 199 | 28 |
| 4 | **Random Forest** | 0.918 | **0.619** | 0.296 | **0.993** | **0.816** | **0.930** | 0.435 | 40 | 1336 | 9 | 95 |
| 5 | **K-Nearest Neighbors** | 0.877 | 0.509 | 0.370 | 0.987 | 0.735 | 0.930 | 0.493 | 50 | 1327 | 18 | 85 |
| 6 | **Gaussian Naive Bayes** | 0.875 | 0.305 | 0.911 | 0.787 | 0.300 | 0.798 | 0.451 | 123 | 1058 | 287 | 12 |
| 7 | **Decision Tree (Entropy)** | 0.821 | 0.443 | 0.341 | 0.989 | 0.754 | 0.930 | 0.469 | 46 | 1330 | 15 | 89 |
| 8 | **Decision Tree (Gini)** | 0.821 | 0.513 | 0.363 | 0.982 | 0.671 | 0.926 | 0.471 | 49 | 1321 | 24 | 86 |

### Leakage-Safe Operational Threshold Selection (`reports/tables/final_threshold_evaluation.csv`)
Thresholds were derived strictly from 5-fold training out-of-fold predictions and evaluated once on the frozen held-out test set:

| Threshold Method | Threshold | Accuracy | Precision | Recall | Specificity | F1-Score | TP | TN | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Youden-J Selected (Deployment)** | **0.48** | 0.8568 | 0.3746 | **0.8519** | 0.8572 | 0.5204 | 115 | 1153 | 192 | 20 |
| **F1-Max Selected** | **0.77** | **0.9074** | **0.4946** | 0.6741 | **0.9309** | **0.5705** | 91 | 1252 | 93 | 44 |

- **Operational Selection**: The Youden's J threshold of `0.48` captures **85.2%** of all genuine chart hits (115/135) with **85.7%** specificity, directly serving the commercial priority of minimizing costly false negatives.