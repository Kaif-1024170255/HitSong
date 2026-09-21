# Hit Song Predictor

Binary supervised machine learning system predicting whether a song charts on the Billboard Hot 100 using Spotify audio features. This project focuses on correctness, artist-leakage prevention, multi-criteria model evaluation, and interpretable deployment.

---

## 1. Project Overview

The Hit Song Predictor addresses a core commercial problem in music A&R (Artists and Repertoire): identifying which newly recorded songs have the acoustic characteristics associated with charting on the Billboard Hot 100.

- **Task**: Binary supervised classification (`charted`: 0 = Non-Charted, 1 = Charted).
- **Core Challenge**: Severe class imbalance (only ~9.1% of tracks chart), high risk of artist leakage across splits, and asymmetric operational error costs (missing a genuine hit is far costlier than reviewing a non-charting track).
- **Evaluation Philosophy**: Beyond raw accuracy (which misleadingly yields ~90.9% for a trivial majority classifier), models are evaluated on ROC-AUC, PR-AUC, recall, specificity, interpretability, and operational threshold trade-offs.

---

## 2. Dataset & Modelling Schema

The pipeline processes raw Spotify metadata and historical Billboard Year-End Hot 100 records (2000–2023) stored in `data/processed/songs.parquet`:

- **Total Processed Records**: 7,403 songs
- **Class Distribution**:
  - Positive Class (`charted = 1`): 673 songs (9.09%)
  - Negative Class (`charted = 0`): 6,730 songs (90.91%)
- **Source Filtering**: Downsampled at a 10:1 negative-to-positive ratio from an initial candidate pool of 80,237 tracks (673 positives, 79,564 negatives), excluding 11 ambiguous tracks with conflicting chart statuses.
- **Model Features (10 Total)**:
  - **9 Continuous Audio Features**: `tempo`, `energy`, `danceability`, `valence`, `loudness`, `acousticness`, `instrumentalness`, `speechiness`, `duration_sec`
  - **1 Categorical Feature**: `genre` (114 one-hot encoded categories)
- **Restricted Features (Never Used Predictively)**:
  - `primary_artist`: Used exclusively as a grouping variable (4,581 unique normalized artists) to prevent data leakage.
  - `chart_year`: Preserved strictly for post-hoc coverage audits; omitted from features because non-charting tracks lack verified release timestamps.

---

## 3. Repository Structure

```text
HitSong/
├── data/
│   ├── raw/                  # Untouched raw source CSVs
│   └── processed/            # Cleaned parquet dataset (songs.parquet)
├── src/                      # Standalone, importable pipeline modules
│   ├── preprocess.py         # Artist-grouped train/test splitting & transformers
│   ├── train.py              # 5-fold StratifiedGroupKFold inner tuning (8 models)
│   ├── evaluate.py           # Held-out test evaluation & curve plotting
│   ├── explain.py            # Feature importances, coefficients, & hypotheses
│   ├── threshold_analysis.py # 5-fold OOF leakage-safe threshold selection
│   ├── select_final.py       # Final model serialization & metadata generation
│   └── clustering.py         # Unsupervised K-Means audio feature clustering
├── models/                   # Serialized pipeline artifacts
│   ├── *.joblib              # Intermediate tuned models
│   └── final/                # Production deployment bundle
│       ├── model.joblib      # Final fitted Logistic Regression pipeline
│       └── metadata.json     # Dynamic schema, slider bounds, and genres
├── reports/                  # Complete academic documentation & assets
│   ├── FINAL_REPORT.md       # Comprehensive academic research report
│   ├── SLIDE_NOTES.md        # 8-slide presentation notes & figure mappings
│   ├── VIVA_PREP.md          # 15 examiner defense questions and answers
│   ├── model_selection.md    # Multi-criteria deployment selection rationale
│   ├── threshold_notes.md    # Commercial A&R threshold analysis notes
│   ├── explainability.md     # Feature importance and hypothesis analysis
│   ├── evaluation_notes.md   # Model comparison and diagnostic notes
│   ├── figures/              # High-resolution publication plots (150 dpi)
│   └── tables/               # Generated metrics, summaries, and parameters
├── app/                      # FastAPI production backend
│   └── main.py               # REST API endpoints (/health, /schema, /predict, /insights)
├── web/                      # Vanilla static frontend (design.md compliant)
│   ├── index.html            # Manuscript-style accessible interface
│   ├── style.css             # Fluid layout, custom sliders, diverging bars
│   └── app.js                # Debounced interaction, presets, and live inference
├── tests/                    # Automated test suite (35 passing unit tests)
│   ├── test_preprocess.py    # Split integrity, artist separation, zero NaNs
│   ├── test_evaluate.py      # Metric calculations, zero-division safety
│   ├── test_explain.py       # Importance aggregation and hypothesis stats
│   ├── test_threshold_analysis.py # OOF determinism, zero test leakage
│   ├── test_select_final.py  # Pipeline serialization and metadata schema
│   ├── test_api.py           # API endpoints, 422 validation, static serving
│   └── test_clustering.py    # K-Means pipeline, PCA projection, profiles
├── AGENTS.md                 # Binding engineering rules and reproducibility constraints
├── design.md                 # Binding visual and interaction specification
├── requirements.txt          # Pinned Python package dependencies
└── README.md                 # Project documentation
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kaif-1024170255/HitSong.git
   cd HitSong
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. End-to-End Reproduction Workflow

Execute every pipeline step sequentially from the project root:

```bash
# Step 1: Preprocess dataset and generate artist-grouped train/test split (80/20)
python -m src.preprocess

# Step 2: Train and tune all 8 classifiers using 5-fold StratifiedGroupKFold CV
python -m src.train

# Step 3: Evaluate models on the held-out test set (generates comparison CSV & curves)
python -m src.evaluate

# Step 4: Run explainability analysis (importances, coefficients, tree splits)
python -m src.explain

# Step 5: Execute leakage-safe threshold selection on training OOF predictions
python -m src.threshold_analysis

# Step 6: Select final model and serialize models/final/ production bundle
python -m src.select_final

# Step 7: Run unsupervised audio feature clustering (K-Means & PCA projection)
python -m src.clustering

# Step 8: Run the full automated test suite (35 unit tests)
pytest tests/ -v
```

---

## 6. Running the Web Application

Launch the unified FastAPI backend and static web client:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```text
http://localhost:8000
```

### Available API Endpoints
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
