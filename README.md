# Hit Song Predictor

Binary supervised classification project to predict whether a song charts on the Billboard Hot 100 based on Spotify audio features.

This is a graded academic machine learning project where correctness, reproducibility, and explainability take precedence over cleverness.

---

## Repository Structure

```text
├── data/
│   ├── raw/            # Untouched source CSVs (never modified)
│   └── processed/      # Cleaned parquet/csv written by the pipeline
├── src/                # Pipeline code as importable modules
├── notebooks/          # Exploratory Data Analysis (EDA) only
├── models/             # Joblib model artifacts
├── reports/
│   ├── figures/        # Generated figures and charts (150 dpi)
│   └── tables/         # Generated metric evaluation CSVs
├── app/                # FastAPI backend service
├── web/                # Static frontend (plain HTML/CSS/vanilla JS)
├── AGENTS.md           # Persistent agent rules and guidelines
├── design.md           # Visual design specification for web/
├── requirements.txt    # Pinned Python package dependencies
└── README.md           # Project documentation
```

---

## Tech Stack

- **Python Version:** 3.11+
- **Data & ML:** pandas, numpy, scikit-learn, joblib, pyarrow
- **Visualization:** matplotlib, seaborn
- **Backend:** FastAPI, uvicorn
- **Frontend:** Plain HTML, CSS, Vanilla JavaScript (no build step, no frameworks)

---

## Getting Started

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Guidelines:**
   All agents and contributors must adhere to the rules outlined in [`AGENTS.md`](AGENTS.md).