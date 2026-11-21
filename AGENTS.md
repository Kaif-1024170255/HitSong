PROJECT: Hit Song Predictor. Binary supervised classification — predict whether a song
charts on the Billboard Hot 100 from Spotify audio features. This is a graded academic ML
project, so correctness and explainability matter more than cleverness.

STACK: Python 3.11+, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, FastAPI,
uvicorn. Frontend is plain HTML/CSS/vanilla JS — no React, no build step, no Tailwind.

STRUCTURE:
  data/raw/          untouched source CSVs, never modified
  data/processed/    cleaned parquet/csv written by the pipeline
  src/               all pipeline code as importable modules
  notebooks/         EDA only
  models/            joblib artifacts
  reports/figures/   every generated PNG, 150 dpi
  reports/tables/    every generated CSV of metrics
  app/               FastAPI backend
  web/               static frontend

HARD RULES:
1. Never fabricate a number. If you have not actually run the code and read the output,
   do not write a metric, a row count, or a class ratio anywhere — not in code comments,
   not in markdown, not in a report. Print it, read it, then write it.
2. Set random_state=42 on every split, model, and sampler. Results must reproduce.
3. Never fit any transformer on test data. Fit on train, transform both.
4. Every script runs standalone from the repo root: `python -m src.<module>`.
5. Every figure is saved to reports/figures/ AND displayed — never only shown.
6. When something is ambiguous (which column is the label, which dataset to use, what
   counts as a duplicate), STOP and ask me. Do not pick a default and continue silently.
7. Keep functions under 40 lines. No notebook logic that isn't also in src/.

---

SCOPE: add "upload an MP3 -> estimate features -> existing pipeline -> prediction". Nothing else.
ADDITIVE ONLY: new files/modules, minimal hooks in existing code. Never edit or retrain the
existing trained model, scaler or training scripts. Estimated features must go through the
same saved preprocessing and model the current predictor already uses.
FEATURE FLAG: everything is gated by ENABLE_AUDIO_UPLOAD (default false). With it false the
app must behave exactly as before.
THEME LOCK: reuse the existing CSS variables, fonts, spacing and components. No new colors,
fonts, icon sets or UI libraries. Add only the upload block and its states.
FREE ONLY: open-source libraries running locally. No paid or cloud audio APIs. Uploaded audio
is processed in a temp file and deleted, never stored.
HONESTY: never present estimated features as Spotify's values. Never change thresholds after
seeing results in order to force a pass.
DEFINITION OF DONE: tests pass AND you opened the screen in a running browser and used it.
ASK ME WHEN: an install fails, column names are ambiguous, a threshold needs deciding, or you
need audio files only I have. Stop and say exactly what you need.

---
