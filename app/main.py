"""FastAPI production service wrapping Hit Song Predictor models/final/ artifacts."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pydantic import BaseModel, field_validator

from src.preprocess import ALL_MODEL_FEATURES


def load_metadata(path: str = "models/final/metadata.json") -> dict[str, Any]:
    """Load production metadata from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


METADATA = load_metadata()
BOUNDS = METADATA.get("per_feature_min_max", {})
VALID_GENRES = set(METADATA.get("genre_categories", []))


class SongFeatures(BaseModel):
    """Validated input schema for song audio features and genre."""

    tempo: float
    energy: float
    danceability: float
    valence: float
    loudness: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    duration_sec: float
    genre: str

    @field_validator(
        "tempo", "energy", "danceability", "valence", "loudness",
        "acousticness", "instrumentalness", "speechiness", "duration_sec"
    )
    @classmethod
    def validate_numeric(cls, v: float, info) -> float:
        """Validate numeric value is within training bounds."""
        field = info.field_name
        b = BOUNDS.get(field)
        if b is not None and (v < b["min"] or v > b["max"]):
            raise ValueError(
                f"Field '{field}' must be between {b['min']} and {b['max']} (received {v})"
            )
        return v

    @field_validator("genre")
    @classmethod
    def validate_genre(cls, v: str) -> str:
        """Validate genre matches one of the known category names."""
        v_clean = v.strip().lower()
        if VALID_GENRES and v_clean not in VALID_GENRES:
            raise ValueError(
                f"Field 'genre' must be one of the known genre categories (received '{v}')"
            )
        return v_clean


def get_genre_effect(preproc: Any, coefs: Any, genre_val: str, num_count: int, X_trans: Any) -> float:
    """Calculate the log-odds effect for categorical genre."""
    cat_names = list(preproc.named_transformers_["cat"].get_feature_names_out(["genre"]))
    genre_target = f"genre_{genre_val}"
    if genre_target in cat_names:
        idx = num_count + cat_names.index(genre_target)
        return float(X_trans[0, idx] * coefs[idx])
    return 0.0


def compute_linear_contributions(pipe: Any, sample_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute linear model feature contributions (standardised value * coefficient).

    Method: Scaled numeric features and one-hot genre dummy multiplied by model coefs.
    """
    preproc, model = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    X_trans = preproc.transform(sample_df)
    X_trans = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans
    coefs = model.coef_[0]

    num_cols = [
        "tempo", "energy", "danceability", "valence", "loudness",
        "acousticness", "instrumentalness", "speechiness", "duration_sec"
    ]
    contribs = [
        {
            "feature": col,
            "value": float(sample_df[col].iloc[0]),
            "effect": round(float(X_trans[0, i] * coefs[i]), 4),
        }
        for i, col in enumerate(num_cols)
    ]
    genre_val = str(sample_df["genre"].iloc[0])
    g_effect = get_genre_effect(preproc, coefs, genre_val, len(num_cols), X_trans)
    contribs.append({"feature": "genre", "value": genre_val, "effect": round(g_effect, 4)})
    return sorted(contribs, key=lambda c: abs(c["effect"]), reverse=True)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Load model pipeline and metadata once at startup."""
    app_instance.state.model = joblib.load("models/final/model.joblib")
    app_instance.state.metadata = load_metadata("models/final/metadata.json")
    yield


app = FastAPI(title="Hit Song Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return explicit 422 JSON detailing offending fields and allowed ranges."""
    errors = []
    for err in exc.errors():
        loc = err.get("loc", [])
        field = loc[-1] if loc else "unknown"
        msg = err.get("msg", "")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        errors.append({"field": str(field), "message": msg})
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/api/health")
def get_health():
    """Service health and model metadata check."""
    meta = getattr(app.state, "metadata", METADATA)
    return {
        "status": "ok",
        "model_name": meta.get("model_name", "logistic_regression"),
        "trained_at": meta.get("trained_at", ""),
    }


@app.get("/api/schema")
def get_schema():
    """Return feature list, slider ranges, genres, and operational threshold."""
    meta = getattr(app.state, "metadata", METADATA)
    return {
        "features": meta.get("exact_ordered_feature_list", ALL_MODEL_FEATURES),
        "sliders": meta.get("per_feature_min_max", {}),
        "per_feature_min_max": meta.get("per_feature_min_max", {}),
        "genres": meta.get("genre_categories", []),
        "genre_categories": meta.get("genre_categories", []),
        "threshold": meta.get("chosen_threshold", 0.48),
    }


@app.get("/api/insights")
def get_insights():
    """Return feature importances and model comparison benchmark records."""
    feat_path = Path("reports/tables/feature_importance.json")
    comp_path = Path("reports/tables/model_comparison.csv")
    feat_imp = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.exists() else []
    comp = pd.read_csv(comp_path).to_dict(orient="records") if comp_path.exists() else []
    return {
        "feature_importance": feat_imp,
        "model_comparison": comp,
    }


@app.post("/api/predict")
def predict_hit(song: SongFeatures):
    """Predict Billboard Hot 100 charting probability and feature contributions."""
    pipe = app.state.model
    meta = getattr(app.state, "metadata", METADATA)
    th = float(meta.get("chosen_threshold", 0.48))
    ordered_feats = meta.get("exact_ordered_feature_list", ALL_MODEL_FEATURES)
    sample_df = pd.DataFrame([song.model_dump()])[ordered_feats]

    proba = float(pipe.predict_proba(sample_df)[0, 1])
    label = 1 if proba >= th else 0
    contributions = compute_linear_contributions(pipe, sample_df)

    return {
        "probability": round(proba, 4),
        "label": label,
        "threshold": th,
        "contributions": contributions,
    }


if os.path.exists("reports/figures"):
    app.mount("/reports/figures", StaticFiles(directory="reports/figures"), name="figures")

if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")

