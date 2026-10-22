
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "reports" / "best_demand_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

# Load only the trusted model artifact created by this project.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Decision-support API for experimental daily retail demand "
        "forecasting. Predictions require human review."
    ),
    version="1.0.0",
)


class PredictionInput(BaseModel):
    item_id: str = Field(
        description="Product ID, for example HOBBIES_1_001"
    )
    weekday: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2011, le=2035)
    lag_7: float = Field(ge=0)
    lag_14: float = Field(ge=0)
    lag_28: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    event_name_1: str = "NoEvent"
    event_type_1: str = "NoEvent"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": bundle.get("algorithm", type(model).__name__),
        "feature_count": len(feature_columns),
    }


@app.post("/predict")
def predict(request: PredictionInput):
    row = {column: 0 for column in feature_columns}

    # Populate the numeric input features.
    row.update({
        "weekday": request.weekday,
        "month": request.month,
        "year": request.year,
        "lag_7": request.lag_7,
        "lag_14": request.lag_14,
        "lag_28": request.lag_28,
        "rolling_mean_7": request.rolling_mean_7,
        "rolling_mean_28": request.rolling_mean_28,
    })

    # Match the one-hot encoding used during model training.
    categories = [
        f"item_id_{request.item_id}",
        f"event_name_1_{request.event_name_1}",
        f"event_type_1_{request.event_type_1}",
    ]

    for category in categories:
        if category not in feature_columns:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported category: {category}. "
                    "Use a category represented in the training data."
                ),
            )
        row[category] = 1

    features = pd.DataFrame([row], columns=feature_columns)
    prediction = float(model.predict(features)[0])

    return {
        "item_id": request.item_id,
        "predicted_units": round(max(0.0, prediction), 3),
        "prediction_type": "rolling one-step-ahead estimate",
        "model": bundle.get("algorithm", type(model).__name__),
        "decision_support_only": True,
        "warning": (
            "Experimental estimate; review against business context "
            "before making inventory decisions."
        ),
    }
