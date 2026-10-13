#load the joblib file from the local path
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

#input and output valitations
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mlr_mpg_v1.joblib"

app = FastAPI(title="Multiple Linear Regression API", version="1.0")

class PredictionRequest(BaseModel):
    displacement: float = Field(..., description="Engine displacement in cubic inches")
    horsepower: float = Field(..., description="Engine horsepower")
    weight: float = Field(..., description="Vehicle weight in pounds")
    acceleration: float = Field(..., description="Time taken to accelerate from 0 to 60 mph in seconds")
    model_year: int = Field(..., description="Model year of the vehicle")
    origin: Literal[1, 2, 3] = Field(..., description="Origin of the vehicle (1=USA, 2=Europe, 3=Japan)")


class PredictionResponse(BaseModel):
    mpg: float = Field(..., description="Predicted miles per gallon (MPG) for the vehicle")

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")
else:
    model_load_error = None

#convert incoming json data to pandas dataframe for prediction
def build_inference_frame(request: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([request.model_dump()])

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_load_error:
        raise HTTPException(status_code=500, detail=f"Model loading error: {model_load_error}")

    inference_frame = build_inference_frame(request)
    predicted_mpg = model.predict(inference_frame)[0]
    return PredictionResponse(mpg=predicted_mpg)

@app.get("/health", response_model=dict)
def health_check():
    if model_load_error:
        return {"status": "error", "model_loaded": MODEL_PATH.name}



