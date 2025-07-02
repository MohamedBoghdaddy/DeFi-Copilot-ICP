from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import joblib

router = APIRouter()

# === Response Schemas ===
class ForecastResponse(BaseModel):
    model: str
    predictions: list[float]
    actual: list[float]

class MetricResponse(BaseModel):
    model: str
    MAE: float
    RMSE: float
    R2: float

class BestModelResponse(BaseModel):
    best_model: str
    rmses: dict

class PredictResponse(BaseModel):
    model: str
    days: int
    predictions: list[float]

class ExtendedForecastResponse(BaseModel):
    forecasts: dict[str, float]

# === Helper ===
def load_json_file(filename: str):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {filename}: {str(e)}")

# === /gold/history ===
@router.get("/history")
def get_gold_history(period: str = Query("1y", enum=["5y", "3y", "1y", "6mo", "3mo", "1mo", "15d", "7d", "1d"])):
    try:
        df = pd.read_csv("data/data.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["24K - Global Price"] = pd.to_numeric(df["24K - Global Price"], errors="coerce")
        df = df.dropna(subset=["Date", "24K - Global Price"]).sort_values("Date")

        today = df["Date"].max()
        period_map = {
            "5y": today - timedelta(days=5 * 365),
            "3y": today - timedelta(days=3 * 365),
            "1y": today - timedelta(days=365),
            "6mo": today - timedelta(days=180),
            "3mo": today - timedelta(days=90),
            "1mo": today - timedelta(days=30),
            "15d": today - timedelta(days=15),
            "7d": today - timedelta(days=7),
            "1d": today - timedelta(days=2),
        }

        cutoff = period_map.get(period)
        df = df[df["Date"] >= cutoff]

        history = df[["Date", "24K - Global Price"]].copy()
        history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")

        return {
            "period": period,
            "dates": history["Date"].tolist(),
            "prices": history["24K - Global Price"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History fetch error: {str(e)}")

# === /gold/forecast ===
@router.get("/forecast", response_model=list[ForecastResponse])
def get_all_gold_forecasts():
    try:
        results = load_json_file("data/Gold_forecast_results.json")
        df = pd.read_csv("data/data.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        y_true = df[results["best_model"]][-(len(results["rmses"]["Prophet"])):]  # Replace with correct column/length logic

        return [
            ForecastResponse(model=model, predictions=[], actual=[]) for model in results["rmses"]  # Placeholder
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast fetch error: {str(e)}")

# === /gold/metrics ===
@router.get("/metrics", response_model=list[MetricResponse])
def get_model_metrics():
    try:
        data = load_json_file("data/Gold_forecast_results.json")
        return [
            MetricResponse(model=model, MAE=-1, RMSE=rmses, R2=-1)  # Placeholder MAE and R2
            for model, rmses in data["rmses"].items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metric computation error: {str(e)}")

# === /gold/best-model ===
@router.get("/best-model", response_model=BestModelResponse)
def get_best_model():
    try:
        best_model = load_json_file("data/Gold_forecast_results.json")
        return BestModelResponse(
            best_model=best_model["best_model"],
            rmses=best_model["rmses"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Best model fetch error: {str(e)}")

# === /gold/predict ===
@router.get("/predict", response_model=PredictResponse)
def predict_next_days(days: int = Query(15, ge=1, le=60)):
    try:
        df = pd.read_csv("data/data.csv")
        df["24K - Global Price"] = pd.to_numeric(df["24K - Global Price"], errors="coerce")
        df["24K - Global Price"].fillna(df["24K - Global Price"].mean(), inplace=True)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df["24K - Global Price"].values.reshape(-1, 1))
        window = scaled[-60:].reshape(1, 60, 1)

        model = load_model("models/Gold_lstm_forecast_model.keras")
        pred_scaled = []

        for _ in range(days):
            pred = model.predict(window, verbose=0)[0][0]
            pred_scaled.append(pred)
            window = np.append(window[:, 1:, :], [[[pred]]], axis=1)

        predictions = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten().tolist()
        return PredictResponse(model="LSTM", days=days, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# === /gold/future-forecasts ===
@router.get("/future", response_model=ExtendedForecastResponse)
def get_extended_forecasts():
    try:
        forecast = load_json_file("data/Gold_forecast_extended.json")
        return ExtendedForecastResponse(forecasts=forecast)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast load error: {str(e)}")

# === FastAPI App ===
app = FastAPI(title="🌍 Gold Forecasting API")
app.include_router(router, prefix="/api/gold", tags=["Gold"])
