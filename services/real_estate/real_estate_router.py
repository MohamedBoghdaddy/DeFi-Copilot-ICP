from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

router = APIRouter()

# === Response Schemas ===
class ForecastResponse(BaseModel):
    model: str
    forecast: list[float]

class MetricResponse(BaseModel):
    model: str
    MAE: float
    RMSE: float
    R2: float

class BestModelResponse(BaseModel):
    model: str
    RMSE: float

class PredictResponse(BaseModel):
    model: str
    days: int
    predictions: list[float]

class ReturnResponse(BaseModel):
    model: str
    predicted_return: str

# === /realestate/forecast ===
@router.get("/forecast", response_model=list[ForecastResponse])
def get_forecasts():
    try:
        with open("data/REAL_forecast_results.json", "r") as f:
            results = json.load(f)
        return [
            ForecastResponse(model=model, forecast=data["Forecast"])
            for model, data in results.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

# === /realestate/metrics ===
@router.get("/metrics", response_model=list[MetricResponse])
def get_metrics():
    try:
        with open("data/REAL_forecast_results.json", "r") as f:
            results = json.load(f)
        return [
            MetricResponse(model=model, MAE=data["MAE"], RMSE=data["RMSE"], R2=data["R2"])
            for model, data in results.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metric error: {str(e)}")

# === /realestate/best-model ===
@router.get("/best-model", response_model=BestModelResponse)
def get_best_model():
    try:
        with open("data/REAL_forecast_results.json", "r") as f:
            results = json.load(f)
        best = min(results.items(), key=lambda x: x[1]["RMSE"])
        return BestModelResponse(model=best[0], RMSE=best[1]["RMSE"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Best model error: {str(e)}")

# === /realestate/predict?days=15 ===
@router.get("/predict", response_model=PredictResponse)
def predict(days: int = Query(15, ge=1, le=60)):
    try:
        model = load_model("models/REAL_lstm_forecast_model.keras")
        df = pd.read_csv("egypt_House_prices.csv")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Price"].fillna(df["Price"].mean(), inplace=True)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

        window = scaled[-60:].reshape(1, 60, 1)
        predictions_scaled = []

        for _ in range(days):
            pred = model.predict(window, verbose=0)[0][0]
            predictions_scaled.append(pred)
            window = np.append(window[:, 1:, :], [[[pred]]], axis=1)

        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten().tolist()

        return PredictResponse(model="LSTM", days=days, predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# === /realestate/return ===
@router.get("/return", response_model=ReturnResponse)
def get_predicted_return():
    try:
        if not os.path.exists("models/REAL_xgb_model.pkl"):
            raise HTTPException(status_code=404, detail="Cached XGBoost model not found.")

        model = joblib.load("models/REAL_xgb_model.pkl")
        df = pd.read_csv("egypt_House_prices.csv")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
        df["Price"].fillna(df["Price"].mean(), inplace=True)
        df["Area"].fillna(df["Area"].mean(), inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days
        df.rename(columns={"Area": "Built Area"}, inplace=True)
        df = df.dropna(subset=["Days_Since_Start", "Built Area", "Price"])

        X_test = df[['Days_Since_Start', 'Built Area']].fillna(0)
        y_test = df['Price']

        test_pred = model.predict(X_test)
        current_price = y_test.iloc[-1]
        predicted_price = test_pred[-1]
        predicted_return = ((predicted_price - current_price) / current_price) * 100

        return ReturnResponse(model="XGBoost", predicted_return=f"{predicted_return:.2f}%")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Return calculation error: {str(e)}")

# === Standalone App ===
app = FastAPI(title="🏡 Real Estate Forecast API")
app.include_router(router, prefix="/api/realestate", tags=["Real Estate"])
