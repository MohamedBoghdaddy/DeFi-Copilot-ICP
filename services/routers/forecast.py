from fastapi import APIRouter, Query, HTTPException
import pandas as pd
import joblib
import os
import json
from datetime import datetime, timedelta
from services.data_service import fetch_data, preprocess_data, create_features
from services.model_factory import train_and_forecast
from config import MODEL_DIR

router = APIRouter(tags=["Stock Forecasting"])

@router.get("/train")
def train_endpoint(symbol: str = Query("AAPL")):
    """Train all models and return results"""
    try:
        result = train_and_forecast(symbol)
        # Save JSON output for frontend
        with open(f"{MODEL_DIR}/{symbol}_forecast.json", 'w') as f:
            json.dump({
                "symbol": symbol,
                "best_model": result["best_model"],
                "forecast_dates": result["future_dates"],
                "forecast_values": result["future_forecast"]
            }, f)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict")
def predict(symbol: str = Query("AAPL")):
    """Get predictions from pre-trained models"""
    try:
        # Check if we have a recent forecast
        forecast_path = f"{MODEL_DIR}/{symbol}_forecast.json"
        if os.path.exists(forecast_path):
            with open(forecast_path) as f:
                return json.load(f)
        
        # If not, train new models
        result = train_and_forecast(symbol)
        return {
            "symbol": symbol,
            "best_model": result["best_model"],
            "forecast_dates": result["future_dates"],
            "forecast_values": result["future_forecast"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical")
def historical(
    symbol: str = Query("AAPL"),
    period: str = Query("1y", enum=["1d", "7d", "1mo", "3mo", "6mo", "1y", "3y", "5y"])
):
    """Get historical data"""
    try:
        end = datetime.today()
        start_map = {
            "1d": end - timedelta(days=1),
            "7d": end - timedelta(days=7),
            "1mo": end - timedelta(days=30),
            "3mo": end - timedelta(days=90),
            "6mo": end - timedelta(days=180),
            "1y": end - timedelta(days=365),
            "3y": end - timedelta(days=3*365),
            "5y": end - timedelta(days=5*365)
        }
        start = start_map.get(period, end - timedelta(days=365))
        
        df = fetch_data(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        df = preprocess_data(df)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        return {
            "symbol": symbol,
            "period": period,
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))