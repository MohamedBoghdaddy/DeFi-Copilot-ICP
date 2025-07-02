import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import joblib
import os
from utils.config import MODEL_DIR

def train_arima(train_data: pd.Series) -> SARIMAX:
    """Train ARIMA/SARIMA model with automatic parameter selection"""
    # Check stationarity
    result = adfuller(train_data)
    if result[1] > 0.05:
        d = 1  # Needs differencing
    else:
        d = 0
    
    # Automatically select best parameters
    model = auto_arima(
        train_data, 
        seasonal=True, 
        m=7,  # Weekly seasonality
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic'
    )
    
    # Train final model
    arima_model = SARIMAX(
        train_data,
        order=model.order,
        seasonal_order=model.seasonal_order
    )
    return arima_model.fit(disp=False)

def save_arima_model(model, symbol: str):
    model.save(os.path.join(MODEL_DIR, f"{symbol}_arima.pkl"))

def load_arima_model(symbol: str):
    return SARIMAX.load(os.path.join(MODEL_DIR, f"{symbol}_arima.pkl"))

def forecast_arima(model, steps: int) -> np.ndarray:
    return model.forecast(steps=steps)