import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .arima_model import train_arima, forecast_arima
from .xgboost_model import train_xgboost, forecast_xgboost
from .lstm_model import train_lstm, forecast_lstm, prepare_sequences
from .rnn_model import train_rnn, forecast_rnn
from .prophet_model import train_prophet, forecast_prophet
from .random_forest_model import train_random_forest, forecast_random_forest
from .data_service import create_features
from utils.config import MODEL_DIR, FORECAST_DAYS, TEST_SIZE, SEQ_LENGTH
from sklearn.preprocessing import MinMaxScaler

def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred),
        "SMAPE": 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    }

def train_and_forecast(symbol: str):
    # Fetch and preprocess data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    df = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = preprocess_data(df)
    df = create_features(df)
    
    # [Rest of the training and evaluation logic...]
    # This would include:
    # 1. Train-test split
    # 2. Training all models
    # 3. Evaluating models
    # 4. Selecting best model
    # 5. Saving models
    # 6. Forecasting future values
    
    # Return results dictionary
    return {
        "symbol": symbol,
        "best_model": best_model_name,
        "metrics": metrics,
        "future_dates": future_dates_str,
        "future_forecast": future_pred.tolist()
    }
    
    
    
    
    
def preload_models():
    print("🧠 Models preloaded successfully")