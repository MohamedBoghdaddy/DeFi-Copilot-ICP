from xgboost import XGBRegressor
import joblib
import os
from utils.config import MODEL_DIR

def train_xgboost(X_train, y_train):
    """Train XGBoost model with optimal parameters"""
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    return model

def save_xgboost_model(model, symbol: str):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{symbol}_xgboost.pkl"))

def load_xgboost_model(symbol: str):
    return joblib.load(os.path.join(MODEL_DIR, f"{symbol}_xgboost.pkl"))

def forecast_xgboost(model, X):
    return model.predict(X)