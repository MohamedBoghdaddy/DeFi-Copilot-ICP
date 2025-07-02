import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from typing import Union
from utils.config import MODEL_DIR


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, symbol: str) -> RandomForestRegressor:
    """
    Train a Random Forest model and save it to disk.
    """
    print(f"[🌲] Training Random Forest model for {symbol}...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    path = os.path.join(MODEL_DIR, f"{symbol}_rf.pkl")
    joblib.dump(model, path)
    print(f"[💾] Model saved to {path}")
    return model


def forecast_random_forest(
    model: RandomForestRegressor,
    last_sequence: np.ndarray,
    steps: int,
    scaler
) -> np.ndarray:
    """
    Forecast future values using a Random Forest regressor.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        input_data = current_seq.reshape(1, -1)
        pred = model.predict(input_data)[0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def load_random_forest_model(symbol: str) -> Union[RandomForestRegressor, None]:
    """
    Load a trained Random Forest model from disk.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}_rf.pkl")
    if not os.path.exists(path):
        print(f"[⚠️] No saved model found for {symbol}")
        return None
    return joblib.load(path)
