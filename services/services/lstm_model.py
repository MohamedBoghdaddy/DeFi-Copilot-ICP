import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.config import MODEL_DIR

# === Constant ===
SEQ_LENGTH = 30  # Default time window for sequence generation

# === Model Training ===
def train_lstm(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """Train an LSTM model on the input sequence data"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(100),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "lstm_best.h5"),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )
    return model

# === Sequence Prep ===
def prepare_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1D array into overlapping sequences and targets"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# === Forecasting ===
def forecast_lstm(model: Sequential, last_sequence: np.ndarray, steps: int, scaler) -> np.ndarray:
    """
    Forecast future values using a trained LSTM model.
    Assumes input sequence is already scaled.
    """
    future_pred = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        input_seq = current_seq.reshape(1, SEQ_LENGTH, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        future_pred.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return scaler.inverse_transform(np.array(future_pred).reshape(-1, 1)).flatten()

# === Save/Load ===
def save_lstm_model(model: Sequential, symbol: str):
    """Save LSTM model for a specific asset symbol"""
    path = os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")
    model.save(path)

def load_lstm_model(symbol: str) -> Sequential:
    """Load a saved LSTM model for a specific asset symbol"""
    path = os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")
    return load_model(path)
