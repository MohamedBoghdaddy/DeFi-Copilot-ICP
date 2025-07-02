import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.config import MODEL_DIR

# === Constants ===
SEQ_LENGTH = 30  # Number of time steps in each sequence


def train_rnn(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """Train a Simple RNN model for forecasting"""
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        SimpleRNN(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "rnn_best.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min"
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


def prepare_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH):
    """Convert 1D time series into RNN-ready sequences"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def forecast_rnn(model: Sequential, last_sequence: np.ndarray, steps: int, scaler) -> np.ndarray:
    """Generate future predictions from RNN model"""
    future_preds = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        input_seq = current_seq.reshape(1, SEQ_LENGTH, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        future_preds.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()


def save_rnn_model(model: Sequential, symbol: str):
    """Save trained RNN model for a specific asset"""
    model.save(os.path.join(MODEL_DIR, f"{symbol}_rnn.h5"))


def load_rnn_model(symbol: str) -> Sequential:
    """Load saved RNN model"""
    return load_model(os.path.join(MODEL_DIR, f"{symbol}_rnn.h5"))
