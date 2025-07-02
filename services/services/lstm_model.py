import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from utils.config import MODEL_DIR

def train_lstm(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """Train LSTM model"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(100),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Callbacks
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

def save_lstm_model(model, symbol: str):
    model.save(os.path.join(MODEL_DIR, f"{symbol}_lstm.h5"))

def load_lstm_model(symbol: str):
    return load_model(os.path.join(MODEL_DIR, f"{symbol}_lstm.h5"))

def prepare_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH):
    """Prepare sequences for LSTM models"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def forecast_lstm(model, last_sequence: np.ndarray, steps: int, scaler):
    """Forecast future values using LSTM"""
    future_pred = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape input for LSTM
        input_seq = current_seq.reshape(1, SEQ_LENGTH, 1)
        
        # Predict next value
        pred = model.predict(input_seq, verbose=0)[0][0]
        future_pred.append(pred)
        
        # Update sequence
        current_seq = np.append(current_seq[1:], pred)
    
    # Inverse transform predictions
    return scaler.inverse_transform(np.array(future_pred).reshape(-1, 1)).flatten()