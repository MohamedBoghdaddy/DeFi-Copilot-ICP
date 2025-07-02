import json
import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import joblib

from prophet import Prophet
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

def train_and_forecast(file_path, results_path, lstm_model_path, xgb_model_path):
    def log(message):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    # === Load and clean data ===
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.dropna(axis=1, thresh=0.4 * len(df), inplace=True)
    df.dropna(thresh=int(0.7 * len(df.columns)), inplace=True)

    print("✅ Cleaned Data Columns:", df.columns.tolist())
    possible_target_cols = [col for col in df.columns if "24K" in col and "price" in col.lower()]
    if not possible_target_cols:
        raise ValueError("❌ No suitable target column found.")

    TARGET_COLUMN = possible_target_cols[0]
    print(f"✅ Using target column: {TARGET_COLUMN}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').interpolate(method='linear').ffill().bfill()

    df['rolling_mean_7'] = df[TARGET_COLUMN].rolling(window=7).mean()
    df['lag_1'] = df[TARGET_COLUMN].shift(1)
    df['lag_7'] = df[TARGET_COLUMN].shift(7)
    df['Days_Since'] = (df.index - df.index.min()).days
    df.dropna(inplace=True)

    # === Split ===
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    X = df[['Days_Since', 'rolling_mean_7', 'lag_1', 'lag_7']]
    y = df[TARGET_COLUMN]
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # === Prophet ===
    log("Training Prophet...")
    prophet_data = df.reset_index()[['Date', TARGET_COLUMN]].rename(columns={'Date': 'ds', TARGET_COLUMN: 'y'})
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=len(y_test))
    forecast = prophet_model.predict(future)
    prophet_preds = forecast['yhat'][-len(y_test):].values
    prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_preds))

    # === LSTM ===
    log("Training LSTM...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[TARGET_COLUMN]])
    seq_len = 60

    def create_seq(data):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X_all, y_all = create_seq(scaled_data)
    X_train_lstm = X_all[:train_size-seq_len]
    y_train_lstm = y_all[:train_size-seq_len]
    X_val_lstm = X_all[train_size-seq_len:train_size+val_size-seq_len]
    y_val_lstm = y_all[train_size-seq_len:train_size+val_size-seq_len]
    X_test_lstm = X_all[train_size+val_size-seq_len:]
    y_test_lstm = y_all[train_size+val_size-seq_len:]

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm), epochs=10, batch_size=32, verbose=0)
    lstm_preds_scaled = lstm_model.predict(X_test_lstm)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
    actual_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
    lstm_rmse = np.sqrt(mean_squared_error(actual_lstm, lstm_preds))

    # === XGBoost ===
    log("Training XGBoost...")
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

    # Save models
    joblib.dump(xgb_model, xgb_model_path)
    lstm_model.save(lstm_model_path)

    # === Forecast saving (1d to 5y) ===
    future_days = {
        "1d": 1, "15d": 15, "1m": 30, "3m": 90, "6m": 180,
        "1y": 365, "3y": 3 * 365, "5y": 5 * 365
    }
    future_forecasts = prophet_model.make_future_dataframe(periods=max(future_days.values()))
    extended_forecast = prophet_model.predict(future_forecasts)
    forecasts_dict = {
        label: extended_forecast.iloc[-days:]['yhat'].mean()
        for label, days in future_days.items()
    }
    with open("data/Gold_forecast_extended.json", "w") as f:
        json.dump(forecasts_dict, f)
    log("📈 Saved future forecasts.")

    # === Evaluation plots ===
    def save_forecast_plot(actual, predicted, model):
        plt.figure()
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label="Predicted")
        plt.legend()
        plt.title(f"Forecast vs Actual - {model}")
        plt.savefig(f"forecast_vs_actual_{model}.png")

    def save_confusion_and_f1(actual, predicted, model):
        bins = np.percentile(actual, [33, 66])
        y_true = np.digitize(actual, bins)
        y_pred = np.digitize(predicted, bins)
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        np.save(f"confusion_{model}.npy", cm)
        return f1

    save_forecast_plot(y_test, prophet_preds, "Prophet")
    save_forecast_plot(actual_lstm, lstm_preds, "LSTM")
    save_forecast_plot(y_test, xgb_preds, "XGBoost")

    # === Model Comparison Plot ===
    rmses = {
        "Prophet": prophet_rmse,
        "LSTM": lstm_rmse,
        "XGBoost": xgb_rmse
    }
    plt.figure(figsize=(8, 5))
    plt.bar(rmses.keys(), rmses.values(), color='skyblue')
    plt.title("Model Comparison - RMSE")
    plt.ylabel("RMSE")
    plt.savefig("model_comparison_metrics.png")

    # === Save Results JSON ===
    f1_scores = {
        "Prophet": save_confusion_and_f1(y_test, prophet_preds, "Prophet"),
        "LSTM": save_confusion_and_f1(actual_lstm.flatten(), lstm_preds.flatten(), "LSTM"),
        "XGBoost": save_confusion_and_f1(y_test, xgb_preds, "XGBoost"),
    }
    best_model = min(rmses.items(), key=lambda x: x[1])[0]

    results = {
        "best_model": best_model,
        "rmses": rmses,
        "f1_scores": f1_scores,
        "future_forecasts": forecasts_dict
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"✅ Best model: {best_model}")

# === Entry Point ===
if __name__ == "__main__":
    train_and_forecast(
        "data/data.csv",
        "data/Gold_forecast_results.json",
        "models/Gold_lstm_forecast_model.keras",
        "models/Gold_xgb_model.pkl"
    )
