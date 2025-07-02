import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === Auto-create folders ===
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# === Preprocessing ===
def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
    df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['Price'].fillna(df['Price'].mean(), inplace=True)
    df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
    df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)
    df['Area'].fillna(df['Area'].mean(), inplace=True)
    if 'Date' not in df.columns:
        df['Date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='D')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days
    df.rename(columns={"Area": "Built Area"}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def create_seq(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def bin_targets(y, bins=3):
    return pd.qcut(y, q=bins, labels=False, duplicates='drop')

def normalized_evaluation(true, pred, model_name):
    bins = 3
    y_true_class = bin_targets(true, bins)
    y_pred_class = bin_targets(pred, bins)
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred),
        "F1": f1_score(y_true_class, y_pred_class, average="weighted"),
        "ConfusionMatrix": confusion_matrix(y_true_class, y_pred_class).tolist(),
        "Forecast": pred.tolist()
    }

def run_xgboost(train_df, test_df, y_train, y_test):
    print("\n=== XGBOOST MODEL ===")
    X_train = train_df[['Days_Since_Start', 'Built Area']].fillna(0)
    X_test = test_df[['Days_Since_Start', 'Built Area']].fillna(0)

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    metrics = normalized_evaluation(y_test, test_pred, "XGBoost")

    current_price = y_test.iloc[-1]
    predicted_price = test_pred[-1]
    predicted_return = ((predicted_price - current_price) / current_price) * 100

    print(f"📈 Predicted Real Estate Return: {predicted_return:.2f}%")
    metrics["predicted_return"] = f"{predicted_return:.2f}%"
    return model, test_pred, metrics

# === Visuals ===
def plot_model_comparisons(results):
    models = list(results.keys())
    rmse = [results[m]['RMSE'] for m in models]
    r2 = [results[m]['R2'] for m in models]
    f1 = [results[m]['F1'] for m in models]

    plt.figure(figsize=(15, 5))
    for idx, metric in enumerate([rmse, r2, f1]):
        plt.subplot(1, 3, idx + 1)
        sns.barplot(x=models, y=metric)
        plt.title(['RMSE', 'R2 Score', 'F1 Score'][idx])
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png")
    plt.close()

def plot_confusion_matrices(results):
    for model, data in results.items():
        cm = np.array(data['ConfusionMatrix'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"{model} - Confusion Matrix")
        plt.savefig(f"confusion_{model}.png")
        plt.close()

def plot_forecast_vs_actual(actual, predictions_dict):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual', color='black', linewidth=2)
    for model, preds in predictions_dict.items():
        plt.plot(actual.index[-len(preds):], preds, label=model)
    plt.title("Forecast vs Actual Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_vs_actual.png")
    plt.close()

# === Training and Forecasting ===
def train_and_forecast(csv_path, output_json, keras_model_path, xgb_model_path):
    df = load_and_clean_data(csv_path)
    target = df['Price']

    train_size = int(len(target) * 0.7)
    val_size = int(len(target) * 0.15)
    train, val, test = target[:train_size], target[train_size:train_size + val_size], target[train_size + val_size:]
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size + val_size:]

    # === ARIMA
    arima_model = ARIMA(train, order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=len(test))

    # === SARIMA
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test))

    # === Prophet
    prophet_df = df.reset_index()[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test))
    forecast = prophet_model.predict(future)
    prophet_pred = forecast[-len(test):]['yhat'].values

    # === LSTM
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(target.values.reshape(-1, 1))
    train_scaled = scaled[:train_size]
    val_scaled = scaled[train_size:train_size + val_size]
    test_scaled = scaled[train_size + val_size:]
    X_train, y_train = create_seq(train_scaled)
    X_val, y_val = create_seq(val_scaled)
    X_test, y_test = create_seq(test_scaled)
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    pred_scaled = lstm_model.predict(X_test)
    predicted = scaler.inverse_transform(pred_scaled)

    # === XGBoost (your version)
    xgb_model, xgb_pred, xgb_metrics = run_xgboost(df_train, df_test, train, test)

    # === Metrics
    def calc_metrics(true, pred):
        bins = 3
        y_true_class = bin_targets(true, bins)
        y_pred_class = bin_targets(pred, bins)
        return {
            "MAE": mean_absolute_error(true, pred),
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "R2": r2_score(true, pred),
            "F1": f1_score(y_true_class, y_pred_class, average="weighted"),
            "ConfusionMatrix": confusion_matrix(y_true_class, y_pred_class).tolist(),
            "Forecast": pred.tolist()
        }

    results = {
        "ARIMA": calc_metrics(test, arima_pred),
        "SARIMA": calc_metrics(test, sarima_pred),
        "Prophet": calc_metrics(test, prophet_pred),
        "LSTM": calc_metrics(test.values[60:], predicted.flatten()),
        "XGBoost": xgb_metrics
    }

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    lstm_model.save(keras_model_path)
    joblib.dump(xgb_model, xgb_model_path)

    plot_model_comparisons(results)
    plot_confusion_matrices(results)
    plot_forecast_vs_actual(test, {
        "ARIMA": arima_pred,
        "SARIMA": sarima_pred,
        "Prophet": prophet_pred,
        "LSTM": predicted.flatten(),
        "XGBoost": xgb_pred
    })

    print("✅ All models trained, saved, and visualized!")

# === Entry Point ===
if __name__ == "__main__":
    train_and_forecast(
        "egypt_House_prices.csv",
        "data/REAL_forecast_results.json",
        "models/REAL_lstm_forecast_model.keras",
        "models/REAL_xgb_model.pkl"
    )
