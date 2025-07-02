import pandas as pd
from prophet import Prophet
from typing import Tuple


def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format dataframe for Prophet training.
    Prophet expects columns named 'ds' (date) and 'y' (target value).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]]
    df.columns = ["ds", "y"]
    return df


def train_prophet(df: pd.DataFrame, symbol: str = "") -> Prophet:
    """
    Train a Prophet forecasting model on the provided data.
    """
    print(f"[🔮] Training Prophet model for {symbol}...")
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model


def forecast_prophet(model: Prophet, periods: int = 15) -> pd.DataFrame:
    """
    Generate future forecast using a trained Prophet model.
    Returns yhat, yhat_lower, and yhat_upper with dates.
    """
    print(f"[📈] Forecasting next {periods} steps using Prophet...")
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
