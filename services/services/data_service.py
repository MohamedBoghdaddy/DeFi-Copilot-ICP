import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


def fetch_data(ticker: str, start: Optional[str] = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical price data from Yahoo Finance"""
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print(f"[📥] Fetching data for {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and interpolation"""
    print("[🧹] Preprocessing data...")
    df = df.copy()
    df = df.dropna(subset=["Close"])  # Ensure Close exists
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df["Return"] = df["Close"].pct_change()
    return df.dropna()


def create_features(df: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create lag features for ML forecasting.
    Example: lag-1 to lag-N close prices.
    """
    print(f"[🛠️] Creating features with window size {window}...")
    df = df.copy()
    for i in range(1, window + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)

    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, window + 1)]]
    y = df["Close"]
    return X, y
