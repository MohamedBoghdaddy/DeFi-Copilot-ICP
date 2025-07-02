import os

# API Configuration
PREFERRED_API_ORDER = ["twelve_data", "marketstack", "alpha_vantage"]
APIS = {
    "twelve_data": "YOUR_TWELVE_DATA_KEY",
    "marketstack": "YOUR_MARKETSTACK_KEY",
    "alpha_vantage": "YOUR_ALPHA_VANTAGE_KEY"
}

# Path setup
MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Model configuration
MODELS = ["ARIMA", "XGBoost", "LSTM", "RandomForest", "RNN", "Prophet"]
FORECAST_DAYS = 30
TEST_SIZE = 0.2
SEQ_LENGTH = 60

def load_config():
    print("⚙️ Configuration loaded")
    return {
        "HOST": "0.0.0.0",
        "PORT": 8000,
        "RELOAD": True
    }
