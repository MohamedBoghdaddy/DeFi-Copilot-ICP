from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import os
import logging
import hashlib
import asyncio
import pandas as pd
import yfinance as yf
import requests
import time
from cachetools import TTLCache
from dotenv import load_dotenv
from scipy.stats import linregress
import joblib
import numpy as np
import httpx
import random
from textblob import TextBlob
from .gemini_api import gemini_model, generate_financial_advice, analyze_articles, analyze_market_impact
from scrapper.egypt_stock_news_scraper import scrape_egypt_stock_news

# Initialize router
router = APIRouter(prefix="/chatbot", tags=["Chatbot"])
logger = logging.getLogger(__name__)

# === Global Caches ===
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
STOCK_DATA_CACHE = TTLCache(maxsize=200, ttl=3600)  # 1-hour cache
SESSION_HISTORY = TTLCache(maxsize=1000, ttl=3600)  # 1-hour session cache
SESSION_LOCK = asyncio.Lock()
ASSET_CACHE = TTLCache(maxsize=500, ttl=1800)  # 30-minute asset cache

# === Financial Knowledge Base ===
RISK_CATEGORIES = {
    "conservative": {
        "stocks": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "T", "VZ", "SO", "DUK"],
        "description": "stable blue-chip companies with consistent dividends",
        "max_volatility": 0.3,
        "allocation": "15-25% of portfolio"
    },
    "balanced": {
        "stocks": ["MSFT", "AAPL", "V", "MA", "DIS", "HD", "LOW", "COST", "JPM", "BAC"],
        "description": "balanced growth stocks with moderate volatility",
        "max_volatility": 0.5,
        "allocation": "25-40% of portfolio"
    },
    "growth": {
        "stocks": ["GOOGL", "AMZN", "TSLA", "META", "NFLX", "ADBE", "CRM", "PYPL", "AVGO", "QCOM"],
        "description": "high-growth potential stocks",
        "max_volatility": 0.8,
        "allocation": "20-35% of portfolio"
    },
    "aggressive": {
        "stocks": ["NVDA", "AMD", "SNOW", "CRWD", "PLTR", "MRNA", "BILL", "DDOG", "NET", "ZS"],
        "description": "high-risk/high-reward innovative companies",
        "max_volatility": 1.2,
        "allocation": "10-20% of portfolio"
    }
}

FINANCIAL_TIPS = [
    "💰 Save at least 20% of your income each month.",
    "📉 Avoid impulse buying by waiting 24 hours before making a purchase.",
    "📊 Invest in diversified assets to reduce risk.",
    "🏦 Use high-yield savings accounts for emergency funds.",
    "💳 Pay off high-interest debt as soon as possible to avoid extra fees.",
    "📈 Consider dollar-cost averaging to reduce market timing risk.",
    "🌍 Diversify internationally to hedge against country-specific risks.",
    "📅 Rebalance your portfolio at least once per year.",
    "🧾 Keep investment expenses below 0.5% of assets annually.",
    "🛡️ Maintain 3-6 months of living expenses in cash equivalents."
]

FAQS = {
    "how to save money": "💰 Save at least 20% of your income each month and avoid impulse purchases.",
    "best way to invest": "📊 Diversify your investments and consider low-cost index funds.",
    "how to improve credit score": "✅ Pay bills on time and keep credit utilization below 30%.",
    "how to start budgeting": "📋 Track your expenses and allocate your income into savings, needs, and wants.",
    "what is dollar cost averaging": "⏳ Invest fixed amounts regularly to reduce market timing risk.",
    "how much to invest in stocks": "📈 Allocate (100 - your age)% in stocks, e.g., 70% stocks if you're 30.",
    "best long term investments": "🌱 Consider index funds, blue-chip stocks, and real estate for long-term growth.",
    "how to analyze stocks": "🔍 Look at P/E ratio, growth rates, competitive advantage, and management quality.",
    "dollar cost averaging": "⏳ Investing fixed amounts at regular intervals regardless of market conditions",
    "portfolio diversification": "📊 Spreading investments across different assets to reduce risk"
}

# === API Configuration ===
PREFERRED_API_ORDER = [
    "twelve_data",
    "marketstack",
    "alpha_vantage",
    "yfinance"
]

APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
    "binance": {
        "key": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_SECRET_KEY")
    },
}

# === Path Constants ===
AI_INSIGHTS_PATH = "ai_insights.json"
MODEL_PATH = "model/rf_model.pkl"
SCALER_PATH = "model/rf_scaler.pkl"

# === Asset Data Sources ===
ASSET_DATA = {
    "real_estate": {
        "forecast": "real_estate/data/REAL_forecast_results.json",
        "entities": "real_estate/egypt_House_prices.csv"
    },
    "stocks": {
        "forecast": "stock/Stock_forecast_results.json",
        "entities": "stock/stocks_dataset"
    },
    "gold": {
        "forecast": "gold/data/GOLD_forecast_results.json",
        "entities": "gold/data/data.csv"
    }
}

# ========================
# === Helper Functions ===
# ========================
def build_prompt(user_input: dict, goal: str) -> str:
    """Build prompt for financial advice"""
    logger.info("🔧 Building financial advice prompt")
    
    # Extract key fields with fallbacks
    income = user_input.get("salary") or user_input.get("income", "0")
    expenses = user_input.get("totalMonthlyExpenses", "0")
    savings = user_input.get("savingAmount") or (str(float(income) - float(expenses))) if income and expenses else "0"
    predictions = user_input.get("modelPredictions", {})
    volatility = user_input.get("marketVolatility", {})
    risk = user_input.get("riskTolerance", "5")
    horizon = user_input.get("investmentHorizon", "3")
    favorite_sectors = user_input.get("favoriteSectors", [])
    past_stocks = user_input.get("previousInvestments", [])
    goals = user_input.get("financialGoals", "")
    dependents = user_input.get("dependents", "0")

    # === Smart Enhancements ===
    sector_focus = ", ".join(favorite_sectors) if favorite_sectors else "none specified"
    past_investments = ", ".join(past_stocks) if past_stocks else "no known history"
    financial_goals = goals if goals else "not specified"

    prompt = (
        "You are a professional financial advisor specialized in comprehensive financial planning.\n\n"
        "Your task is to analyze the user's profile and provide personalized advice based on their goals.\n\n"
        "Consider these factors:\n"
        f"- Risk Tolerance: {risk}/10\n"
        f"- Investment Horizon: {horizon} years\n"
        f"- Financial Goals: {financial_goals}\n"
        f"- Dependents: {dependents}\n"
        f"- Monthly Income: {income} EGP\n"
        f"- Monthly Expenses: {expenses} EGP\n"
        f"- Monthly Savings: {savings} EGP\n"
        f"- Favorite Sectors: {sector_focus}\n"
        f"- Previous Investments: {past_investments}\n\n"
        "Market Conditions:\n" +
        "".join([f"- {asset}: Predicted Return = {value}\n" for asset, value in predictions.items()]) +
        "".join([f"- {asset}: Volatility = {value}\n" for asset, value in volatility.items()]) +
        f"\nUser Query: {goal}\n\n"
        "Provide comprehensive advice covering:\n"
        "1. Specific investment recommendations\n"
        "2. Risk assessment based on profile\n"
        "3. Portfolio allocation strategy\n"
        "4. Budgeting suggestions\n"
        "5. Educational explanations of financial concepts\n\n"
        "Structure your response with clear sections and use financial terminology appropriately."
    )
    return prompt

def load_ai_insights() -> Dict[str, Dict]:
    """Load predicted returns and volatility from file"""
    logger.info("📂 Loading AI insights...")
    try:
        if os.path.exists(AI_INSIGHTS_PATH):
            logger.info(f"✅ Found AI insights file at {AI_INSIGHTS_PATH}")
            with open(AI_INSIGHTS_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load AI insights file: {str(e)}")
    
    logger.warning("⚠️ Using default AI insights")
    return {
        "predicted_returns": {"stocks": "8.9%", "gold": "6.2%", "real_estate": "7.15%"},
        "market_volatility": {"stocks": "0.06", "gold": "0.02", "real_estate": "0.01"}
    }

async def fetch_user_profile(token: str) -> dict:
    """Fetch user profile from ICP canister"""
    logger.info("🔍 Fetching user profile from ICP canister")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:4000/api/profile/me",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            logger.error(f"Profile API error: {response.text}")
    except Exception as e:
        logger.error(f"Profile fetch error: {str(e)}")
    return {}

async def fetch_stock_data(symbol: str, days: int = 30) -> dict:
    """Fetch stock data with robust fallback mechanism"""
    cache_key = f"{symbol}_{days}d"
    if cache_key in STOCK_DATA_CACHE:
        return STOCK_DATA_CACHE[cache_key]
    
    # Try APIs in preferred order
    for api_name in PREFERRED_API_ORDER:
        try:
            if api_name == "yfinance":
                stock = yf.Ticker(symbol)
                hist = stock.history(period=f"{days}d")
                if not hist.empty:
                    data = {
                        "Symbol": symbol,
                        "Close": hist['Close'].tolist(),
                        "Volume": hist['Volume'].tolist(),
                        "Current": hist['Close'].iloc[-1]
                    }
                    STOCK_DATA_CACHE[cache_key] = data
                    return data
            
            elif api_name == "twelve_data" and APIS.get("twelve_data"):
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.twelvedata.com/time_series",
                        params={
                            "symbol": symbol,
                            "interval": "1day",
                            "outputsize": days,
                            "apikey": APIS["twelve_data"]
                        }
                    )
                data = response.json()
                if "values" in data:
                    closes = [float(item["close"]) for item in data["values"]]
                    data = {"Close": closes, "Current": closes[-1]}
                    STOCK_DATA_CACHE[cache_key] = data
                    return data
                    
        except Exception as e:
            logger.warning(f"{api_name} failed for {symbol}: {str(e)}")
    
    # Gemini fallback
    try:
        prompt = f"Provide realistic stock data for {symbol} for the last {days} days"
        response = gemini_model.generate_content(prompt)
        data = json.loads(response.text)
        STOCK_DATA_CACHE[cache_key] = data
        return data
    except:
        return {}

async def fetch_multiple_stocks(symbols: List[str]) -> Dict[str, Any]:
    """Fetch multiple stocks in parallel"""
    tasks = [fetch_stock_data(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    return {sym: result for sym, result in zip(symbols, results) if result}

async def fetch_asset_price(asset_type: str, symbol: str = None) -> str:
    """Fetch price for various asset types"""
    if asset_type == "stock":
        return await fetch_stock_price(symbol)
    elif asset_type == "crypto":
        return await fetch_crypto_price(symbol)
    elif asset_type == "real_estate":
        return "🏠 Real estate pricing varies by location - provide specific area for analysis"
    elif asset_type == "gold":
        return await fetch_metal_prices("gold")
    return "Asset type not supported"

async def fetch_crypto_price(symbol: str) -> str:
    """Fetch cryptocurrency price from Binance"""
    clean_symbol = symbol.upper() + "USDT"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}"
            )
        data = response.json()
        return f"🚀 {symbol.upper()}: ${data['price']}" if 'price' in data else ""
    except:
        return ""

async def fetch_metal_prices(metal: str) -> str:
    """Fetch metal prices from Finnhub"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": "GC1!" if metal=="gold" else "SI1!", "token": APIS["finnhub"]}
            )
        data = response.json()
        return f"🥇 {metal.capitalize()}: ${data['c']}/oz"
    except:
        return ""

def get_risk_category(risk_score: int) -> str:
    """Categorize risk tolerance"""
    if risk_score < 3: return "conservative"
    if risk_score < 6: return "balanced"
    if risk_score < 9: return "growth"
    return "aggressive"

def calculate_technical_metrics(prices: List[float]) -> dict:
    """Calculate technical indicators for stocks"""
    if len(prices) < 5: return {}
    series = pd.Series(prices)
    return {
        "sma_10": series.rolling(10).mean().iloc[-1] if len(prices) >= 10 else None,
        "sma_20": series.rolling(20).mean().iloc[-1] if len(prices) >= 20 else None,
        "trend_slope": linregress(range(len(prices)), prices).slope
    }

def load_prediction_model():
    """Load ML model for predictions"""
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except:
        return None, None

async def predict_stock(symbol: str) -> dict:
    """Predict stock performance using ML model"""
    model, scaler = load_prediction_model()
    if not model or not scaler:
        return {}

    stock_data = await fetch_stock_data(symbol, 30)
    if not stock_data or not stock_data.get("Close"):
        return {}

    try:
        close = stock_data["Close"]
        features = np.array([
            close[-1], np.mean(close), np.std(close),
            (max(close) - min(close)), np.corrcoef(close[-10:], range(10))[0,1]
        ]).reshape(1, -1)
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        return {
            "symbol": symbol,
            "predicted_return": f"{prediction:.2%}",
            "confidence": "High" if abs(prediction) > 0.05 else "Medium"
        }
    except:
        return {}

def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "😊 Positive"
    elif analysis.sentiment.polarity < -0.1:
        return "😞 Negative"
    return "😐 Neutral"

# ===================================
# === Recommendation Engine ===
# ===================================
async def generate_recommendation(asset_type: str, user_profile: dict) -> str:
    """Generate personalized investment recommendation"""
    risk = user_profile.get("riskTolerance", 5)
    horizon = user_profile.get("investmentHorizon", 5)
    
    # Get risk-based recommendations
    risk_category = get_risk_category(risk)
    category_data = RISK_CATEGORIES[risk_category]
    
    if asset_type == "stocks":
        # Generate stock recommendation
        candidate_symbols = category_data["stocks"]
        stocks_data = await fetch_multiple_stocks(candidate_symbols)
        
        candidates = []
        for symbol, data in stocks_data.items():
            if not data: continue
            prices = data.get('Close', [])
            if len(prices) < 5: continue
            
            # Technical analysis
            technicals = calculate_technical_metrics(prices)
            trend_strength = "Strong" if technicals.get('trend_slope', 0) > 0.5 else "Moderate"
            
            # Prediction
            prediction = await predict_stock(symbol)
            pred_info = f"\n📈 Predicted: {prediction['predicted_return']} ({prediction['confidence']})" if prediction else ""
            
            candidates.append({
                "symbol": symbol,
                "price": data['Current'],
                "trend": ((prices[-1] - prices[0]) / prices[0]) * 100,
                "technicals": technicals,
                "trend_strength": trend_strength,
                "pred_info": pred_info
            })
        
        if not candidates:
            return "⚠️ Couldn't generate stock recommendations at this time"
        
        best_stock = max(candidates, key=lambda x: x['technicals'].get('trend_slope', 0))
        
        response = (
            f"📊 **Recommendation for {risk_category.title()} Investor**\n\n"
            f"**Stock**: {best_stock['symbol']} (${best_stock['price']:.2f})\n"
            f"**30-Day Trend**: {best_stock['trend']:.1f}%\n"
            f"**Trend Strength**: {best_stock['trend_strength']}{best_stock['pred_info']}\n\n"
            f"**Portfolio Advice**:\n"
            f"- Allocate {category_data['allocation']}\n"
            f"- Hold for {'long-term' if horizon > 3 else 'short-term'}"
        )
        return response
    
    else:
        # Generate non-stock asset recommendation
        ai_insights = load_ai_insights()
        pred_return = ai_insights["predicted_returns"].get(asset_type, "N/A")
        volatility = ai_insights["market_volatility"].get(asset_type, "N/A")
        
        response = (
            f"📈 **{asset_type.replace('_', ' ').title()} Investment**\n\n"
            f"Predicted Return: {pred_return}\n"
            f"Market Volatility: {volatility}\n\n"
            f"**Recommendation**:\n"
            f"Suitable for {risk_category} investors with {horizon} year horizon"
        )
        return response

# =====================
# === Request Models ===
# =====================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = ""
    token: Optional[str] = ""

class AdviceRequest(BaseModel):
    goal: str

# =====================
# === API Endpoints ===
# =====================
@router.post("/egypt-stock-news")
async def get_egypt_stock_news():
    """Get analyzed Egyptian stock news"""
    try:
        raw_articles = await scrape_egypt_stock_news()
        processed_articles = analyze_articles(raw_articles)
        market_analysis = analyze_market_impact(processed_articles)
        return {
            "articles": processed_articles,
            "market_analysis": market_analysis
        }
    except Exception as e:
        logger.error(f"Egypt stock news error: {str(e)}")
        raise HTTPException(500, "Failed to analyze Egyptian stock news")

@router.post("/analyze-news")
async def analyze_custom_news(request: Request):
    """Analyze custom news articles"""
    try:
        data = await request.json()
        articles = data.get("articles", [])
        processed_articles = analyze_articles(articles)
        market_analysis = analyze_market_impact(processed_articles)
        return {
            "articles": processed_articles,
            "market_analysis": market_analysis
        }
    except Exception as e:
        logger.error(f"News analysis error: {str(e)}")
        raise HTTPException(500, "Failed to analyze news articles")

@router.post("/generate/investment")
async def generate_investment_advice(request: Request):
    """Generate personalized investment advice"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header: 
            return JSONResponse({"error": "Authorization required"}, 401)
            
        token = auth_header.split(" ")[1]
        profile = await fetch_user_profile(token)
        if not profile:
            return JSONResponse({"error": "Profile not found"}, 404)
        
        prompt = build_prompt(profile, "investment advice")
        advice = generate_financial_advice(prompt)
        return {"advice": advice}
    except Exception as e:
        logger.error(f"Investment advice error: {str(e)}")
        return JSONResponse({"error": str(e)}, 500)

@router.post("/chat")
async def chat_with_bot(request: ChatRequest):
    """Main chatbot interaction endpoint"""
    try:
        user_message = request.message.strip()
        token = request.token
        session_id = request.session_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
        if not user_message:
            return {"output": "Please enter a valid question", "session_id": session_id}
        
        # Get cached response if available
        cache_key = hashlib.sha256(user_message.encode()).hexdigest()
        if cache_key in RESPONSE_CACHE:
            return {
                "output": RESPONSE_CACHE[cache_key],
                "session_id": session_id
            }
        
        # Fetch user profile
        profile = await fetch_user_profile(token) if token else {}
        
        # Handle specific question types
        lower_msg = user_message.lower()
        
        # Price inquiries
        if "price of" in lower_msg:
            asset_type = "stock"
            symbol = "AAPL"  # Default
            if "crypto" in lower_msg or "bitcoin" in lower_msg:
                asset_type = "crypto"
                symbol = "BTC"
            elif "gold" in lower_msg:
                asset_type = "gold"
            elif "real estate" in lower_msg:
                asset_type = "real_estate"
                
            response = await fetch_asset_price(asset_type, symbol)
        
        # Educational content
        elif any(keyword in lower_msg for keyword in ["what is", "explain"]):
            concept = next((k for k in FAQS if k in lower_msg), "general financial concept")
            response = FAQS.get(concept, "I can explain financial concepts like dollar cost averaging, portfolio diversification, etc.")
        
        # Investment recommendations
        elif any(keyword in lower_msg for keyword in ["invest", "buy", "recommendation"]):
            asset_type = "stocks"
            if "real estate" in lower_msg: asset_type = "real_estate"
            elif "gold" in lower_msg: asset_type = "gold"
            response = await generate_recommendation(asset_type, profile)
        
        # Budgeting advice
        elif any(keyword in lower_msg for keyword in ["budget", "save money"]):
            income = profile.get("salary", 0)
            expenses = profile.get("totalMonthlyExpenses", 0)
            savings = income - expenses if income and expenses else 0
            
            prompt = f"""
            Create a budgeting plan for someone with:
            - Monthly income: {income} EGP
            - Monthly expenses: {expenses} EGP
            - Monthly savings: {savings} EGP
            Provide specific allocation percentages and tips.
            """
            response = generate_financial_advice(prompt)
        
        # General financial advice
        else:
            prompt = f"""
            As a financial advisor, respond to this query: "{user_message}"
            Consider the user's profile if available:
            {json.dumps(profile, indent=2) if profile else "No profile available"}
            Provide comprehensive, professional advice.
            """
            response = generate_financial_advice(prompt)
        
        # Cache and return response
        RESPONSE_CACHE[cache_key] = response
        return {
            "output": response,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"error": "Service unavailable"}, 500

@router.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "operational",
        "services": {
            "gemini": "active",
            "stock_data": "available" if await fetch_stock_data("AAPL") else "degraded"
        }
    }