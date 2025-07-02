import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# === Setup Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# === Load .env ===
load_dotenv()

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DeFiCopilot")

# === FastAPI App ===
app = FastAPI(
    title="DeFi Copilot API",
    description="AI-Powered Financial Assistant: Forecasting, Chat, Recommendations",
    version="2.0.0",
    docs_url="/api/docs"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Routers ===
from routers import chat, forecast, recommend
app.include_router(chat.router, prefix="/chatbot", tags=["Chatbot"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
logger.info("✅ Routers loaded: /chatbot, /forecast, /recommend")

# === Trigger Services or Background Jobs ===
try:
    from scrapper.egypt_stock_news_scraper import init_scraper
    from services.gemini_api import init_gemini
    from services.model_factory import preload_models
    from utils.config import load_config
    from utils.scoring import init_scoring_system

    @app.on_event("startup")
    async def startup_tasks():
        logger.info("🚀 Running startup initializations...")
        load_config()
        init_scoring_system()
        preload_models()
        init_scraper()
        init_gemini()
        logger.info("✅ All services initialized.")

except ImportError as e:
    logger.warning(f"⚠️ Optional modules not loaded: {e}")

# === Root & Health Check ===
@app.get("/")
async def root():
    return {
        "message": "🚀 DeFi Copilot API is Live",
        "version": app.version,
        "docs": "/api/docs",
        "routes": {
            "/chatbot/chat": "AI Chatbot",
            "/chatbot/egypt-stock-news": "Scraped Stock News",
            "/forecast/{asset_type}": "Asset Forecasting (gold, real_estate, etc.)",
            "/recommend/investments": "Personalized Investment Advice"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}

# === Run App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True)
