import google.generativeai as genai
import os
import json
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set in environment")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
logger.info("✅ Gemini model initialized")

def analyze_articles(articles: List[Dict]) -> List[Dict]:
    """Analyze articles using Gemini for summarization and entity extraction"""
    processed = []
    for article in articles:
        try:
            content = article.get('content', '')[:10000]  # Truncate long content
            prompt = f"""
            Analyze this financial news article about Egyptian stocks and extract key information:
            
            Title: {article.get('title', '')}
            Content: {content}
            
            Provide analysis in JSON format with these keys:
            - summary: 2-3 sentence summary of main points
            - entities: list of stock tickers/companies mentioned
            - sentiment: overall sentiment (positive, negative, neutral)
            - impact_level: potential market impact (low, medium, high)
            - topics: key financial topics discussed
            """
            
            response = gemini_model.generate_content(prompt)
            analysis = json.loads(response.text.strip())
            
            processed.append({
                "title": article.get('title', ''),
                "url": article.get('url', ''),
                "published": article.get('published', ''),
                **analysis
            })
        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
    return processed

def analyze_market_impact(articles: List[Dict]) -> Dict:
    """Perform market impact analysis on processed articles"""
    articles_str = "\n\n".join(
        f"Title: {a['title']}\nSentiment: {a['sentiment']}\nImpact: {a['impact_level']}"
        for a in articles
    )
    
    prompt = f"""
    Based on these analyzed Egyptian stock news articles:
    {articles_str}
    
    Provide overall market analysis in JSON format with:
    - overall_sentiment: aggregate sentiment across all articles
    - high_impact_tickers: list of stocks likely to be significantly affected
    - market_trend_prediction: short-term market trend prediction
    - risk_assessment: potential risks identified
    - recommended_actions: suggested investor actions
    """
    
    response = gemini_model.generate_content(prompt)
    return json.loads(response.text.strip())

def generate_financial_advice(prompt: str) -> str:
    """Generate financial advice using Gemini"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini advice generation failed: {str(e)}")
        return "I'm having trouble generating advice right now. Please try again later."
    
    
    
def init_gemini():
    print("🔮 Gemini service initialized")