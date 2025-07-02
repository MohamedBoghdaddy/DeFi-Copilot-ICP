import os
import json
import time
import requests
import yfinance as yf
import feedparser
import pandas as pd
import google.generativeai as genai
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional
from newspaper import Article, ArticleException

# -------------------------------
# 🔑 CONFIGURATION
# -------------------------------
GEMINI_API_KEY = "YOUR_API_KEY"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

PRICE_TICKERS = ["COMI.CA", "HRHO.CA", "ETEL.CA"]  # Yahoo Finance tickers
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# -------------------------------
# 📈 STOCK PRICE SCRAPERS
# -------------------------------
def scrape_yahoo_finance(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch historical price data from Yahoo Finance"""
    print("[📈] Downloading from Yahoo Finance...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE)
            if not df.empty:
                data[t] = df
                print(f"[✅] Downloaded {t} ({len(df)} rows)")
            else:
                print(f"[⚠️] No data for {t}")
        except Exception as e:
            print(f"[❌] Error fetching {t}: {str(e)}")
    return data

def scrape_egx_press_releases():
    """Scrape official EGX press releases"""
    print("[📈] Scraping EGX Press Releases...")
    url = "https://www.egx.com.eg/en/news.aspx"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        releases = []
        
        for row in soup.select('table.rgMasterTable tr')[1:6]:  # First 5 items
            cells = row.find_all('td')
            if len(cells) >= 2:
                releases.append({
                    'date': cells[0].text.strip(),
                    'title': cells[1].text.strip(),
                    'link': url
                })
        return releases
    except Exception as e:
        print(f"[❌] EGX scrape failed: {str(e)}")
        return []

# -------------------------------
# 📰 NEWS SCRAPERS
# -------------------------------
def scrape_mubasher_news(max_articles=10) -> List[Dict]:
    """Scrape financial news from Mubasher"""
    print("[📰] Scraping Mubasher News...")
    base_url = "https://english.mubasher.info"
    try:
        resp = requests.get(f"{base_url}/news", timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        
        for card in soup.select('.news-card')[:max_articles]:
            title = card.select_one('a.title').text.strip()
            link = base_url + card.select_one('a.title')['href']
            date = card.select_one('.date').text.strip() if card.select_one('.date') else str(datetime.today().date())
            
            articles.append({
                'title': title,
                'link': link,
                'source': 'Mubasher',
                'date': date
            })
        return articles
    except Exception as e:
        print(f"[❌] Mubasher scrape failed: {str(e)}")
        return []

def scrape_almal_news(max_articles=10) -> List[Dict]:
    """Scrape Arabic financial news from Al Mal"""
    print("[📰] Scraping Al Mal News...")
    url = "https://almalnews.com/category/بورصة/"
    try:
        resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        
        for item in soup.select('article')[:max_articles]:
            title = item.select_one('h2.entry-title').text.strip()
            link = item.select_one('a')['href']
            date = item.select_one('time.entry-date').text.strip() if item.select_one('time.entry-date') else str(datetime.today().date())
            
            articles.append({
                'title': title,
                'link': link,
                'source': 'Al Mal',
                'date': date,
                'language': 'ar'
            })
        return articles
    except Exception as e:
        print(f"[❌] Al Mal scrape failed: {str(e)}")
        return []

def scrape_google_news(query="Egypt stock market", max_results=10) -> List[Dict]:
    """Fetch news via Google News RSS"""
    print("[📰] Scraping Google News...")
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")
        return [{
            'title': entry.title,
            'link': entry.link,
            'source': 'Google News',
            'date': entry.published
        } for entry in feed.entries[:max_results]]
    except Exception as e:
        print(f"[❌] Google News scrape failed: {str(e)}")
        return []

# -------------------------------
# 🧠 GEMINI PROCESSING
# -------------------------------
def process_with_gemini(article: Dict) -> Dict:
    """Enhance article with Gemini analysis"""
    try:
        # Extract full article text
        news_article = Article(article['link'])
        news_article.download()
        news_article.parse()
        full_text = f"{article['title']}\n\n{news_article.text}"
        
        # Prepare prompt for Gemini
        prompt = f"""
        Analyze this financial news article and extract:
        1. Summary (3 sentences)
        2. All mentioned stock tickers (Egyptian market)
        3. Overall sentiment (positive/negative/neutral)
        4. Key financial topics
        5. Potential market impact (high/medium/low)
        
        Format as JSON with keys: summary, tickers, sentiment, topics, impact.
        
        Article:
        {full_text[:15000]}  # Truncate to avoid token limits
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        gemini_data = json.loads(response.text.strip("```json").strip())
        
        # Enhance original article
        article.update({
            'text': full_text,
            'summary': gemini_data.get('summary', ''),
            'tickers': gemini_data.get('tickers', []),
            'sentiment': gemini_data.get('sentiment', 'neutral'),
            'topics': gemini_data.get('topics', []),
            'impact': gemini_data.get('impact', 'medium'),
            'processed_at': datetime.now().isoformat()
        })
        return article
        
    except (ArticleException, json.JSONDecodeError, ValueError) as e:
        print(f"[⚠️] Processing failed for '{article['title']}': {str(e)}")
        return article
    except Exception as e:
        print(f"[❌] Gemini processing error: {str(e)}")
        return article

# -------------------------------
# 💾 DATA MANAGEMENT
# -------------------------------
def save_data(data: Dict, filename: str):
    """Save data to JSON file"""
    os.makedirs('data', exist_ok=True)
    path = f"data/{filename}"
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[💾] Data saved to {path}")

# -------------------------------
# 🚀 MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Egyptian Stock Intelligence System\n")
    
    # Step 1: Scrape market data
    price_data = scrape_yahoo_finance(PRICE_TICKERS)
    press_releases = scrape_egx_press_releases()
    
    # Step 2: Scrape news articles
    all_news = []
    all_news += scrape_mubasher_news()
    all_news += scrape_almal_news()
    all_news += scrape_google_news()
    all_news += press_releases
    
    # Step 3: Process with Gemini
    print("\n[🧠] Processing articles with Gemini...")
    processed_news = []
    for i, article in enumerate(all_news):
        print(f"  Processing {i+1}/{len(all_news)}: {article['title'][:50]}...")
        processed = process_with_gemini(article)
        processed_news.append(processed)
        time.sleep(1)  # Rate limiting
    
    # Step 4: Save results
    save_data(price_data, "market_data.json")
    save_data(processed_news, "processed_news.json")
    
    print("\n✅ Operation Complete!")
    print(f"- Collected {len(price_data)} tickers")
    print(f"- Processed {len(processed_news)} articles")
    print("- Files saved to /data directory")
    
    
def init_scraper():
    print("📈 Stock news scraper initialized")
