import pandas as pd
import datetime
from tavily import TavilyClient
import time
from tqdm import tqdm
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.utils import setup_logger

logger = setup_logger("news_fetcher")

"""
MODULE: news_fetcher.py
PURPOSE: Fetches daily financial news headlines and summaries.
USAGE: 
    - Called by `run_pipeline.py` with specific date ranges.
    - Can be run directly for testing purposes (see `if __name__ == "__main__":` block).
NOTE: The dates used in the `if __name__ == "__main__":` block are SAMPLES for testing only.
      Actual execution dates are controlled by arguments passed to `run_pipeline.py`.
"""

def fetch_news(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches financial news headlines and summaries for a given date range using Tavily API.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'source', 'headline', 'summary', 'url']
    """
    if not config.TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY is missing. Cannot fetch news.")
        return pd.DataFrame()

    tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
    
    # Generate date range
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    delta = end - start
    
    all_news = []
    
    logger.info(f"Fetching news from {start_date} to {end_date}...")
    
    for i in tqdm(range(delta.days + 1)):
        current_date = start + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        query = f"financial markets news S&P 500 stock market {date_str}"
        
        try:
            # Tavily search
            response = tavily.search(query=query, search_depth="basic", max_results=10)
            
            for result in response.get('results', []):
                all_news.append({
                    'date': current_date, # Keep as date object
                    'source': result.get('url', 'unknown'), # Use URL as proxy for source
                    'headline': result.get('title', ''),
                    'summary': result.get('content', ''),
                    'url': result.get('url', '')
                })
            
            # Rate limit handling (simple sleep)
            time.sleep(1) 
            
        except Exception as e:
            logger.error(f"Error fetching news for {date_str}: {e}")
            
    df = pd.DataFrame(all_news)
    
    if df.empty:
        logger.warning("No news fetched.")
        return df

    # Deduplicate
    initial_len = len(df)
    df.drop_duplicates(subset=['date', 'headline'], inplace=True)
    logger.info(f"Fetched {len(df)} news items (deduplicated from {initial_len}).")
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_news.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved news to {output_path}")
    
    return df

if __name__ == "__main__":
    # --- TESTING ONLY ---
    # This block is only executed when running this file directly (e.g., `python -m src.news_fetcher`).
    # It is used to verify that the module works correctly on a small sample.
    # The dates below are NOT used when running the full pipeline via `run_pipeline.py`.
    logger.info("Running news_fetcher in TEST mode...")
    fetch_news("2023-01-01", "2023-01-03")
