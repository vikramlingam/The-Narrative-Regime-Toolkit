import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.utils import setup_logger

logger = setup_logger("llm_narratives")

"""
MODULE: llm_narratives.py
PURPOSE: Uses an LLM (Grok via OpenRouter) to summarize daily news into structured narratives.
USAGE:
    - Called by `run_pipeline.py` after news fetching.
    - Consumes `raw_news.csv` and produces `daily_narratives.csv`.
NOTE: This module relies on the OPENROUTER_API_KEY in .env.
"""

def call_grok(messages: list[dict]) -> str:
    """
    Calls OpenRouter API with Grok model.
    """
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/regime-narratives", # Optional
    }
    
    payload = {
        "model": config.GROK_MODEL,
        "messages": messages,
        "temperature": 0.3, # Low temp for factual summary
        "response_format": {"type": "json_object"} # Request JSON output
    }
    
    try:
        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling Grok: {e}")
        return ""

def build_daily_narrative_prompt(date: str, headlines: list[str], summaries: list[str]) -> list[dict]:
    """
    Constructs the prompt for Grok.
    """
    news_text = ""
    for h, s in zip(headlines, summaries):
        news_text += f"Headline: {h}\nSummary: {s}\n\n"
        
    system_prompt = """You are a financial research assistant. You read daily market news and summarize the dominant market narrative in a concise, research-friendly way.
    Return the result as a valid JSON object with keys: "date", "narrative_label", "bullet_points", "summary_paragraph"."""
    
    user_prompt = f"""Given the following news headlines and summaries for DATE = {date}, identify the single dominant market narrative and produce a structured response.

    Requirements:
    1. narrative_label: a short 3–6 word label (e.g., "Fed tightening fears dominate").
    2. bullet_points: 3–5 strings describing main themes, macro drivers, and sectors.
    3. summary_paragraph: A short paragraph (max 120 words) describing the overall narrative.
    
    News items:
    {news_text[:15000]} 
    """ # Truncate to avoid context limit if necessary, though Grok has large context.
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def generate_daily_narratives(news_df: pd.DataFrame, max_headlines_per_day: int = 40) -> pd.DataFrame:
    """
    Generates narratives for each day in the news DataFrame.
    """
    if news_df.empty:
        logger.warning("News DataFrame is empty.")
        return pd.DataFrame()
        
    # Ensure date is string for grouping
    news_df['date_str'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')
    
    daily_groups = news_df.groupby('date_str')
    results = []
    
    # Check for existing data to resume
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'daily_narratives.csv')
    existing_dates = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            existing_dates = set(existing_df['date'].astype(str))
            results = existing_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not read existing narratives: {e}")

    logger.info(f"Generating narratives for {len(daily_groups)} days...")
    
    for date_str, group in tqdm(daily_groups):
        if date_str in existing_dates:
            continue
            
        # Limit headlines
        group = group.head(max_headlines_per_day)
        
        headlines = group['headline'].tolist()
        summaries = group['summary'].tolist()
        
        messages = build_daily_narrative_prompt(date_str, headlines, summaries)
        
        response_content = call_grok(messages)
        
        if response_content:
            try:
                data = json.loads(response_content)
                # Ensure keys exist
                results.append({
                    "date": date_str,
                    "narrative_label": data.get("narrative_label", ""),
                    "bullet_points": str(data.get("bullet_points", [])), # Store as string representation
                    "summary_paragraph": data.get("summary_paragraph", "")
                })
                
                # Save incrementally
                pd.DataFrame(results).to_csv(output_path, index=False)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for {date_str}: {response_content[:100]}...")
        
        time.sleep(0.5) # Rate limit nice-ness
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # --- TESTING ONLY ---
    # This block is only executed when running this file directly.
    # It checks if `raw_news.csv` exists and tries to generate narratives for it.
    # This is NOT the main entry point for the pipeline.
    logger.info("Running llm_narratives in TEST mode...")
    
    # Load raw news if exists
    raw_news_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_news.csv')
    if os.path.exists(raw_news_path):
        df = pd.read_csv(raw_news_path)
        # Test with a small sample if file is huge, or just run it.
        # Here we just pass the dataframe.
        generate_daily_narratives(df)
    else:
        logger.warning("No raw_news.csv found. Run news_fetcher.py first.")
