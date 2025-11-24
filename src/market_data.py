import pandas as pd
import yfinance as yf
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logger

logger = setup_logger("market_data")

"""
MODULE: market_data.py
PURPOSE: Fetches historical market data (OHLCV) and computes derived metrics (volatility, returns).
USAGE:
    - Called by `run_pipeline.py` to align market data with narrative regimes.
    - Uses `yfinance` to download data.
NOTE: The dates in `if __name__ == "__main__":` are SAMPLES for testing.
"""

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily OHLCV data and computes derived metrics.
    """
    logger.info(f"Fetching market data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        logger.error("No market data fetched.")
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex, usually (Price, Ticker). We just want Price.
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    
    # Normalize column names to lowercase and snake_case
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    logger.info(f"Columns found: {df.columns.tolist()}")

    
    # Rename specifically if needed, but the above should handle 'Adj Close' -> 'adj_close'
    # Just in case 'Date' became 'date' already.
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])

    # Use adj_close if available, else close
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    if price_col not in df.columns:
        logger.error(f"Price column not found. Columns: {df.columns}")
        return pd.DataFrame()
    
    # Compute returns
    df['return'] = df[price_col].pct_change()
    
    # Realized Volatility (20-day rolling std dev of returns, annualized)
    df['vol_20d'] = df['return'].rolling(window=20).std() * np.sqrt(252)
    
    # Forward metrics (what happens NEXT)
    # Forward 5-day return
    df['ret_next_5d'] = df[price_col].shift(-5) / df[price_col] - 1
    
    # Forward 5-day volatility (annualized)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df['vol_next_5d'] = df['return'].rolling(window=indexer).std() * np.sqrt(252)
    
    # Max Drawdown in next 20 days
    # We need a rolling max of the next 20 days relative to current price? 
    # Or max drawdown FROM current price over next 20 days?
    # Let's do: Max % drop from the highest high in the next 20 days to the lowest low in that same window?
    # Simpler: Max drawdown within the next 20 day window.
    
    # Let's compute it by iterating or using a smart rolling apply.
    # For simplicity/speed in this context, let's just use the min return in next 20 days relative to today.
    # Actually, standard definition: Max peak-to-trough decline within the window.
    
    # We'll skip complex rolling max-dd for now and just use "Min return over next 20 days" as a proxy for downside risk.
    indexer_20 = pd.api.indexers.FixedForwardWindowIndexer(window_size=20)
    df['min_ret_next_20d'] = df['return'].rolling(window=indexer_20).min()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved market data to {output_path}")
    
    return df

if __name__ == "__main__":
    # --- TESTING ONLY ---
    # This block is only executed when running this file directly.
    # The dates below are SAMPLES to verify that yfinance fetching works.
    # They do NOT affect the main pipeline execution.
    logger.info("Running market_data in TEST mode...")
    fetch_market_data("^GSPC", "2023-01-01", "2023-06-01")
