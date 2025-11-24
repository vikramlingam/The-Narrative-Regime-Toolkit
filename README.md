# Regime Narratives: Detecting Market Regimes from News

## Overview
This project implements a research-grade pipeline to detect "market regimes" (e.g., "Inflation Fears", "Tech Boom") from daily financial news using Large Language Models (LLMs) and clustering techniques. It then tests whether these narrative regimes have predictive power for future market volatility.

**Goal**: To move beyond price-based regime detection (like Hidden Markov Models on returns) and instead use *narrative* signals derived from news text to identify market states.

## How It Works
The pipeline consists of 5 main steps:
1.  **News Ingestion**: Collects daily financial news headlines and summaries (via Tavily API).
2.  **Narrative Extraction**: Uses an LLM (Grok via OpenRouter) to read each day's news and summarize the dominant narrative into a concise label and paragraph.
3.  **Embedding & Clustering**: Converts these daily narrative texts into vector embeddings (using `sentence-transformers`) and clusters them into distinct groups (Regimes).
4.  **Market Alignment**: Aligns these regimes with S&P 500 market data (prices, volatility).
5.  **Analysis & Validation**:
    *   Visualizes regimes over time.
    *   Tests if regimes predict future high volatility using a time-series validation strategy (avoiding lookahead bias).

## Project Structure
```
regime_narratives/
├── run_pipeline.py         # <--- MAIN ENTRY POINT. Run this to execute the project.
├── config.py               # Configuration (API keys, defaults).
├── requirements.txt        # Python dependencies.
├── .env                    # Environment variables (API keys).
├── data/                   # Stores generated CSV data.
│   ├── raw_news.csv
│   ├── daily_narratives.csv
│   ├── daily_narratives_with_regimes.csv
│   ├── market_data.csv
│   └── merged_regimes_analysis.csv
├── plots/                  # Stores generated visualizations (timestamped).
└── src/                    # Source code modules.
    ├── news_fetcher.py         # Fetches news from Tavily.
    ├── llm_narratives.py       # Calls LLM to generate narratives.
    ├── embeddings_clustering.py# Computes embeddings and clusters regimes.
    ├── market_data.py          # Fetches S&P 500 data via yfinance.
    ├── regime_analysis.py      # Merges data and runs validation models.
    ├── visualizations.py       # Generates professional plots.
    └── utils.py                # Helper functions (logging).
```

## How to Run
### 1. Setup
Ensure you have Python 3.10+ and install dependencies:
```bash
pip install -r requirements.txt
```
Create a `.env` file with your API keys:
```
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

### 2. Execute Pipeline
The entire process is controlled by `run_pipeline.py`. You can specify the date range and other parameters via command-line arguments.

**Example Run:**
```bash
python run_pipeline.py --start-date 2018-01-01 --end-date 2024-12-31 --ticker ^GSPC --clusters 4
```

**Arguments:**
*   `--start-date`: Start date for analysis (YYYY-MM-DD).
*   `--end-date`: End date for analysis (YYYY-MM-DD).
*   `--ticker`: Market ticker symbol (default: `^GSPC` for S&P 500).
*   `--clusters`: Number of regimes to identify (default: 4).

### 3. Output
*   **Console**: You will see logs for each step (Fetching news -> Generating narratives -> Clustering -> Analysis).
*   **Data**: CSV files are saved in `data/`.
*   **Plots**: Visualizations are saved in `plots/` with timestamps (e.g., `regime_timeline_20251124_183000.png`).

## Code Explanation
*   **`src/` Files**: Each file in `src/` is a module responsible for one part of the pipeline.
    *   **Note on Dates**: You might see hardcoded dates (e.g., "2023-01-01") inside the `if __name__ == "__main__":` blocks of these files. These are **ONLY for testing** individual modules during development. They are **NOT** used when you run `run_pipeline.py`. The pipeline uses the dates you provide in the command line arguments.
*   **Validation Strategy**: The `regime_analysis.py` module implements a strict **time-series split** (training on the first 70% of data, testing on the last 30%) to evaluate if the identified regimes can predict whether future volatility (next 5 days) will be high. This ensures the results are robust and realistic.

## Visualizations
The pipeline generates the following plots in `plots/`:
1.  **Regime Timeline**: Shows the S&P 500 price history with points colored by the active regime.
2.  **PCA Embeddings**: A 2D scatter plot showing how daily narratives cluster together in semantic space.
3.  **Forward Volatility Boxplot**: Shows the distribution of future volatility for each regime (e.g., does Regime 1 consistently lead to higher risk?).
4.  **Transition Matrix**: A heatmap showing the probability of switching from one regime to another.
5.  **ROC Curve**: Shows the performance of the predictive model on the test set.
