import pandas as pd
import os
import sys
import argparse
from datetime import datetime

# Add src to path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
from src.news_fetcher import fetch_news
from src.llm_narratives import generate_daily_narratives
from src.embeddings_clustering import compute_narrative_embeddings, cluster_regimes, plot_regime_embedding_space
from src.market_data import fetch_market_data
from src.regime_analysis import analyze_regimes
from src.utils import setup_logger

from src.visualizations import (
    plot_regime_timeline, 
    plot_pca_embeddings, 
    plot_forward_vol_boxplot, 
    plot_regime_transition_matrix,
    plot_roc_curve
)

logger = setup_logger("pipeline_runner")

def main():
    parser = argparse.ArgumentParser(description="Run the Regime Narratives Pipeline")
    parser.add_argument("--start-date", type=str, default="2024-11-24", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-11-24", help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Market ticker (default: ^GSPC)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of regime clusters")
    
    args = parser.parse_args()
    
    start_date = args.start_date
    end_date = args.end_date
    ticker = args.ticker
    n_clusters = args.clusters
    
    logger.info(f"Starting pipeline run from {start_date} to {end_date} for {ticker}")
    
    # 1. Data Ingestion: News
    logger.info("Step 1: Fetching News...")
    news_df = fetch_news(start_date, end_date)
    if news_df.empty:
        logger.error("No news found. Exiting.")
        return

    # 2. LLM Narrative Extraction
    logger.info("Step 2: Generating Narratives...")
    narratives_df = generate_daily_narratives(news_df)
    if narratives_df.empty:
        logger.error("No narratives generated. Exiting.")
        return

    # 3. Embeddings and Clustering
    logger.info("Step 3: Computing Embeddings and Clustering...")
    embeddings = compute_narrative_embeddings(narratives_df)
    regime_ids = cluster_regimes(embeddings, n_clusters=n_clusters)
    narratives_df['regime_id'] = regime_ids
    
    # Save intermediate result
    narratives_df.to_csv(os.path.join("data", "daily_narratives_with_regimes.csv"), index=False)
    
    # Plot: PCA Embeddings
    pca_path = plot_pca_embeddings(embeddings, regime_ids)
    logger.info(f"PCA plot saved to {pca_path}")

    # 4. Market Data Alignment
    logger.info("Step 4: Fetching Market Data...")
    market_df = fetch_market_data(ticker, start_date, end_date)
    if market_df.empty:
        logger.error("No market data found. Exiting.")
        return

    # 5. Regime Analysis & Validation
    logger.info("Step 5: Analyzing Regimes & Validation...")
    merged_df, validation_results = analyze_regimes(narratives_df, market_df)
    
    if merged_df is not None and not merged_df.empty:
        output_path = os.path.join("data", "merged_regimes_analysis.csv")
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
        # Visualizations
        logger.info("Generating visualizations...")
        
        # Timeline
        tl_path = plot_regime_timeline(merged_df)
        logger.info(f"Timeline plot saved to {tl_path}")
        
        # Forward Volatility Boxplot
        vol_path = plot_forward_vol_boxplot(merged_df)
        logger.info(f"Volatility boxplot saved to {vol_path}")
        
        # Transition Matrix
        tm_path = plot_regime_transition_matrix(merged_df)
        logger.info(f"Transition matrix saved to {tm_path}")
        
        # ROC Curve (if validation succeeded)
        if validation_results:
            roc_path = plot_roc_curve(validation_results['y_test'], validation_results['y_prob'])
            logger.info(f"ROC curve saved to {roc_path}")
            logger.info(f"Validation AUC: {validation_results['auc']:.3f}")
        else:
            logger.warning("Validation skipped (insufficient data).")
            
    else:
        logger.warning("Analysis produced no results (likely insufficient data overlap).")

if __name__ == "__main__":
    main()
