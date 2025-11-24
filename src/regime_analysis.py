import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logger

logger = setup_logger("regime_analysis")

"""
MODULE: regime_analysis.py
PURPOSE: Merges narrative regimes with market data and performs statistical analysis/validation.
USAGE:
    - Called by `run_pipeline.py`.
    - Performs time-series validation (Train/Test split) to test predictive power of regimes.
"""

def perform_regime_validation(merged_df: pd.DataFrame, test_size: float = 0.3):
    """
    Performs time-series validation of regime predictive power.
    
    Strategy:
    1. Sort by date.
    2. Split into first (1-test_size)% as train, last (test_size)% as test.
    3. Train Logistic Regression to predict 'high_vol_target' using 'regime_id'.
    4. Evaluate on test set.
    """
    # Prepare data
    # Target: High Volatility next 5 days (Top 33% of entire dataset for consistency threshold)
    threshold = merged_df['vol_next_5d'].quantile(0.67)
    merged_df['high_vol_target'] = (merged_df['vol_next_5d'] > threshold).astype(int)
    
    # Features: Regime ID (one-hot) + Current Volatility
    features = pd.get_dummies(merged_df['regime_id'], prefix='regime')
    features['current_vol'] = merged_df['vol_20d'].fillna(0)
    
    X = features
    y = merged_df['high_vol_target']
    
    # Drop NaNs (rows where target or features are missing)
    valid_idx = X.index[X['current_vol'].notna() & y.notna()]
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    if len(X) < 50:
        logger.warning("Not enough data for validation (<50 samples).")
        return None
        
    # Time-series split (no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    
    logger.info(f"Validation AUC: {auc_score:.3f}")
    
    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "auc": auc_score,
        "report": report
    }

def analyze_regimes(narratives_with_regimes: pd.DataFrame, market_df: pd.DataFrame):
    """
    Merges data, computes stats, and runs validation.
    Returns merged DataFrame and validation results.
    """
    # Merge
    narratives_with_regimes['date'] = pd.to_datetime(narratives_with_regimes['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    
    merged = pd.merge(narratives_with_regimes, market_df, on='date', how='inner')
    
    if merged.empty:
        logger.error("Merged DataFrame is empty. Check date overlaps.")
        return None, None
    
    logger.info(f"Merged data has {len(merged)} rows.")
    
    # 1. Descriptive Stats per Regime
    stats = merged.groupby('regime_id')[['vol_next_5d', 'ret_next_5d', 'vol_20d']].mean()
    logger.info("Regime Statistics:\n" + str(stats))
    
    # 2. Validation
    validation_results = perform_regime_validation(merged)
        
    return merged, validation_results

if __name__ == "__main__":
    # --- TESTING ONLY ---
    pass
