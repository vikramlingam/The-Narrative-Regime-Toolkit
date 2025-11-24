import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
from datetime import datetime
import sys

# Setup professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

"""
MODULE: visualizations.py
PURPOSE: Generates and saves professional, paper-ready plots.
USAGE:
    - Called by `run_pipeline.py` after analysis.
    - Saves plots to the `plots/` directory with timestamped filenames.
"""

def _get_plot_path(base_name: str) -> str:
    """
    Generates a timestamped path for saving plots.
    """
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.png"
    return os.path.join(plots_dir, filename)

def plot_regime_timeline(df: pd.DataFrame) -> str:
    """
    Plots the index price with regimes colored.
    
    Args:
        df: DataFrame with 'date', 'close', 'regime_id'
        
    Returns:
        Path to saved plot.
    """
    if df.empty:
        return ""
        
    plt.figure(figsize=(12, 6))
    
    # Create a scatter plot for regimes to handle non-contiguous segments better than line coloring
    # Or use colored background bands. Let's use colored scatter points on top of a thin line.
    
    plt.plot(df['date'], df['close'], color='gray', alpha=0.5, linewidth=1, label='Index Price')
    
    scatter = plt.scatter(df['date'], df['close'], c=df['regime_id'], cmap='tab10', s=10, zorder=2)
    
    plt.title('Market Regimes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index Price')
    plt.colorbar(scatter, label='Regime ID')
    plt.legend()
    plt.tight_layout()
    
    path = _get_plot_path("regime_timeline")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_pca_embeddings(embeddings: np.ndarray, regime_ids: np.ndarray) -> str:
    """
    Plots 2D PCA of embeddings colored by regime.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=regime_ids, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Regime ID')
    plt.title('Narrative Embeddings (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    path = _get_plot_path("pca_embeddings")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_forward_vol_boxplot(df: pd.DataFrame) -> str:
    """
    Boxplot of forward volatility by regime.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter NaNs
    plot_data = df.dropna(subset=['vol_next_5d'])
    
    if plot_data.empty:
        return ""
        
    sns.boxplot(data=plot_data, x='regime_id', y='vol_next_5d', palette='tab10')
    plt.title('Forward 5-Day Volatility by Regime')
    plt.xlabel('Regime ID')
    plt.ylabel('Annualized Volatility (Next 5 Days)')
    
    path = _get_plot_path("forward_vol_by_regime")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_regime_transition_matrix(df: pd.DataFrame) -> str:
    """
    Heatmap of regime transition probabilities.
    """
    if 'regime_id' not in df.columns:
        return ""
        
    # Calculate transitions
    df = df.sort_values('date')
    current_regime = df['regime_id']
    next_regime = df['regime_id'].shift(-1)
    
    transitions = pd.crosstab(current_regime, next_regime, normalize='index')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(transitions, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Transition Probability'})
    plt.title('Regime Transition Matrix')
    plt.xlabel('Next Regime')
    plt.ylabel('Current Regime')
    
    path = _get_plot_path("regime_transition_matrix")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_roc_curve(y_test, y_prob) -> str:
    """
    Plots ROC curve for the high volatility prediction model.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: Predicting High Volatility with Regimes')
    plt.legend(loc="lower right")
    
    path = _get_plot_path("roc_high_vol_regime")
    plt.savefig(path, dpi=300)
    plt.close()
    return path
