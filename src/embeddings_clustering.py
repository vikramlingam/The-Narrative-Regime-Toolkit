import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logger

logger = setup_logger("embeddings_clustering")

"""
MODULE: embeddings_clustering.py
PURPOSE: Converts text narratives into vector embeddings and clusters them into regimes.
USAGE:
    - Called by `run_pipeline.py`.
    - Uses `sentence-transformers` for embeddings and `KMeans` for clustering.
"""

def compute_narrative_embeddings(narratives_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Computes embeddings for the daily narratives.
    """
    logger.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Combine label and summary for richer context
    text_data = narratives_df['narrative_label'] + ". " + narratives_df['summary_paragraph']
    text_data = text_data.fillna("").tolist()
    
    logger.info(f"Computing embeddings for {len(text_data)} items...")
    embeddings = model.encode(text_data, show_progress_bar=True)
    
    return embeddings

def cluster_regimes(embeddings: np.ndarray, n_clusters: int = 6) -> np.ndarray:
    """
    Clusters embeddings into regimes using KMeans.
    """
    logger.info(f"Clustering into {n_clusters} regimes...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regime_ids = kmeans.fit_predict(embeddings)
    return regime_ids

def plot_regime_embedding_space(embeddings: np.ndarray, regime_ids: np.ndarray, output_path: str):
    """
    Plots the 2D projection of embeddings colored by regime.
    """
    logger.info("Projecting to 2D for visualization...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=regime_ids, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Regime ID')
    plt.title('Market Regimes in Narrative Embedding Space (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved plot to {output_path}")

if __name__ == "__main__":
    # --- TESTING ONLY ---
    pass
