import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import logging
import pickle
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('clustering.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'max_clusters': 10,
    'random_state': 42,
    'file_path': 'estat_tec00107_filtered_en.csv',
    'fill_method': 'median',
    'dbscan_eps': 0.5,
    'dbscan_min_samples': 5
}

def find_optimal_clusters(X_scaled: np.ndarray, max_clusters: int, random_state: int) -> int:
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores.append(0)
    optimal_k = np.argmax(silhouette_scores) + 2 if silhouette_scores else 2
    logger.info(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def cluster_countries(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Starting clustering")
    if df.empty:
        logger.warning("Empty DataFrame provided for clustering")
        return pd.DataFrame(columns=['geo', 'fdi_mean', 'fdi_std', 'change_mean', 'cluster'])
    
    df_agg = df.groupby('geo').agg({
        'OBS_VALUE': ['mean', 'std'],
        'yearly_change': 'mean'
    }).reset_index()
    df_agg.columns = ['geo', 'fdi_mean', 'fdi_std', 'change_mean']
    df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    initial_rows = len(df_agg)
    if config.get('fill_method') == 'median':
        df_agg.fillna({
            'fdi_mean': df_agg['fdi_mean'].median(),
            'change_mean': df_agg['change_mean'].median(),
            'fdi_std': df_agg['fdi_std'].median()
        }, inplace=True)
    elif config.get('fill_method') == 'mean':
        df_agg.fillna({
            'fdi_mean': df_agg['fdi_mean'].mean(),
            'change_mean': df_agg['change_mean'].mean(),
            'fdi_std': df_agg['fdi_std'].mean()
        }, inplace=True)
    df_agg.dropna(subset=['fdi_mean', 'change_mean'], inplace=True)
    logger.info(f"After processing NaN, {len(df_agg)} rows remain (initially {initial_rows})")
    
    if df_agg.empty:
        logger.warning("No data available for clustering after preprocessing")
        return pd.DataFrame(columns=['geo', 'fdi_mean', 'fdi_std', 'change_mean', 'cluster'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_agg[['fdi_mean', 'change_mean']])
    
    methods = {
        'KMeans': KMeans(n_clusters=find_optimal_clusters(X_scaled, config['max_clusters'], config['random_state']), random_state=config['random_state']),
        'DBSCAN': DBSCAN(eps=config['dbscan_eps'], min_samples=config['dbscan_min_samples']),
        'Agglomerative': AgglomerativeClustering(n_clusters=3),
        'Spectral': SpectralClustering(n_clusters=3, random_state=config['random_state']),
        'GMM': GaussianMixture(n_components=3, random_state=config['random_state'])
    }
    
    results = {}
    for name, model in methods.items():
        if name == 'GMM':
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        results[name] = labels
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            logger.info(f"{name}: {len(set(labels))} clusters, Silhouette={silhouette:.3f}, Davies-Bouldin={davies_bouldin:.3f}")
    
    label_matrix = np.array([results[name] for name in results]).T
    ensemble_labels = [np.bincount(row[row >= 0]).argmax() if np.any(row >= 0) else -1 for row in label_matrix]
    df_agg['cluster'] = ensemble_labels
    if len(set(ensemble_labels)) > 1:
        ensemble_silhouette = silhouette_score(X_scaled, ensemble_labels)
        logger.info(f"Ensemble Silhouette Score: {ensemble_silhouette:.3f}")
    
    fig = px.scatter(df_agg, x='fdi_mean', y='change_mean', color='cluster', hover_data=['geo'], title='Country Clustering')
    fig.write_html('clusters.html')
    fig.show()  # Добавлено для отображения в Jupyter
    logger.info("Clustering visualization saved and displayed")
    
    for name, model in methods.items():
        with open(f'{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    logger.info("Models saved")
    
    return df_agg

if __name__ == "__main__":
    from data_loading import load_data
    from data_preprocessing import preprocess_data
    config = DEFAULT_CONFIG
    df = load_data(config['file_path'])
    df_processed = preprocess_data(df, config)
    result = cluster_countries(df_processed, config)
    if not result.empty:
        print("\nClustering Result: Rows clustered =", len(result))