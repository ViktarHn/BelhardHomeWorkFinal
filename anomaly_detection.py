import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import logging
import os
import pickle
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('anomaly_detection.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, file_path: str = 'estat_tec00107_filtered_en.csv'):
        self.file_path = file_path
        self.df: pd.DataFrame = None
        self.results: Dict[str, Any] = {}
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
    
    def load_data(self) -> bool:
        from data_loading import load_data
        logger.info("Loading data...")
        self.df = load_data(self.file_path)
        if self.df is None or self.df.empty:
            logger.warning("No data loaded or empty DataFrame")
            return False
        self.df = self.df.sort_values('TIME_PERIOD')
        logger.info(f"Loaded {len(self.df)} records")
        return True
    
    def detect_anomalies(self) -> bool:
        if not self.load_data():
            return False
        logger.info("Detecting anomalies...")
        if self.df['OBS_VALUE'].dropna().empty:
            logger.warning("No valid OBS_VALUE data for anomaly detection")
            return False
        X = self.df[['OBS_VALUE']].dropna().values
        
        methods = {
            'IsolationForest': IsolationForest(contamination=0.05, random_state=42),
            'LOF': LocalOutlierFactor(contamination=0.05, novelty=False),
            'OneClassSVM': OneClassSVM(nu=0.05),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        for name, model in methods.items():
            if name in ['LOF', 'DBSCAN']:
                labels = model.fit_predict(X)
            else:
                labels = model.fit_predict(X)
            self.df[f'{name}_anomaly'] = np.nan
            self.df.loc[self.df['OBS_VALUE'].notna(), f'{name}_anomaly'] = np.where(labels == -1, 1, 0)
            self.results[name] = {'model': model, 'anomaly_col': f'{name}_anomaly'}
            logger.info(f"{name} anomalies detected")
        
        self._save_models()
        return True
    
    def _save_models(self) -> None:
        for name, result in self.results.items():
            model_path = f'results/models/{name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        logger.info("Models saved")
    
    def visualize_results(self) -> None:
        if not self.results:
            logger.warning("No results to visualize")
            return
        fig = px.scatter(self.df, x='TIME_PERIOD', y='OBS_VALUE', title='Detected Anomalies in FDI Data')
        for name, result in self.results.items():
            anomalies = self.df[self.df[result['anomaly_col']] == 1]
            fig.add_scatter(x=anomalies['TIME_PERIOD'], y=anomalies['OBS_VALUE'], mode='markers', name=f'{name} Anomalies')
        fig.write_html('results/plots/anomalies_detection.html')
        fig.show()
        logger.info("Anomaly plot saved and displayed")
        self.df.to_csv('results/anomalies_marked.csv', index=False)
        logger.info("Data with anomaly labels saved")

def main():
    detector = AnomalyDetector()
    if detector.load_data() and detector.detect_anomalies():
        detector.visualize_results()

if __name__ == "__main__":
    main()