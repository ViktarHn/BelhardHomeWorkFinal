import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from logging.handlers import RotatingFileHandler
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from scipy import stats

logger = logging.getLogger('eda')
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = RotatingFileHandler('eda.log', maxBytes=1e6, backupCount=3)
fh.setFormatter(formatter)
logger.addHandler(fh)

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: Dict[str, Any] = {
            'metadata': {'analysis_date': datetime.now().isoformat(), 'data_source': 'estat_tec00107_filtered_en.csv', 'stats': {}},
            'missing_values': {},
            'outliers': {},
            'correlations': {},
            'trends': {},
            'recommendations': []
        }
    
    def _save_plot(self, fig: go.Figure, filename: str) -> None:
        os.makedirs('plots', exist_ok=True)
        path = f"plots/{filename}.html"
        fig.write_html(path)
        fig.show()
        logger.info(f"Plot saved to {path}")
    
    def _add_recommendation(self, message: str) -> None:
        self.results['recommendations'].append(message)
        logger.info(f"Recommendation: {message}")
    
    def _safe_describe(self, series: pd.Series) -> Dict[str, float]:
        try:
            return {
                'count': series.count(), 'mean': series.mean(), 'std': series.std(),
                'min': series.min(), '25%': series.quantile(0.25), '50%': series.median(),
                '75%': series.quantile(0.75), 'max': series.max()
            }
        except Exception as e:
            logger.error(f"Error computing stats: {str(e)}")
            return {}

    def basic_statistics(self) -> Dict[str, Any]:
        logger.info("Analyzing basic statistics...")
        self.results['metadata'].update({
            'total_rows': len(self.df), 'total_columns': len(self.df.columns),
            'time_period': {'start': str(self.df['TIME_PERIOD'].min()), 'end': str(self.df['TIME_PERIOD'].max())},
            'unique_countries': self.df['geo'].nunique()
        })
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.results['metadata']['stats'][col] = self._safe_describe(self.df[col])
            if self.df[col].isnull().all():
                self._add_recommendation(f"Column {col} is completely empty - consider removing")
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.results['metadata']['stats'][col] = {
                'count': self.df[col].count(), 'unique': self.df[col].nunique(),
                'top': self.df[col].mode().iloc[0] if not self.df[col].empty else None,
                'freq': self.df[col].value_counts().iloc[0] if not self.df[col].empty else None
            }
        return self.results['metadata']['stats']

    def analyze_missing_values(self) -> Dict[str, Any]:
        logger.info("Analyzing missing values...")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        self.results['missing_values'] = {'count': missing.to_dict(), 'percentage': missing_pct.to_dict()}
        fig = px.imshow(self.df.isnull(), title="Missing Values Heatmap")
        self._save_plot(fig, 'missing_values_heatmap')
        return self.results['missing_values']

    def detect_outliers(self, threshold: float = 3.5) -> Dict[str, Any]:
        logger.info("Detecting outliers...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 10:
                logger.warning(f"Not enough data for {col} (n={len(data)})")
                continue
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > threshold]
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = data[(data < (q1 - 1.5*iqr)) | (data > (q3 + 1.5*iqr))]
            outliers[col] = {
                'z_score': {'count': len(z_outliers), 'indices': z_outliers.index.tolist(), 'values': z_outliers.values.tolist()},
                'iqr': {'count': len(iqr_outliers), 'indices': iqr_outliers.index.tolist(), 'values': iqr_outliers.values.tolist()}
            }
            fig = px.box(self.df, y=col, title=f"Outliers in {col}")
            self._save_plot(fig, f'outliers_boxplot_{col}')
        self.results['outliers'] = outliers
        return outliers

    def analyze_distributions(self) -> None:
        logger.info("Analyzing distributions...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 5:
                continue
            fig = px.histogram(self.df, x=col, nbins=30, title=f"Distribution of {col}", marginal="rug")
            self._save_plot(fig, f'distribution_{col}')
            stat, p = stats.shapiro(data)
            self.results['metadata']['stats'][col].update({
                'normality_test': {'shapiro_stat': float(stat), 'shapiro_p': float(p), 'is_normal': bool(p > 0.05)}
            })

    def analyze_temporal_trends(self) -> None:
        logger.info("Analyzing temporal trends...")
        if 'TIME_PERIOD' not in self.df.columns:
            logger.warning("Missing TIME_PERIOD column")
            return
        self.df['year'] = self.df['TIME_PERIOD'].dt.year
        yearly_stats = self.df.groupby('year')['OBS_VALUE'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        self.results['trends']['yearly_stats'] = yearly_stats.to_dict()
        fig = px.line(yearly_stats.reset_index(), x='year', y='mean', error_y='std', title='Mean FDI by Year with Confidence Interval')
        self._save_plot(fig, 'yearly_trends')

    def analyze_correlations(self) -> None:
        logger.info("Analyzing correlations...")
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        if len(numeric_df.columns) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return
        corr_matrix = numeric_df.corr()
        self.results['correlations']['matrix'] = corr_matrix.to_dict()
        fig = px.imshow(corr_matrix, text_auto='.2f', title="Correlation Matrix", color_continuous_scale='RdBu_r')
        self._save_plot(fig, 'correlation_matrix')

    def save_results(self) -> None:
        def default_serializer(obj):
            if isinstance(obj, (np.int64, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)
        with open('eda_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False, default=default_serializer)
        logger.info("Results saved to eda_results.json")

    def run_full_analysis(self) -> Dict[str, Any]:
        if self.df.empty:
            logger.warning("Empty DataFrame provided for EDA")
            return self.results
        logger.info("Running full EDA analysis")
        self.basic_statistics()
        self.analyze_missing_values()
        self.detect_outliers()
        self.analyze_distributions()
        self.analyze_temporal_trends()
        self.analyze_correlations()
        self.save_results()
        logger.info("EDA analysis completed")
        return self.results

if __name__ == "__main__":
    from data_loading import DataLoader
    loader = DataLoader()
    df = loader.load_data()
    if df is not None and not df.empty:
        analyzer = EDAAnalyzer(df)
        results = analyzer.run_full_analysis()
        print("\n=== SUMMARY RESULTS ===")
        print(f"Rows analyzed: {len(df)}")