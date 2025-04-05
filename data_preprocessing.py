import pandas as pd
import numpy as np
import plotly.express as px
import logging
import os
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import yaml
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data_processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'window_sizes': {'mean': 3, 'std': 3, 'min': 5, 'max': 5},
    'output_dir': 'processed_data',
    'fillna_methods': {'OBS_VALUE': 'interpolate', 'rolling_mean': 'zero', 'yearly_change': 'zero', 'rolling_std': 'zero'},
    'features_to_scale': ['OBS_VALUE', 'rolling_mean', 'yearly_change'],
    'visualization': True,
    'max_abs_value': 1e6,
    'save_parquet': False,
    'outlier_threshold': 3.0,
    'save_stats': True,
    'save_excel': False,
    'drop_columns': ['OBS_FLAG', 'CONF_STATUS'],
    'validation_rules': {'OBS_VALUE': {'min': -1000, 'max': 1000}, 'yearly_change': {'min': -100, 'max': 100}},
    'drop_na': False
}

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
            def deep_update(source, overrides):
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in source:
                        deep_update(source[key], value)
                    else:
                        source[key] = value
                return source
            return deep_update(DEFAULT_CONFIG.copy(), config)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG

def setup_output_dir(output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'stats'), exist_ok=True)
    logger.info(f"Directory structure created in {output_dir}")

def handle_missing_values(df: pd.DataFrame, methods: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, method in methods.items():
        if col not in df.columns:
            continue
        if method == 'interpolate':
            df[col] = df[col].interpolate(method='linear').ffill().bfill()
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'zero':
            df[col] = df[col].fillna(0)
        elif method == 'ffill':
            df[col] = df[col].ffill()
        elif method == 'bfill':
            df[col] = df[col].bfill()
    logger.info(f"Missing values handled. Remaining NaN in OBS_VALUE: {df['OBS_VALUE'].isna().sum()}")
    return df

def calculate_rolling_features(df: pd.DataFrame, window_sizes: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    window = window_sizes.get('mean', 3)
    df['rolling_mean'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['yearly_change'] = df.groupby('geo')['OBS_VALUE'].transform(
        lambda x: x.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100)
    df['yearly_change'] = df.groupby('geo')['yearly_change'].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else 0))
    window = window_sizes.get('std', 3)
    df['rolling_std'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(window, min_periods=1).std())
    window = window_sizes.get('min', 5)
    df['rolling_min'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(window, min_periods=1).min())
    window = window_sizes.get('max', 5)
    df['rolling_max'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(window, min_periods=1).max())
    logger.info("Rolling features calculated")
    return df

def validate_data(df: pd.DataFrame, rules: Dict[str, Dict[str, float]]) -> bool:
    is_valid = True
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        min_val = rule.get('min')
        max_val = rule.get('max')
        if min_val is not None and (df[col] < min_val).any():
            logger.warning(f"Found values below {min_val} in {col}")
            is_valid = False
        if max_val is not None and (df[col] > max_val).any():
            logger.warning(f"Found values above {max_val} in {col}")
            is_valid = False
    return is_valid

def create_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    fig = px.histogram(df, x='OBS_VALUE', nbins=30, title='Distribution of OBS_VALUE')
    fig.write_html(os.path.join(output_dir, 'plots', 'data_histogram.html'))
    fig.show()
    fig = px.box(df, y=['OBS_VALUE', 'rolling_mean', 'yearly_change'], title='Boxplots of Key Features')
    fig.write_html(os.path.join(output_dir, 'plots', 'data_boxplot.html'))
    fig.show()
    sample_countries = df['geo'].unique()[:5]
    fig = px.line(df[df['geo'].isin(sample_countries)], x='TIME_PERIOD', y='OBS_VALUE', color='geo', title='Sample Country Time Series')
    fig.write_html(os.path.join(output_dir, 'plots', 'sample_time_series.html'))
    fig.show()
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', title='Correlation Matrix', color_continuous_scale='RdBu_r')
    fig.write_html(os.path.join(output_dir, 'plots', 'correlation_matrix.html'))
    fig.show()
    logger.info(f"Visualizations saved and displayed from {output_dir}/plots")

def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info(f"Starting preprocessing. Initial shape: {df.shape}")
    if df.empty:
        logger.warning("Empty DataFrame provided for preprocessing")
        return df
    df = df.drop(columns=config.get('drop_columns', []), errors='ignore')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"NaN in OBS_VALUE before handling: {df['OBS_VALUE'].isna().sum()}")
    df = handle_missing_values(df, config.get('fillna_methods', {}))
    df = calculate_rolling_features(df, config.get('window_sizes', {}))
    if 'TIME_PERIOD' in df.columns:
        df['year'] = df['TIME_PERIOD'].dt.year
        df['quarter'] = df['TIME_PERIOD'].dt.quarter
    scaler = StandardScaler()
    features_to_scale = [f for f in config.get('features_to_scale', []) if f in df.columns]
    if features_to_scale:
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    logger.info(f"Preprocessing completed. Final shape: {df.shape}")
    return df

def save_processed_data(df: pd.DataFrame, output_dir: str, config: Dict[str, Any]) -> None:
    if df.empty:
        logger.warning("No data to save")
        return
    csv_path = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Data saved to CSV: {csv_path}")
    if config.get('save_stats', True):
        stats = {
            'summary_stats': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, 'stats', 'data_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Stats saved in {output_dir}/stats")

def main():
    logger.info("Starting preprocessing pipeline")
    config = load_config()
    setup_output_dir(config['output_dir'])
    from data_loading import load_data
    df = load_data()
    if df is not None and not df.empty:
        df_processed = preprocess_data(df, config)
        if config['visualization']:
            create_visualizations(df_processed, config['output_dir'])
        save_processed_data(df_processed, config['output_dir'], config)
        return df_processed
    logger.warning("No data processed")
    return pd.DataFrame()

if __name__ == "__main__":
    df_result = main()
    if not df_result.empty:
        print("\nPreprocessing Result: Rows processed =", len(df_result))