import pandas as pd
import logging
import os
from urllib.request import urlretrieve
from typing import Optional, Dict, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data_loading.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataLoader:
    CONFIDENTIAL_FLAG = 'C'
    EXPECTED_COLUMNS = ['geo', 'TIME_PERIOD', 'OBS_VALUE', 'CONF_STATUS']
    DEFAULT_CONFIG = {
        'encoding': 'utf-8',
        'date_columns': ['TIME_PERIOD'],
        'source': 'local',
        'handle_confidential': True
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.df: Optional[pd.DataFrame] = None
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        missing_columns = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False
        if not pd.api.types.is_numeric_dtype(df['OBS_VALUE']):
            logger.warning("OBS_VALUE contains non-numeric values")
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
            df.drop_duplicates(inplace=True)
        return True
    
    def handle_confidential_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'CONF_STATUS' in df.columns and self.config.get('handle_confidential', True):
            confidential_mask = df['CONF_STATUS'] == self.CONFIDENTIAL_FLAG
            n_confidential = confidential_mask.sum()
            df.loc[confidential_mask, 'OBS_VALUE'] = np.nan
            logger.info(f"Handled {n_confidential} confidential values (replaced with NaN)")
            df['OBS_VALUE'] = df['OBS_VALUE'].interpolate(method='linear').ffill().bfill()
            if df['OBS_VALUE'].isna().all():
                logger.warning("All OBS_VALUE are NaN after interpolation; filling with 0")
                df['OBS_VALUE'] = df['OBS_VALUE'].fillna(0)
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> None:
        missing_values = df['OBS_VALUE'].isna().sum()
        logger.info(f"Found {missing_values} missing values in OBS_VALUE after processing")
        min_year = df['TIME_PERIOD'].min()
        max_year = df['TIME_PERIOD'].max()
        logger.info(f"Data covers years: {min_year} to {max_year}")
        countries = df['geo'].nunique()
        logger.info(f"Data contains {countries} unique countries")
        if len(df) < 10:
            logger.warning(f"Dataset has only {len(df)} rows, which may be insufficient")
    
    def load_data(self, file_path: str = 'estat_tec00107_filtered_en.csv') -> Optional[pd.DataFrame]:
        try:
            if self.config['source'] == 'url':
                logger.info(f"Downloading data from URL: {file_path}")
                local_path = os.path.basename(file_path)
                urlretrieve(file_path, local_path)
                file_path = local_path
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")
                
            try:
                df = pd.read_csv(file_path, encoding=self.config['encoding'])
            except UnicodeDecodeError:
                logger.warning(f"Failed with encoding {self.config['encoding']}, trying latin-1")
                df = pd.read_csv(file_path, encoding='latin-1')
            
            for col in self.config['date_columns']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='%Y', errors='coerce')
            
            if not self.validate_data_structure(df):
                logger.error("Data structure validation failed")
                return None
                
            df = self.handle_confidential_data(df)
            self.check_data_quality(df)
            self.df = df
            logger.info(f"Data loaded successfully: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}", exc_info=True)
            return None

def load_data(file_path: str = 'estat_tec00107_filtered_en.csv', config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    loader = DataLoader(config=config)
    df = loader.load_data(file_path)
    if df is None or df.empty:
        logger.warning("No data loaded or empty DataFrame returned")
        return pd.DataFrame()
    return df

if __name__ == "__main__":
    logger.info("=== Загрузка данных FDI ===")
    df = load_data("estat_tec00107_filtered_en.csv")
    if not df.empty:
        logger.info(f"Loaded {len(df)} rows")