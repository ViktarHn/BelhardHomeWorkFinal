import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
import os
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('time_series_analysis.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df = df[df['geo'] == country].copy()
    if df.empty:
        logger.warning(f"No data available for country: {country}")
        return df
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
    df.set_index('TIME_PERIOD', inplace=True)
    df = df.sort_index()
    df['OBS_VALUE'] = df['OBS_VALUE'].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
    df['year'] = df.index.year
    df['time_idx'] = np.arange(len(df))
    logger.info(f"Prepared data for {country} with {len(df)} rows")
    return df

def arima_forecast(train: pd.Series, test: pd.Series, steps: int) -> tuple:
    model = auto_arima(train, seasonal=False, trace=False)
    model_fit = model.fit(train)
    test_pred = model_fit.predict(n_periods=len(test))
    future_pred = model_fit.predict(n_periods=steps)
    return test_pred, future_pred

def prophet_forecast(train: pd.DataFrame, test: pd.DataFrame, steps: int) -> tuple:
    df = train.reset_index().rename(columns={'TIME_PERIOD': 'ds', 'OBS_VALUE': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps + len(test), freq='Y')
    forecast = model.predict(future)
    test_pred = forecast['yhat'].iloc[:len(test)]
    future_pred = forecast['yhat'].iloc[len(test):]
    return test_pred, future_pred

def rf_forecast(train: pd.DataFrame, test: pd.DataFrame, steps: int) -> tuple:
    X_train = train[['year', 'time_idx']]
    y_train = train['OBS_VALUE']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    test_pred = model.predict(test[['year', 'time_idx']])
    future_dates = pd.date_range(start=train.index[-1], periods=steps+1, freq='Y')[1:]
    future_df = pd.DataFrame({'year': future_dates.year, 'time_idx': np.arange(len(train), len(train)+steps)})
    future_pred = model.predict(future_df)
    return test_pred, future_pred

def sarima_forecast(train: pd.Series, test: pd.Series, steps: int) -> tuple:
    model = auto_arima(train, seasonal=False, trace=False)
    order = model.order
    sarima_model = SARIMAX(train, order=order)
    model_fit = sarima_model.fit(disp=False)
    test_pred = model_fit.forecast(steps=len(test))
    future_pred = model_fit.forecast(steps=steps)
    return test_pred, future_pred

def exp_smoothing_forecast(train: pd.Series, test: pd.Series, steps: int) -> tuple:
    model = ExponentialSmoothing(train, trend='add')
    model_fit = model.fit()
    test_pred = model_fit.forecast(len(test))
    future_pred = model_fit.forecast(steps)
    return test_pred, future_pred

def ensemble_forecast(predictions: Dict[str, np.ndarray], test: pd.Series) -> tuple:
    valid_preds = {name: pred for name, pred in predictions.items() if pred is not None}
    if not valid_preds:
        return None, None
    weights = {}
    total_mae = 0
    for name, pred in valid_preds.items():
        mae = mean_absolute_error(test, pred)
        weights[name] = 1 / (mae + 1e-6)
        total_mae += weights[name]
    weights = {name: w / total_mae for name, w in weights.items()}
    ensemble_test = np.zeros_like(list(valid_preds.values())[0])
    for name, pred in valid_preds.items():
        ensemble_test += weights[name] * pred
    return ensemble_test, weights

def forecast_fdi(df: pd.DataFrame, country: str = 'Germany', steps: int = 3) -> Dict[str, Dict[str, float]]:
    logger.info(f"Time series analysis for {country}")
    if df.empty:
        logger.warning("Empty DataFrame provided for forecasting")
        return {model: {'mae': float('nan')} for model in ['ARIMA', 'Prophet', 'RandomForest', 'SARIMA', 'ExpSmoothing', 'Ensemble']}
    
    data = prepare_data(df, country)
    if data.empty:
        return {model: {'mae': float('nan')} for model in ['ARIMA', 'Prophet', 'RandomForest', 'SARIMA', 'ExpSmoothing', 'Ensemble']}
    
    if len(data) < 6:
        logger.warning(f"Insufficient data for {country} ({len(data)} rows). Skipping cross-validation.")
        return {model: {'mae': float('nan')} for model in ['ARIMA', 'Prophet', 'RandomForest', 'SARIMA', 'ExpSmoothing', 'Ensemble']}
    
    tscv = TimeSeriesSplit(n_splits=min(5, len(data) - 1))
    models = {
        'ARIMA': arima_forecast,
        'Prophet': prophet_forecast,
        'RandomForest': rf_forecast,
        'SARIMA': sarima_forecast,
        'ExpSmoothing': exp_smoothing_forecast
    }
    metrics = {}
    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        predictions, future_predictions = {}, {}
        for name, func in models.items():
            test_pred, future_pred = func(
                train['OBS_VALUE'] if name in ['ARIMA', 'SARIMA', 'ExpSmoothing'] else train,
                test['OBS_VALUE'] if name in ['ARIMA', 'SARIMA', 'ExpSmoothing'] else test,
                steps
            )
            predictions[name] = test_pred
            future_predictions[name] = future_pred
        
        ensemble_test, weights = ensemble_forecast(predictions, test['OBS_VALUE'])
        ensemble_future = ensemble_forecast({k: v for k, v in future_predictions.items()}, test['OBS_VALUE'])[0]
        
        for name, pred in predictions.items():
            if pred is not None:
                metrics.setdefault(name, []).append(mean_absolute_error(test['OBS_VALUE'], pred))
        if ensemble_test is not None:
            metrics.setdefault('Ensemble', []).append(mean_absolute_error(test['OBS_VALUE'], ensemble_test))
    
    avg_metrics = {name: {'mae': np.mean(maes)} for name, maes in metrics.items()}
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['OBS_VALUE'], mode='lines+markers', name='Historical'))
    future_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='Y')[1:]
    if ensemble_future is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=ensemble_future, mode='lines+markers', name='Ensemble Forecast'))
    fig.update_layout(title=f'FDI Forecast for {country}', xaxis_title='Date', yaxis_title='FDI Value')
    os.makedirs('plots', exist_ok=True)
    fig.write_html(f'plots/forecast_{country}.html')
    fig.show()
    logger.info(f"Forecast plot saved and displayed")
    
    return avg_metrics

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    from data_loading import load_data
    from data_preprocessing import preprocess_data
    df = load_data()
    df_clean = preprocess_data(df, {})
    metrics = forecast_fdi(df_clean, 'Germany')
    print("\nForecast Metrics:", metrics)