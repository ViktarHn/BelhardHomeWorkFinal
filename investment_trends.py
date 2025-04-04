import pandas as pd
import plotly.express as px
import logging
from data_loading import load_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('investment_trends.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def plot_investment_trends():
    logger.info("Starting investment trends analysis")
    df = load_data('estat_tec00107_filtered_en.csv')
    if df.empty:
        logger.warning("No data loaded")
        return
    
    # Фильтрация по странам и последним 10 годам
    countries = ['Germany', 'Sweden', 'Italy', 'Spain', 'France']
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
    df = df[df['geo'].isin(countries) & (df['TIME_PERIOD'].dt.year >= 2014)]
    
    # Создание графика
    fig = px.line(df, x='TIME_PERIOD', y='OBS_VALUE', color='geo', 
                  title='Объемы инвестиций для Германии, Швеции, Италии, Испании, Франции (2014–2023)',
                  labels={'TIME_PERIOD': 'Год', 'OBS_VALUE': 'Значение инвестиций', 'geo': 'Страна'})
    fig.write_html('investment_trends.html')
    fig.show()
    logger.info("Investment trends plot saved and displayed")

if __name__ == "__main__":
    plot_investment_trends()