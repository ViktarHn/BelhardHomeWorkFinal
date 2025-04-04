import pandas as pd
import plotly.express as px
import logging
from data_loading import load_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('europe_vs_special.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def compare_investments():
    logger.info("Starting Europe vs Special countries comparison")
    df = load_data('estat_tec00107_filtered_en.csv')
    if df.empty:
        logger.warning("No data loaded")
        return
    
    # Фильтрация по странам и последним 10 годам
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
    df = df[df['TIME_PERIOD'].dt.year >= 2014]
    
    # Группы стран
    europe = df[df['geo'] == 'European Union - 27 countries (from 2020)']
    special = df[df['geo'].isin(['Cyprus', 'Luxembourg', 'Malta'])]
    
    # Средние значения
    europe_mean = europe['OBS_VALUE'].mean()
    special_mean = special.groupby('geo')['OBS_VALUE'].mean()
    data = pd.DataFrame({
        'Группа': ['Европа (EU-27)'] + special_mean.index.tolist(),
        'Средние инвестиции': [europe_mean] + special_mean.values.tolist()
    })
    
    # График
    fig = px.bar(data, x='Группа', y='Средние инвестиции', 
                 title='Сравнение средних инвестиций: Европа vs Кипр, Люксембург, Мальта (2014–2023)',
                 labels={'Средние инвестиции': 'Среднее значение инвестиций'})
    fig.write_html('europe_vs_special.html')
    fig.show()
    logger.info("Comparison plot saved and displayed")

if __name__ == "__main__":
    compare_investments()