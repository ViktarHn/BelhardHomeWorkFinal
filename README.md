# Анализ данных FDI (Прямые иностранные инвестиции)

Этот проект представляет собой выполнение финального домашнего задания по курсу Data Science. Цель проекта — анализ данных об интенсивности прямых иностранных инвестиций (FDI) с использованием методов машинного обучения, включая загрузку данных, разведочный анализ (EDA), предобработку, кластеризацию, прогнозирование временных рядов, обнаружение аномалий и анализ важности признаков.

## Структура проекта

Проект состоит из 7 Python-скриптов, каждый из которых выполняет определенную задачу. Файлы предназначены для последовательного запуска в Jupyter Notebook, где результаты отображаются на экране и сохраняются в папке проекта.

### Файлы

1. **`data_loading.py`**
   - **Описание**: Загружает данные из CSV-файла `estat_tec00107_filtered_en.csv`, проверяет их структуру, обрабатывает конфиденциальные значения и выполняет базовые проверки качества.
   - **Выход**: DataFrame с загруженными данными.
   - **Вывод**: Первые строки, статистика и распределение по странам.

2. **`eda.py`**
   - **Описание**: Выполняет разведочный анализ данных (EDA): статистики, пропущенные значения, выбросы, распределения, временные тренды, корреляции.
   - **Выход**: JSON-файл с результатами (`eda_results.json`), текстовый отчет (`eda_summary.txt`), визуализации в папке `plots/`.
   - **Вывод**: Основные статистики, пропущенные значения, выбросы, рекомендации.

3. **`data_preprocessing.py`**
   - **Описание**: Предобрабатывает данные: удаляет ненужные столбцы, заполняет пропуски, создает новые признаки, масштабирует данные.
   - **Выход**: Обработанный CSV-файл (`processed_data.csv`), визуализации в папке `processed_data/plots/`.
   - **Вывод**: Первые строки обработанных данных, статистика, пропущенные значения.

4. **`clustering.py`**
   - **Описание**: Выполняет кластеризацию стран по FDI с использованием 5 методов (KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, GaussianMixture) и ансамбля через голосование.
   - **Выход**: CSV-файл с кластерами (`results/clusters.csv`), интерактивный график (`plots/clusters_ensemble.html`).
   - **Вывод**: Результаты кластеризации, средний Silhouette Score.

5. **`time_series.py`**
   - **Описание**: Прогнозирует временные ряды FDI для выбранной страны с использованием 5 методов (ARIMA, Prophet, RandomForest, LSTM, ExponentialSmoothing) и ансамбля через взвешенное среднее.
   - **Выход**: Интерактивный график прогноза (`plots/forecast_{country}.html`).
   - **Вывод**: Метрики качества (MAE) для каждой модели и ансамбля.

6. **`anomaly_detection.py`**
   - **Описание**: Обнаруживает аномалии в данных с использованием IsolationForest, LocalOutlierFactor и OneClassSVM.
   - **Выход**: DataFrame с метками аномалий (`results/anomalies_marked.csv`), график (`results/plots/anomalies_detection.png`).
   - **Вывод**: Анализ аномалий по методам.

7. **`feature_importance.py`**
   - **Описание**: Анализирует важность признаков с использованием RandomForest и permutation importance.
   - **Выход**: График важности (`feature_importance_results.png`).
   - **Вывод**: Таблица важности признаков.

## Установка

Для запуска проекта установите необходимые библиотеки:
```bash
pip install pandas numpy matplotlib seaborn plotly sklearn statsmodels prophet pmdarima tensorflow pyyaml tqdm scipy

## Использование
1. Поместите файл `estat_tec00107_filtered_en.csv` в корневую директорию проекта.
2. Откройте Jupyter Notebook:
```bash
jupyter notebook
3. Создайте новый ноутбук и добавьте ячейки для каждого файла в порядке их запуска:
`data_loading.py`
`eda.py`
`data_preprocessing.py`
`clustering.py`
`time_series.py`
`anomaly_detection.py`
`feature_importance.py`
4. Запустите каждую ячейку. Результаты отобразятся в ноутбуке и сохранятся в папках plots/, results/, processed_data/.
## Датасет
**Файл:** `estat_tec00107_filtered_en.csv`
**Описание:** Данные об интенсивности прямых иностранных инвестиций (FDI) по странам и годам.
**Колонки:**
geo: Код страны (категориальный).
TIME_PERIOD: Год (временной).
OBS_VALUE: Значение FDI (числовой).
CONF_STATUS: Статус конфиденциальности (категориальный, 'C' для конфиденциальных данных).
Результаты
EDA: Полный анализ данных с рекомендациями.
Кластеризация: Группы стран по характеристикам FDI.
Временные ряды: Прогноз FDI для Германии на 3 года вперед.
Аномалии: Выявленные аномальные значения в данных.
Важность признаков: Оценка влияния признаков на FDI.
Лицензия
Этот проект распространяется под лицензией MIT. См. файл LICENSE для подробностей.

Автор
Имя: VIKTAR
Контакт: minskmfi@gmail.com