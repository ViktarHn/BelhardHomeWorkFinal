import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.inspection import permutation_importance
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('feature_importance_analysis.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_params_ = None
        self.importance_df_ = None
        self.cv_results_ = None
    
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        from data_loading import load_data
        from data_preprocessing import preprocess_data
        logger.info(f"Loading data from {filepath}...")
        df = load_data(filepath)
        if df.empty:
            logger.warning("Empty DataFrame after loading")
            return df
        df = preprocess_data(df, {})
        logger.info(f"Preprocessed data: {len(df)} rows")
        return df
    
    def build_pipeline(self) -> Pipeline:
        numeric_features = ['year', 'rolling_mean', 'yearly_change']
        categorical_features = ['geo']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=self.random_state))
        ])
        return pipeline
    
    def optimize_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        if X.empty or y.empty:
            logger.warning("Empty data provided for model optimization")
            return pipeline
        if len(X) < self.n_splits:
            logger.warning(f"Insufficient data ({len(X)} rows) for {self.n_splits}-fold CV. Using single fit.")
            pipeline.fit(X, y)
            return pipeline
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=self.n_splits, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X, y)
        self.best_params_ = grid_search.best_params_
        logger.info(f"Best parameters: {self.best_params_}")
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if X.empty or y.empty:
            logger.warning("Empty data provided for model evaluation")
            return {'test_MAE': [float('nan')], 'test_R2': [float('nan')]}
        kfold = KFold(n_splits=min(self.n_splits, len(X)), shuffle=True, random_state=self.random_state)
        scoring = {'MAE': make_scorer(mean_absolute_error, greater_is_better=False), 'R2': make_scorer(r2_score)}
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1, return_train_score=True)
        self.cv_results_ = cv_results
        return cv_results
    
    def compute_feature_importance(self, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if X.empty or y.empty:
            logger.warning("Empty data provided for feature importance")
            return pd.DataFrame(columns=['feature', 'rf_importance', 'permutation_importance'])
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        importances = model.named_steps['model'].feature_importances_
        result = permutation_importance(model, X, y, n_repeats=20, random_state=self.random_state, n_jobs=-1)
        self.importance_df_ = pd.DataFrame({
            'feature': feature_names,
            'rf_importance': importances,
            'permutation_importance': result.importances_mean
        }).sort_values('rf_importance', ascending=False)
        return self.importance_df_
    
    def visualize_results(self, df: pd.DataFrame) -> None:
        if self.importance_df_ is None or self.importance_df_.empty:
            logger.warning("No feature importance data to visualize")
            return
        fig = px.bar(self.importance_df_.head(10), x='feature', y='rf_importance', title='Top 10 Feature Importances (Random Forest)')
        fig.write_html('feature_importance_rf.html')
        fig.show()
        fig = px.scatter(self.importance_df_, x='rf_importance', y='permutation_importance', text='feature', title='Feature Importance Comparison')
        fig.write_html('feature_importance_comparison.html')
        fig.show()
        fig = px.histogram(df, x='OBS_VALUE', nbins=30, title='Distribution of FDI Intensity')
        fig.write_html('fdi_distribution.html')
        fig.show()
        top_countries = df.groupby('geo')['OBS_VALUE'].mean().nlargest(10).reset_index()
        fig = px.bar(top_countries, x='geo', y='OBS_VALUE', title='Top 10 Countries by Average FDI Intensity')
        fig.write_html('top_countries.html')
        fig.show()
        logger.info("Visualizations saved and displayed")

if __name__ == "__main__":
    analyzer = FeatureImportanceAnalyzer()
    df = analyzer.load_and_preprocess('estat_tec00107_filtered_en.csv')
    if not df.empty:
        pipeline = analyzer.build_pipeline()
        pipeline = analyzer.optimize_model(pipeline, df, df['OBS_VALUE'])
        cv_results = analyzer.evaluate_model(pipeline, df, df['OBS_VALUE'])
        importance_df = analyzer.compute_feature_importance(pipeline, df, df['OBS_VALUE'])
        analyzer.visualize_results(df)
        print("\nFeature Importance (Top 15):")
        print(importance_df.head(15))
        print("\nModel Metrics:")
        print(f"Test MAE: {-cv_results['test_MAE'].mean():.3f}")
        print(f"Test R2: {cv_results['test_R2'].mean():.3f}")
    else:
        logger.warning("No data for feature importance analysis")