import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from iris.foundation.types import ProblemType

class AutoFeatureEngineer:
    """
    🛠️ Auto Feature Engineer:
    - Creates Interaction Features (A * B)
    - Performs Target Encoding (for high cardinality cats)
    - Selects Best Features (K-Best)
    """
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.feature_names_: List[str] = []
        self.target_encodings_: Dict[str, Dict[str, float]] = {}
        self.global_mean_: float = 0.0
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task: ProblemType) -> pd.DataFrame:
        X_new = X.copy()
        n_samples = len(X)
        
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2 and n_samples < 5000:
            X_new = self._add_interactions(X_new, num_cols[:5]) # Limit top 5 to avoid explosion
        
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if task in [ProblemType.REGRESSION, ProblemType.BINARY_CLASSIFICATION]:
             self.global_mean_ = y.mean()
             for col in cat_cols:
                 if X[col].nunique() > 10: # Only for high cardinality
                     encoded = self._fit_target_encode(X[col], y)
                     X_new[f'{col}_tenc'] = encoded

        if len(num_cols) > 1 and n_samples < 10000:
            X_new['num_sum'] = X[num_cols].sum(axis=1)
            X_new['num_mean'] = X[num_cols].mean(axis=1)
        
        if X_new.shape[1] > self.max_features:
            X_new = self._select_features(X_new, y, task)
        
        self.feature_names_ = X_new.columns.tolist()
        return X_new

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same transformations to new data."""
        X_new = X.copy()
        
        # 1. Re-apply Target Encoding
        for col, encoding_map in self.target_encodings_.items():
            if col in X.columns:
                X_new[f'{col}_tenc'] = X[col].map(encoding_map).fillna(self.global_mean_)
        
        # 2. Re-apply Stats
        num_cols_original = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'num_sum' in self.feature_names_:
             X_new['num_sum'] = X[num_cols_original].sum(axis=1)
        if 'num_mean' in self.feature_names_:
             X_new['num_mean'] = X[num_cols_original].mean(axis=1)

        # 3. Handle Missing / Extra Columns
        # Add missing cols with 0
        for col in self.feature_names_:
            if col not in X_new.columns:
                X_new[col] = 0
                
        return X_new[self.feature_names_]

    def _add_interactions(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for i in range(len(cols)):
            for j in range(i+1, min(i+3, len(cols))): 
                col_name = f'{cols[i]}_x_{cols[j]}'
                X[col_name] = X[cols[i]] * X[cols[j]]
        return X

    def _fit_target_encode(self, series: pd.Series, y: pd.Series) -> pd.Series:
        # Smoothing logic
        agg = pd.DataFrame({'cat': series, 'y': y}).groupby('cat')['y'].agg(['mean', 'count'])
        smoothing = 10
        agg['smooth'] = (agg['count'] * agg['mean'] + smoothing * self.global_mean_) / (agg['count'] + smoothing)
        self.target_encodings_[series.name] = agg['smooth'].to_dict()
        return series.map(agg['smooth']).fillna(self.global_mean_)

    def _select_features(self, X: pd.DataFrame, y: pd.Series, task: ProblemType) -> pd.DataFrame:
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        
        score_func = f_regression if task == ProblemType.REGRESSION else f_classif
        k = min(self.max_features, X_num.shape[1])
        
        try:
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X_num, y)
            selected_num = X_num.columns[selector.get_support()].tolist()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            return X[selected_num + cat_cols]
        except:
            return X