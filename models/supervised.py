from typing import Tuple, Dict
import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from iris.models.base import CandidateModel

try: 
    import lightgbm as lgb
except ImportError: 
    lgb = None
try: 
    import catboost as cb
except ImportError: 
    cb = None
try: 
    import shap
except ImportError: 
    shap = None

def get_preprocessor(X):
    """Creates a standard preprocessing pipeline for numerical and categorical data."""
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(exclude=['number']).columns
    
    transformers = [
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num_cols),
        ('cat', make_pipeline(
            SimpleImputer(strategy='constant', fill_value='missing'), 
            OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)
        ), cat_cols)
    ]
    return ColumnTransformer(transformers)

def get_smart_params(mode, n_samples, is_tight, model_type='lgb'):
    """
    Auto-tunes parameters based on dataset size and time constraints.
    """
    is_tiny = n_samples < 2000
    is_small = n_samples < 10000
    
    if model_type == 'lgb':
        if mode == 'fast' or is_tight:
            return {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
            
        if mode == 'accurate':
            if is_tiny:
                return {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 32}
            elif is_small:
                return {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 64}
            else:
                return {'n_estimators': 1000, 'learning_rate': 0.02, 'num_leaves': 128}
        
        else:
            if is_tiny: return {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31}
            return {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 64}

    elif model_type == 'cb':
        if mode == 'fast' or is_tight:
            return {'iterations': 100, 'depth': 4, 'learning_rate': 0.1}
            
        if mode == 'accurate':
            if is_tiny:
                return {'iterations': 500, 'depth': 5, 'learning_rate': 0.05}
            elif is_small:
                return {'iterations': 800, 'depth': 6, 'learning_rate': 0.03}
            else:
                return {'iterations': 1000, 'depth': 8, 'learning_rate': 0.02}
        
        else:
            if is_tiny: return {'iterations': 200, 'depth': 4, 'learning_rate': 0.05}
            return {'iterations': 500, 'depth': 6, 'learning_rate': 0.05}
            
    return {}

class RidgeRegressorModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        self._model = make_pipeline(get_preprocessor(X), Ridge(alpha=1.0))
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict(X)

class LogisticClassifierModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        self._model = make_pipeline(get_preprocessor(X), LogisticRegression(max_iter=1000))
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict_proba(X)

class RandomForestRegressorModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        n_est = 50 if (time_limit and time_limit < 15) else 100
        self._model = make_pipeline(get_preprocessor(X), RandomForestRegressor(n_estimators=n_est, n_jobs=-1))
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict(X)

class RandomForestClassifierModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        n_est = 50 if (time_limit and time_limit < 15) else 100
        self._model = make_pipeline(get_preprocessor(X), RandomForestClassifier(n_estimators=n_est, n_jobs=-1))
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict_proba(X)

class HistGBRegressorModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        is_tight = (time_limit and time_limit < 15)
        self._model = make_pipeline(
            get_preprocessor(X),
            HistGradientBoostingRegressor(max_iter=50 if is_tight else 200, random_state=42)
        )
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict(X)

class HistGBClassifierModel(CandidateModel):
    def fit(self, X, y, time_limit: int = None):
        is_tight = (time_limit and time_limit < 15)
        self._model = make_pipeline(
            get_preprocessor(X),
            HistGradientBoostingClassifier(max_iter=50 if is_tight else 200, random_state=42)
        )
        self._model.fit(X, y)
    def predict(self, X): return self._model.predict_proba(X)


class LGBMRegressorModel(CandidateModel):
    def __init__(self, name: str, mode: str = "balanced", params: dict = None):
        super().__init__(name, params)
        self.mode = mode
        self.best_iteration_ = None

    def fit(self, X, y, time_limit: int = None):
        if lgb is None: raise ImportError("LightGBM not installed")
        X_p = self._prep_cat(X)
        
        is_tight = (time_limit is not None) and (time_limit < 10)
        p = get_smart_params(self.mode, len(X), is_tight, 'lgb')
        
        self._model = lgb.LGBMRegressor(verbose=-1, n_jobs=-1, **p)

        if self.best_iteration_:
            self._model.set_params(n_estimators=self.best_iteration_)
            self._model.fit(X_p, y)
        else:
            try:
                test_size = 0.15 if len(X) < 2000 else 0.1
                X_t, X_v, y_t, y_v = train_test_split(X_p, y, test_size=test_size, random_state=42)
                
                es = 20 if (is_tight or len(X) < 2000) else 50
                
                self._model.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(es, verbose=False)])
                self.best_iteration_ = self._model.best_iteration_
            except ValueError:
                self._model.fit(X_p, y)

    def predict(self, X): return self._model.predict(self._prep_cat(X))
    def _prep_cat(self, X):
        X_p = X.copy()
        for c in X_p.select_dtypes(include='object').columns: X_p[c] = X_p[c].astype('category')
        return X_p
    def explain(self, X) -> Tuple[float, Dict[str, float]]:
        if shap:
            try:
                explainer = shap.TreeExplainer(self._model)
                vals = explainer.shap_values(self._prep_cat(X))[0]
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)): base = base[-1]
                return float(base), dict(zip(X.columns, vals))
            except: pass
        return 0.0, {}

class CatBoostRegressorModel(CandidateModel):
    def __init__(self, name: str, mode: str = "balanced", params: dict = None):
        super().__init__(name, params)
        self.mode = mode
        self.best_iteration_ = None

    def fit(self, X, y, time_limit: int = None):
        if cb is None: raise ImportError("CatBoost not installed")
        
        X_c = self._clean_cat_for_cb(X)
        cat_features = list(X_c.select_dtypes(include=['object', 'category']).columns)
        is_tight = (time_limit is not None) and (time_limit < 15)
        p_vals = get_smart_params(self.mode, len(X), is_tight, 'cb')
        
        params = {
            'verbose': 0, 'allow_writing_files': False, 'thread_count': -1,
            'cat_features': cat_features, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8,
            **p_vals
        }
        
        if self.best_iteration_:
            params['iterations'] = self.best_iteration_
            self._model = cb.CatBoostRegressor(**params)
            self._model.fit(X_c, y)
        else:
            es = 15 if (is_tight or len(X) < 2000) else 50
            params['early_stopping_rounds'] = es
            self._model = cb.CatBoostRegressor(**params)
            self._model.fit(X_c, y)
            try: self.best_iteration_ = self._model.get_best_iteration()
            except: self.best_iteration_ = self._model.tree_count_

    def predict(self, X): return self._model.predict(self._clean_cat_for_cb(X))
    
    def _clean_cat_for_cb(self, X):
        """Helper to fill NaNs in categorical columns with string placeholder"""
        X_c = X.copy()
        cat_cols = X_c.select_dtypes(include=['object', 'category']).columns
        for c in cat_cols:
            X_c[c] = X_c[c].astype(object).fillna("_MISSING_").astype(str)
        return X_c

    def explain(self, X) -> Tuple[float, Dict[str, float]]:
        if shap:
            try:
                explainer = shap.TreeExplainer(self._model)
                vals = explainer.shap_values(self._clean_cat_for_cb(X))[0]
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)): base = base[-1]
                return float(base), dict(zip(X.columns, vals))
            except: pass
        return 0.0, {}

class LGBMClassifierModel(CandidateModel):
    def __init__(self, name: str, mode: str = "balanced", params: dict = None):
        super().__init__(name, params)
        self.mode = mode
        self.best_iteration_ = None

    def fit(self, X, y, time_limit: int = None):
        if lgb is None: raise ImportError("LightGBM not installed")
        X_p = self._prep_cat(X)
        is_tight = (time_limit is not None) and (time_limit < 10)
        
        p = get_smart_params(self.mode, len(X), is_tight, 'lgb')
        
        self._model = lgb.LGBMClassifier(verbose=-1, n_jobs=-1, **p)

        if self.best_iteration_:
            self._model.set_params(n_estimators=self.best_iteration_)
            self._model.fit(X_p, y)
        else:
            try:
                test_size = 0.15 if len(X) < 2000 else 0.1
                X_t, X_v, y_t, y_v = train_test_split(X_p, y, test_size=test_size, random_state=42, stratify=y)
                es = 20 if (is_tight or len(X) < 2000) else 50
                self._model.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(es, verbose=False)])
                self.best_iteration_ = self._model.best_iteration_
            except ValueError:
                self._model.fit(X_p, y)

    def predict(self, X): return self._model.predict_proba(self._prep_cat(X))
    def _prep_cat(self, X):
        X_p = X.copy()
        for c in X_p.select_dtypes(include='object').columns: X_p[c] = X_p[c].astype('category')
        return X_p
    def explain(self, X): return 0.0, {}

class CatBoostClassifierModel(CandidateModel):
    def __init__(self, name: str, mode: str = "balanced", params: dict = None):
        super().__init__(name, params)
        self.mode = mode
        self.best_iteration_ = None

    def fit(self, X, y, time_limit: int = None):
        if cb is None: raise ImportError("CatBoost not installed")
        
        X_c = self._clean_cat_for_cb(X)
        cat_features = list(X_c.select_dtypes(include=['object', 'category']).columns)
        is_tight = (time_limit is not None) and (time_limit < 15)
        p_vals = get_smart_params(self.mode, len(X), is_tight, 'cb')
        
        params = {
            'verbose': 0, 'allow_writing_files': False, 'thread_count': -1,
            'cat_features': cat_features, 
            'auto_class_weights': 'Balanced',
            **p_vals
        }

        if self.best_iteration_:
            params['iterations'] = self.best_iteration_
            self._model = cb.CatBoostClassifier(**params)
            self._model.fit(X_c, y)
        else:
            es = 15 if (is_tight or len(X) < 2000) else 50
            params['early_stopping_rounds'] = es
            self._model = cb.CatBoostClassifier(**params)
            self._model.fit(X_c, y)
            try: self.best_iteration_ = self._model.get_best_iteration()
            except: self.best_iteration_ = self._model.tree_count_

    def predict(self, X): return self._model.predict_proba(self._clean_cat_for_cb(X))
    
    def _clean_cat_for_cb(self, X):
        X_c = X.copy()
        cat_cols = X_c.select_dtypes(include=['object', 'category']).columns
        for c in cat_cols:
            X_c[c] = X_c[c].astype(object).fillna("_MISSING_").astype(str)
        return X_c

    def explain(self, X) -> Tuple[float, Dict[str, float]]:
        if shap:
            try:
                explainer = shap.TreeExplainer(self._model)
                vals = explainer.shap_values(self._clean_cat_for_cb(X))
                if isinstance(vals, list): val = vals[1] if len(vals) > 1 else vals[0]
                else: val = vals
                
                if val.ndim == 2: val = val[0]
                
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)): base = base[-1]
                
                return float(base), dict(zip(X.columns, val))
            except: pass
        return 0.0, {}