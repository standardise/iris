from iris.engine.base import BaseEngine
from iris.foundation.types import ProblemType, ModelBlueprint, FeatureSchema, ModelMetrics
from iris.engine.models import LGBMRegressorModel, CatBoostRegressorModel, RidgeRegressorModel
from iris.feature_engineering.timeseries import TimeSeriesFeatureEngineer
import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class TimeSeriesEngine(BaseEngine):
    def __init__(self, date_col: str, id_col: str = None):
        super().__init__(task=ProblemType.TIME_SERIES_FORECASTING)
        self.date_col = date_col
        self.id_col = id_col
        self.ts_engineer = None

    def fit(self, dataset, time_limit=300):
        # 1. Feature Engineering
        logger.info("Starting Time Series Feature Engineering...")
        self.ts_engineer = TimeSeriesFeatureEngineer(self.date_col, self.id_col)
        
        # Transform data (adds date parts, handles cats)
        # Use dataset.df (full data) because dataset.features might exclude date_col
        df_processed = self.ts_engineer.fit_transform(dataset.df)
        
        # Combine back with target for splitting
        # df_processed contains features + date_col + target
        # We extract target directly from df_processed to ensure alignment
        y = df_processed[dataset.target_name]
        X = df_processed.drop(columns=[self.date_col, dataset.target_name])
        
        # Ensure consistent feature order for strict models like CatBoost
        X = X.reindex(sorted(X.columns), axis=1)
        
        # 2. Register Time Series Models
        # We use CatBoost and LGBM as they handle the categorical 'id_col' and date features best
        self.candidates = []
        
        # CatBoost is King for Time Series with Categories
        self.register_candidates([
            CatBoostRegressorModel("CatBoost_TS_Fast", mode="fast"),
            CatBoostRegressorModel("CatBoost_TS_Accurate", mode="accurate"),
            LGBMRegressorModel("LGBM_TS", mode="accurate"),
            RidgeRegressorModel("Linear_Trend") # Good for catching the global upward/downward trend
        ])
        
        # 3. Custom Train/Val Split for Time Series
        # We cannot do random split. We must do Time-Based Split.
        # Sort by date
        # Note: X and y are already aligned by index reset above
        
        # We need to sort by time_idx to split correctly
        full_data = pd.concat([X, y], axis=1).sort_values(by="time_idx")
        
        n_samples = len(full_data)
        train_size = int(n_samples * 0.85) # Last 15% for validation
        
        train_data = full_data.iloc[:train_size]
        val_data = full_data.iloc[train_size:]
        
        # Assumes target is last column after concat? No, let's be explicit.
        # But wait, pd.concat([X, y]) might have duplicate columns if y name is in X.
        # X shouldn't have target.
        # Let's extract carefully.
        
        target_name = dataset.target_name
        # If target name was not preserved in y series, pandas names it 0.
        if y.name is None: y.name = target_name
        
        X_train = train_data.drop(columns=[y.name])
        y_train = train_data[y.name]
        X_val = val_data.drop(columns=[y.name])
        y_val = val_data[y.name]
        
        self.feature_dtypes = X.dtypes.to_dict()
        
        logger.info(f"TRAIN COLS: {list(X_train.columns)}")
        logger.info(f"TS Engine started. Train: {len(X_train)}, Val: {len(X_val)}")
        
        # 4. Standard Fit Loop (Reuse BaseEngine logic)
        # We call the internal method we extracted/refactored in BaseEngine if available,
        # otherwise we manually do it.
        # Since I updated BaseEngine.fit to do everything, I can't easily inject X_train/X_val without refactoring BaseEngine to accept them.
        
        # Current BaseEngine.fit does the split internally.
        # To fix this cleanly: I will replicate the training loop here, BUT add the Stacking logic I just added to BaseEngine.
        
        start_time = time.time()
        
        # --- Training Loop (Replicated for TS Split) ---
        val_predictions = {}
        
        for i, model in enumerate(self.candidates):
            elapsed = time.time() - start_time
            remaining_total_time = time_limit - elapsed
            
            if remaining_total_time < 5:
                break
            
            models_left = len(self.candidates) - i
            current_budget = remaining_total_time / models_left
            
            try:
                model.fit(X_train, y_train, time_limit=int(current_budget)) 
                preds = model.predict(X_val)
                val_predictions[model.name] = preds
                self.trained_models[model.name] = model
            except Exception as e:
                logger.error(f"TS Model {model.name} failed: {e}")

        if not self.trained_models:
            raise RuntimeError("All TS models failed.")

        # Ensemble
        logger.info("Optimizing ensemble weights using SLSQP...")
        self.model_weights, weighted_score = self._optimize_weights(y_val, val_predictions)
        
        # --- Stacking Logic (Copied from BaseEngine) ---
        model_names = list(val_predictions.keys())
        first_pred = val_predictions[model_names[0]]
        # TS is Regression
        X_meta = np.column_stack([val_predictions[name] for name in model_names])
        
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        
        meta_score = float('inf')
        best_score = weighted_score
        
        try:
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_meta, y_val)
            meta_pred = self.meta_model.predict(X_meta)
            meta_score = np.sqrt(mean_squared_error(y_val, meta_pred))
            
            logger.info(f"Stacking Score: {meta_score:.4f} vs Weighted Score: {weighted_score:.4f}")
            
            if meta_score < weighted_score:
                self.stacking_active = True
                best_score = meta_score
                logger.info(">> Stacking Strategy WON. Using Meta-Learner.")
            else:
                self.stacking_active = False
                self.meta_model = None
                logger.info(">> Weighted Ensemble WON. Using SLSQP Weights.")
        except Exception as e:
            logger.warning(f"Stacking failed: {e}")
            self.stacking_active = False

        # Refit on Full Data
        logger.info("Refitting best models on full history...")
        
        target_models = self.trained_models.keys() if self.stacking_active else [n for n, w in self.model_weights.items() if w > 0]
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 5:
            budget = max(5, int(remaining / len(target_models)))
            for name in target_models:
                try:
                    self.trained_models[name].fit(X, y, time_limit=budget)
                except Exception as e:
                    logger.warning(f"Refit failed for {name}: {e}")

        return self._create_blueprint(dataset, best_score)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Preprocess
        df_processed = self.ts_engineer.transform(X)
        if self.date_col in df_processed.columns:
            df_processed = df_processed.drop(columns=[self.date_col])
            
        # Debug: Check columns
        if 'time_idx' not in df_processed.columns:
            print(f"DEBUG: time_idx missing! Cols: {df_processed.columns.tolist()}")
            
        # Ensure consistent feature order
        df_processed = df_processed.reindex(sorted(df_processed.columns), axis=1)
        
        # logger.info(f"PREDICT COLS: {list(df_processed.columns)}") # Commented out to avoid spam, enabled for debug
        print(f"DEBUG PREDICT COLS: {list(df_processed.columns)}")
            
        return super().predict(df_processed)